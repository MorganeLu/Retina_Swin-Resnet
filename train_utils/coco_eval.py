import json
from collections import defaultdict

import numpy as np
import copy
import torch
import torch._six
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from .distributed_utils import all_gather, is_main_process


def merge2(img_ids, eval_results):
    """将多个进程之间的数据汇总在一起"""
    all_img_ids = all_gather(img_ids)
    all_eval_results = all_gather(eval_results)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_results = []
    for p in all_eval_results:
        merged_eval_results.extend(p)

    merged_img_ids = np.array(merged_img_ids)

    # keep only unique (and in sorted order) images
    # 去除重复的图片索引，多GPU训练时为了保证每个进程的训练图片数量相同，可能将一张图片分配给多个进程
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_results = [merged_eval_results[i] for i in idx]

    return list(merged_img_ids), merged_eval_results


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = None,
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.img_ids = []  # 记录每个进程处理图片的ids
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["bbox", "segm", "keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name

    def prepare_for_coco_detection(self, targets, outputs):
        """将预测的结果转换成COCOeval指定的格式，针对目标检测任务"""
        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            if img_id in self.img_ids:
                # 防止出现重复的数据
                continue
            self.img_ids.append(img_id)
            per_image_boxes = output["boxes"]
            # 对于coco_eval, 需要的每个box的数据格式为[x_min, y_min, w, h]
            # 而我们预测的box格式是[x_min, y_min, x_max, y_max]，所以需要转下格式
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()

            res_list = []
            # 遍历每个目标的信息
            for object_score, object_class, object_box in zip(
                    per_image_scores, per_image_classes, per_image_boxes):
                object_score = float(object_score)
                class_idx = int(object_class)
                if self.classes_mapping is not None:
                    class_idx = int(self.classes_mapping[str(class_idx)])
                # We recommend rounding coordinates to the nearest tenth of a pixel
                # to reduce resulting JSON file size.
                object_box = [round(b, 2) for b in object_box.tolist()]

                res = {"image_id": img_id,
                       "category_id": class_idx,
                       "bbox": object_box,
                       "score": round(object_score, 3)}
                # print(res["image_id"])
                # print(res["category_id"], res["bbox"])

                res_list.append(res)
            self.results.append(res_list)

    def prepare_for_coco_segmentation(self, targets, outputs):
        """将预测的结果转换成COCOeval指定的格式，针对实例分割任务"""
        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            if img_id in self.img_ids:
                # 防止出现重复的数据
                continue

            self.img_ids.append(img_id)
            per_image_masks = output["masks"]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()

            masks = per_image_masks > 0.5

            res_list = []
            # 遍历每个目标的信息
            for mask, label, score in zip(masks, per_image_classes, per_image_scores):
                rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")

                class_idx = int(label)
                if self.classes_mapping is not None:
                    class_idx = int(self.classes_mapping[str(class_idx)])

                res = {"image_id": img_id,
                       "category_id": class_idx,
                       "segmentation": rle,
                       "score": round(score, 3)}
                res_list.append(res)
            self.results.append(res_list)

    def update(self, targets, outputs):
        if self.iou_type == "bbox":
            self.prepare_for_coco_detection(targets, outputs)
        elif self.iou_type == "segm":
            self.prepare_for_coco_segmentation(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self):
        # 同步所有进程中的数据
        eval_ids, eval_results = merge2(self.img_ids, self.results)
        self.aggregation_results = {"img_ids": eval_ids, "results": eval_results}

        # 主进程上保存即可
        if is_main_process():
            results = []
            [results.extend(i) for i in eval_results]
            # write predict results into json file
            json_str = json.dumps(results, indent=4)
            with open(self.results_file_name, 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # 只在主进程上评估即可
        if is_main_process():
            # accumulate predictions from all images
            coco_true = self.coco
            coco_pre = coco_true.loadRes(self.results_file_name)

            self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)

            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()
            print(f"IoU metric: {self.iou_type}")
            self.coco_evaluator.summarize()

            coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
            return coco_info
        else:
            return None


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions

def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x2 - x1) * (y2 - y1)
            ann['id'] = id + 1
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
