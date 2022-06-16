# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ppdet.core.workspace import register, serializable
from ..losses.iou_loss import GIoULoss
from ..transformers import bbox_cxcywh_to_xyxy

__all__ = ['HungarianAssigner']


@register
@serializable
class HungarianAssigner(nn.Layer):
    __shared__ = ['use_focal_loss']

    def __init__(self,
                 matcher_coeff={'class': 2,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianAssigner, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self, boxes, logits, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]
        num_gts, num_bboxes = gt_bbox.shape[0], boxes.shape[0]
        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits) if self.use_focal_loss else F.softmax(logits)
        # [batch_size * num_queries, 4]
        out_bbox = boxes
        assigned_gt_inds=paddle.full([num_bboxes],-1,dtype=paddle.int64)

        assigned_labels =paddle.full([num_bboxes],-1,dtype=paddle.int64)
        # Also concat the target labels and boxes
        # tgt_ids = paddle.concat(gt_class).flatten()
        # tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = paddle.gather(
                pos_cost_class, gt_class, axis=1) - paddle.gather(
                    neg_cost_class, gt_class, axis=1)
        else:
            cost_class = -paddle.gather(out_prob, gt_class, axis=1)

        # Compute the L1 cost between boxes
        cost_bbox = (out_bbox.unsqueeze(1) - gt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(gt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        # C = C.reshape([bs, num_queries, -1])
        # C = [a.squeeze(0) for a in C.chunk(bs)]
        cost = C.detach().cpu()
        sizes = [a.shape[0] for a in gt_bbox]
        matched_row_inds, matched_col_inds= linear_sum_assignment(cost)

        matched_row_inds = paddle.to_tensor(matched_row_inds,place=
            boxes.place)
        matched_col_inds = paddle.to_tensor(matched_col_inds,place=
            boxes.place)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        gt_class=paddle.cast(gt_class,dtype=paddle.int64)
        # assign foregrounds based on matching results
        for i in range(num_gts):
            assigned_gt_inds[matched_row_inds[i]] = matched_col_inds[i] + 1
            assigned_labels[matched_row_inds[i]] = gt_class[matched_col_inds[i]]

        result={}
        result["num_gts"]=num_gts
        result["gt_inds"]=assigned_gt_inds
        result["labels"]=assigned_labels
        return result
# class BBoxL1Cost(object):
#     """BBoxL1Cost.
#
#      Args:
#          weight (int | float, optional): loss_weight
#          box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
#
#
#     """
#
#     def __init__(self, weight=1., box_format='xyxy'):
#         self.weight = weight
#         assert box_format in ['xyxy', 'xywh']
#         self.box_format = box_format
#
#     def __call__(self, bbox_pred, gt_bboxes):
#         """
#         Args:
#             bbox_pred (Tensor): Predicted boxes with normalized coordinates
#                 (cx, cy, w, h), which are all in range [0, 1]. Shape
#                 [num_query, 4].
#             gt_bboxes (Tensor): Ground truth boxes with normalized
#                 coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
#
#         Returns:
#             torch.Tensor: bbox_cost value with weight
#         """
#         if self.box_format == 'xywh':
#             gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
#         elif self.box_format == 'xyxy':
#             bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
#         bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
#         return bbox_cost * self.weight
class IoUCost(object):
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight


    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def bbox_overlaps(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clip(0)
        h_inter = (ykis2 - ykis1).clip(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps =self. bbox_overlaps(
            bboxes, gt_bboxes)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight

class ClassificationCost(object):
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -paddle.gather(cls_score, gt_labels,1)
        return cls_cost * self.weight