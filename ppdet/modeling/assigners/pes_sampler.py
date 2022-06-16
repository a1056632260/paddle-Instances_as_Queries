import paddle




import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ppdet.core.workspace import register, serializable
from ..losses.iou_loss import GIoULoss
from ..transformers import bbox_cxcywh_to_xyxy


@register
class PseudoSampler(object):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        super(PseudoSampler).__init__()

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = paddle.nonzero(
            assign_result["gt_inds"] > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = paddle.nonzero(
            assign_result["gt_inds" ]== 0, as_tuple=False).squeeze(-1).unique()
        #gt_flags = bboxes.zeros(bboxes.shape[0], dtype=paddle.uint8)
        pos_assigned_gt_inds = paddle.gather(assign_result["gt_inds"],pos_inds )- 1
        pos_gt_bboxes = paddle.gather(gt_bboxes,paddle.to_tensor(pos_assigned_gt_inds))
        pos_gt_labels = paddle.gather(assign_result["labels"],pos_inds)
        sampling_result={}
        pos_assigned_gt_inds = paddle.gather(assign_result["gt_inds"],pos_inds) - 1
        sampling_result["pos_inds"]=pos_inds
        sampling_result["neg_inds"] = neg_inds
        pos_bboxes = paddle.gather(bboxes,pos_inds)
        neg_bboxes = paddle.gather(bboxes,neg_inds)
        sampling_result["pos_bboxes"]=pos_bboxes
        sampling_result["neg_bboxes"]=neg_bboxes
        sampling_result['pos_assigned_gt_inds']=pos_assigned_gt_inds
        sampling_result["pos_gt_bboxes"] = pos_gt_bboxes
        sampling_result["pos_gt_labels"] = pos_gt_labels
        return sampling_result
