import paddle
import paddle.nn as nn






def dice_loss(pred, target):
    x =nn.functional.sigmoid(pred)
    eps = 1e-5
    n_inst = x.shape[0]
    x = x.reshape([n_inst, -1])
    target = target.reshape([n_inst, -1])
    intersection = (x * target).sum(axis=1)
    union = (x ** 2.0).sum(axis=1) + (target ** 2.0).sum(axis=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


class DiceLoss(nn.Layer):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * dice_loss(
            pred, target)
        if reduction == 'mean':
            loss = loss.mean() if loss.size > 0 else 0.0 * loss.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss
