import paddle
import paddle.nn as nn
from paddle import distributed as dist
from ..losses import GIoULoss,FocalLoss,L1Loss
from ppdet.core.workspace import register, create
from ..initializer import xavier_uniform_
from .bbox_head import BBoxHead
import numpy as np
from ..coders import  DeltaBBoxCoder
from functools import partial



class DynamicConv(nn.Layer):
    """Implements Dynamic Convolution.

    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .

    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        act_cfg (dict): The activation config for DynamicConv.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 with_proj=True,
                 init_cfg=None):
        super(DynamicConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj

        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer =nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = nn.LayerNorm(self.feat_channels)
        self.norm_out = nn.LayerNorm(self.out_channels)

        self.activation = nn.ReLU()

        num_output = self.out_channels * input_feat_shape**2
        if self.with_proj:
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm =nn.LayerNorm(self.out_channels)

    def forward(self, param_feature, input_feature):
        """Forward function for `DynamicConv`.

        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, H, W).

        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        num_proposals = param_feature.shape[0]
        input_feature = input_feature.reshape([num_proposals, self.in_channels,
                                           self.input_feat_shape**2])
        input_feature=paddle.transpose(input_feature,perm=[2,0,1])
        input_feature = paddle.transpose(input_feature, perm=[1, 0, 2])
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].reshape([
            -1, self.in_channels, self.feat_channels])
        param_out = parameters[:, -self.num_params_out:].reshape([
            -1, self.feat_channels, self.out_channels])

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = paddle.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = paddle.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features



#
# class DynamicConv(nn.Layer):
#     def __init__(
#             self,
#             head_hidden_dim,
#             head_dim_dynamic,
#             head_num_dynamic, ):
#         super().__init__()
#
#         self.hidden_dim = head_hidden_dim
#         self.dim_dynamic = head_dim_dynamic
#         self.num_dynamic = head_num_dynamic
#         self.num_params = self.hidden_dim * self.dim_dynamic
#         self.dynamic_layer = nn.Linear(self.hidden_dim,
#                                        self.num_dynamic * self.num_params)
#
#         self.norm1 = nn.LayerNorm(self.dim_dynamic)
#         self.norm2 = nn.LayerNorm(self.hidden_dim)
#
#         self.activation = nn.ReLU()
#
#         pooler_resolution = 7
#         num_output = self.hidden_dim * pooler_resolution**2
#         self.out_layer = nn.Linear(num_output, self.hidden_dim)
#         self.norm3 = nn.LayerNorm(self.hidden_dim)
#
#     def forward(self, pro_features, roi_features):
#         '''
#         pro_features: (1,  N * nr_boxes, self.d_model)
#         roi_features: (49, N * nr_boxes, self.d_model)
#         '''
#         roi_features =paddle.reshape(roi_features,[-1, 256,
#                                            7 ** 2])
#         features = paddle.transpose(roi_features,perm=[1, 0, 2])
#         features=self.dynamic_layer(pro_features)
#         parameters =paddle.transpose(features ,perm=[1, 0, 2])
#
#         param1 = parameters[:, :, :self.num_params].reshape(
#             [-1, self.hidden_dim, self.dim_dynamic])
#         param2 = parameters[:, :, self.num_params:].reshape(
#             [-1, self.dim_dynamic, self.hidden_dim])
#
#         features = paddle.bmm(features, param1)
#         features = self.norm1(features)
#         features = self.activation(features)
#
#         features = paddle.bmm(features, param2)
#         features = self.norm2(features)
#         features = self.activation(features)
#
#         features = features.flatten(1)
#         features = self.out_layer(features)
#         features = self.norm3(features)
#         features = self.activation(features)
#
#         return features
def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
@register
class DIIHead(nn.Layer):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 dropout=0.0,
                feat_channels= 64,
                out_channels=256,
                input_feat_shape= 7,
                with_proj=True,
                 **kwargs):

        super(DIIHead, self).__init__()
        self.loss_iou =GIoULoss(loss_weight=2.0,reduction='mean')
        self.loss_cls=FocalLoss()
        self.loss_bbox=L1Loss()
        self.num_classes=num_classes
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.dropout=dropout
        self.attention = nn.MultiHeadAttention(in_channels, num_heads, dropout)
        self.attention_norm =nn.LayerNorm(in_channels)
        self. bbox_coder= DeltaBBoxCoder()
        self.instance_interactive_conv =DynamicConv(  in_channels=in_channels,
                 feat_channels=feat_channels,
                 out_channels=out_channels,
                 input_feat_shape=input_feat_shape,
                 with_proj=with_proj)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = nn.LayerNorm(in_channels)

        self.ffn =nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_channels,feedforward_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels,in_channels),
            nn.Dropout(dropout),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(in_channels)

        self.cls_fcs = nn.LayerList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels,bias_attr=False))
            self.cls_fcs.append(
               nn.LayerNorm(in_channels))
            self.cls_fcs.append(
               nn.ReLU())

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.reg_fcs = nn.LayerList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels,bias_attr=False))
            self.reg_fcs.append(
                nn.LayerNorm(in_channels))
            self.reg_fcs.append(
                nn.ReLU())
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(in_channels, 4)
        self.reg_class_agnostic=True
        self.reg_decoded_bbox=True


    # def init_weights(self):
    #     """Use xavier initialization for all weight parameter and set
    #     classification head bias as a specific value when use focal loss."""
    #     super(DIIHead, self).init_weights()
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #         else:
    #             # adopt the default initialization for
    #             # the weight and bias of the layer norm
    #             pass
    #     if self.loss_cls.use_sigmoid:
    #         bias_init = bias_init_with_prob(0.01)
    #         nn.init.constant_(self.fc_cls.bias, bias_init)


    def forward(self, roi_feat, proposal_feat):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = paddle.transpose(proposal_feat, perm=[1, 0, 2])
        proposal_feat=self.attention(proposal_feat)
        proposal_feat = self.attention_norm(proposal_feat)
        attn_feats = paddle.transpose(proposal_feat, perm=[1, 0, 2])

        # instance interactive

        proposal_feat =paddle.transpose(proposal_feat,perm=[1,0,2]).reshape([-1, self.in_channels])
        proposal_feat_iic =self.instance_interactive_conv(
            proposal_feat, roi_feat)


        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).reshape([N, num_proposals, -1])
        bbox_delta = self.fc_reg(reg_feat).reshape([N, num_proposals, -1])

        return cls_score, bbox_delta, obj_feat.reshape([N, num_proposals, -1]), attn_feats


#    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        """"Loss function of DIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)

        num_pos = paddle.cast(pos_inds.sum(),dtype=paddle.float32)
        avg_factor =paddle.clip(num_pos, min=1.).item()
        if avg_factor >1:
            reduction_override='mean'

        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    reduction="mean"
                    )
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.

            if pos_inds.any():

                pos_bbox_pred =bbox_pred[pos_inds]
                imgs_whwh=imgs_whwh.reshape([-1,4])
                imgs_whwh = imgs_whwh[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                   bbox_targets[pos_inds]/ imgs_whwh,
                    bbox_weights[pos_inds],
                    reduction_override= reduction_override)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                   bbox_targets[pos_inds],
                    bbox_weights[pos_inds]
                   )
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels,  weight):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.shape[0]
        num_neg = neg_bboxes.shape[0]
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = paddle.full((num_samples, ),
                                     self.num_classes,
                                     dtype=paddle.int64)
        label_weights = paddle.zeros([num_samples])
        bbox_targets = paddle.zeros([num_samples, 4])
        bbox_weights = paddle.zeros([num_samples, 4])
        if num_pos > 0:
            for i in range(num_pos):
                labels[pos_inds[i]] = pos_gt_labels[i]
                pos_weight =1 if  weight<0else weight
                label_weights[pos_inds[i]] = pos_weight
                if not self.reg_decoded_bbox:
                    pos_bbox_targets = self.bbox_coder.encode(
                        pos_bboxes, pos_gt_bboxes)
                else:
                    pos_bbox_targets = pos_gt_bboxes
                bbox_targets[pos_inds[i], :] = pos_bbox_targets[i]
                bbox_weights[pos_inds[i], :] = 1
        if num_neg > 0:
            for i in range(num_neg):
                label_weights[neg_inds[i]] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    weight,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res["pos_inds"] for res in sampling_results]
        neg_inds_list = [res["neg_inds"] for res in sampling_results]
        pos_bboxes_list = [res["pos_bboxes"] for res in sampling_results]
        neg_bboxes_list = [res["neg_bboxes"] for res in sampling_results]
        pos_gt_bboxes_list = [res["pos_gt_bboxes"] for res in sampling_results]
        pos_gt_labels_list = [res["pos_gt_labels"] for res in sampling_results]

        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            weight)
        if concat:
            labels = paddle.concat(labels, 0)
            label_weights = paddle.concat(label_weights, 0)
            bbox_targets = paddle.concat(bbox_targets, 0)
            bbox_weights =paddle.concat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids =paddle.to_tensor(rois[:, 0],dtype=paddle.int32).unique()
        assert img_ids.numel() <= len(img_metas["im_id"])

        bboxes_list = []
        for i in range(len(img_metas["im_id"])):
            inds = paddle.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(axis=1)
            num_rois = inds.numel()

            bboxes_ =paddle.gather(rois,inds)[:,1:]
            label_ = paddle.gather(labels,inds)
            bbox_pred_ = paddle.gather(bbox_preds,inds)
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_metas,i)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds=paddle.ones(num_rois)
            #keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.astype(paddle.bool)])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta,i):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.shape[1] == 4 or rois.shape[1] == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = paddle.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred =paddle.gather(bbox_pred, inds,1)
        assert bbox_pred.shape[1] == 4

        if rois.shape[1] == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['im_shape'][i])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['im_shape'][i])
            new_rois = paddle.concat((rois[:, [0]], bboxes), axis=1)

        return new_rois

def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.shape[0]== 0:
        accu = [paddle.to_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.shape[0] == target.shape[0]
    assert maxk <= pred.shape[1], \
        f'maxk {maxk} exceeds pred dimension {pred.shape[1]}'
    pred_value, pred_label = paddle.topk(pred,maxk,1)
    pred_label = paddle.t(pred_label ) # transpose to shape (maxk, N)
    correct = paddle.equal(pred_label,paddle.expand_as(target.reshape([1, -1]),pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct =paddle.t( correct & (pred_value > thresh))
    res = []
    for k in topk:
        correct_k = paddle.to_tensor(correct[:k].reshape([-1]),dtype=paddle.float32).sum(0, keepdim=True)
        res.append(paddle.multiply(correct_k,paddle.to_tensor(100.0 / pred.shape[0])))
    return res[0] if return_single else res
