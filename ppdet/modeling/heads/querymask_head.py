import paddle
import paddle.nn as nn
from ..losses import DiceLoss
import numpy as np
from ppdet.core.workspace import register, create
import os
from .roi_extractor import RoIAlign
from ..ops import  roi_align
from ..proposal_generator import MaskAssigner,polygons_to_mask

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
        self.dynamic_layer = nn.Linear(
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


@register
class DynamicMaskHead(nn.Layer):
    __shared__=['num_classes']

    def __init__(self,
                 num_classes=80,
                 dropout=0.,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=80,
                 class_agnostic=False,
                 scale_factors=2,
                 mask_resolution=14,
                 feat_channels=64,
                 out_channels=256,
                 input_feat_shape=7,
                 with_proj=True,
                 upsample_method="deconv",
                 **kwargs):
        super(DynamicMaskHead, self).__init__()
        self.num_convs=num_convs
        self.fp16_enabled = False
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.conv_out_channels=conv_out_channels
        self.instance_interactive_conv =DynamicConv( in_channels=in_channels,
                 feat_channels=feat_channels,
                 out_channels=out_channels,
                 input_feat_shape=input_feat_shape,
                 with_proj=with_proj)
        self.convs = nn.LayerList()
        self.conv_kernel_size=conv_kernel_size
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.extend(
                [nn.Conv2D(in_channels,conv_out_channels,kernel_size=conv_kernel_size,padding=padding,bias_attr=False),
                nn.BatchNorm(conv_out_channels),
                nn.ReLU()]
                )

        upsample_in_channels = (
        conv_out_channels if num_convs > 0 else in_channels)
        self.upsample=nn.Conv2DTranspose(upsample_in_channels,conv_out_channels,kernel_size=scale_factors,stride=scale_factors)

        self.conv_logits=nn.Conv2D(conv_out_channels,num_classes,1,1)
        self.relu=nn.ReLU()
        self.loss_mask=DiceLoss()
        self.mask_assigner=MaskAssigner(num_classes=num_classes,mask_resolution=mask_resolution)
        self.upsample_method=upsample_method
        self.class_agnostic=False
    # def init_weights(self):
    #     """Use xavier initialization for all weight parameter and set
    #     classification head bias as a specific value when use focal loss."""
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #         nn.init.constant_(self.conv_logits.bias, 0.)

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

        proposal_feat = proposal_feat.reshape([-1, self.in_channels])
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)

        x = paddle.transpose(proposal_feat_iic,perm=[0,2,1]).reshape(roi_feat.shape)

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)

        return mask_pred


    def loss(self, mask_pred, mask_targets, labels):
        num_pos = paddle.ones(labels.shape,dtype=paddle.float32).sum()
        #avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
        loss = dict()
        mask_target_list=[]
        if mask_pred.shape[0] == 0:
            loss_mask = mask_pred.sum()
        else:
            for i in range(num_pos):
                label_index=labels[i]
                mask_p=mask_pred[i, label_index,:,:]
                mask_target_list.append(mask_p)

            mask_pred=paddle.to_tensor(mask_target_list)
            mask_pred=paddle.squeeze(mask_pred,1)

            loss_mask = self.loss_mask(mask_pred,
                                       mask_targets,
                                       )
        loss['loss_mask'] = loss_mask
        return loss

    def get_targets(self,
                    sampling_results,
                    gt_masks,
                   mask_size,
                    cfg):

        pos_proposals = [res["pos_bboxes"] for res in sampling_results]
        pos_assigned_gt_inds = [
            res["pos_assigned_gt_inds"] for res in sampling_results
        ]
        img_wh=cfg["im_shape"]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, mask_size,img_wh)
        return mask_targets

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels,
                      ori_shape, scale_factor, rescale, format=True):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(float | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.



        """
        if isinstance(mask_pred, paddle.Tensor):
            mask_pred = nn.functional.sigmoid(mask_pred)
        else:
            mask_pred = paddle.empty([mask_pred])

        place = mask_pred.place
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        # No need to consider rescale and scale_factor while exporting to ONNX

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(
                    np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0

            if not isinstance(scale_factor, (float, paddle.Tensor)):
                scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes / scale_factor

        # support exporting to ONNX
        # if torch.onnx.is_in_onnx_export():
        #     threshold = rcnn_test_cfg.mask_thr_binary
        #     if not self.class_agnostic:
        #         box_inds = torch.arange(mask_pred.shape[0])
        #         mask_pred = mask_pred[box_inds, labels][:, None]
        #     masks, _ = _do_paste_mask(
        #         mask_pred, bboxes, img_h, img_w, skip_empty=False)
        #     if threshold >= 0:
        #         masks = (masks >= threshold).to(dtype=torch.bool)
        #     else:
        #         # TensorRT backend does not have data type of uint8
        #         is_trt_backend = os.environ.get(
        #             'ONNX_BACKEND') == 'MMCVTensorRT'
        #         target_dtype = torch.int32 if is_trt_backend else torch.uint8
        #         masks = (masks * 255).to(dtype=target_dtype)
        #     return masks

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if place == paddle.CPUPlace():
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h.cpu().numpy() * img_w.cpu().numpy() * 4/(1024**3)))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = paddle.chunk(paddle.arange(N), num_chunks)
        chunks=paddle.to_tensor(chunks,place=place)

        threshold = 0.5
        im_mask = paddle.zeros(
            [N,img_h,img_w],
            dtype=paddle.bool if threshold >= 0 else paddle.uint8)
        im_mask=paddle.to_tensor(im_mask,place=place)
        mask_list=[]
        if not self.class_agnostic:
            for i in range(N):
                mask_pred_i= mask_pred[i, labels[i]]
                mask_list.append(mask_pred_i)
            mask_pred=paddle.to_tensor(mask_list)
            mask_pred=mask_pred.unsqueeze(1)


        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=place == paddle.CPUPlace())

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold)
                masks_chunk=paddle.to_tensor(masks_chunk,dtype=paddle.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255)
                masks_chunk = paddle.to_tensor(masks_chunk, dtype=paddle.uint8)
            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms if format else im_mask


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    place = masks.place
    if skip_empty:
        x0_int, y0_int = paddle.clip(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0)
        x1_int = paddle.clip(
            boxes[:, 2].max().ceil() + 1, max=img_w)
        x1_int=paddle.to_tensor(x1_int,dtype=paddle.int32)
        y1_int = paddle.clip(
            boxes[:, 3].max().ceil() + 1, max=img_h)
        y1_int = paddle.to_tensor(y1_int, dtype=paddle.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = paddle.split(boxes,4, axis=1)  # each is Nx1

    N = masks.shape[0]

    img_y = paddle.arange(y0_int, y1_int)
    img_x = paddle.arange(x0_int, x1_int)
    img_x=paddle.to_tensor(img_x,place=place,dtype=paddle.float32)+0.5
    img_y = paddle.to_tensor(img_y, place=place, dtype=paddle.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0

    if paddle.isinf(img_x).any():
        a=paddle.zeros_like(img_x)
        inds=paddle.where(paddle.isinf(img_x),a,img_x)
        #inds = paddle.where(paddle.isinf(img_x))
        img_x=inds

    if paddle.isinf(img_y).any():
        a = paddle.zeros_like(img_y)
        inds = paddle.where(paddle.isinf(img_y), a, img_y)
        img_y=inds
        # inds = paddle.where(paddle.isinf(img_x))
        # inds =paddle.where(paddle.isinf(img_y))
        # img_y[inds] = 0

    gx = img_x[:, None, :].expand([N, img_y.shape[1], img_x.shape[1]])
    gy = img_y[:, :, None].expand([N, img_y.shape[1], img_x.shape[1]])
    grid = paddle.stack([gx, gy], axis=3)
    masks=paddle.to_tensor(masks,dtype=paddle.float32)
    img_masks = nn.functional.grid_sample(
        masks, grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,mask_szie,
                cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.
    """

    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list,mask_szie,cfg)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = paddle.concat(mask_targets)
        mask_targets=paddle.squeeze(mask_targets,1)
    return mask_targets



def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks,mask_size,cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.

    Example:

    """
    place = pos_proposals.place
    mask_size = (mask_size,mask_size)
    num_pos = pos_proposals.shape[0]
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw =cfg[0], cfg[1]
        maxh=maxh.cpu().numpy()
        maxw=maxw.cpu().numpy()
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        mask_targets =crop_and_resize(
            proposals_np, gt_masks,mask_size,
            inds=pos_assigned_gt_inds ,device=paddle.CPUPlace(),maxw=maxw,maxh=maxh)

        mask_targets = paddle.to_tensor(mask_targets,dtype=paddle.float32)
    else:
        mask_targets = paddle.zeros((0, ) + mask_size)


    return mask_targets


def crop_and_resize(bboxes,
                    masks,
                    out_shape,
                    inds,
                    device=paddle.CPUPlace(),
                    interpolation='bilinear',
                    maxw=0,maxh=0):
    """See :func:`BaseInstanceMasks.crop_and_resize`."""
    if len(masks) == 0:
        empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
        return empty_masks

    # convert bboxes to tensor
    if isinstance(bboxes, np.ndarray):
        bboxes =paddle.to_tensor(bboxes,place=device)
    if isinstance(inds, np.ndarray):
        inds = paddle.to_tensor(inds,place=device)
    mask_list=[]
    poly_num=len(masks)
    for i in range(poly_num):
        # poly_len=len(masks[i][0])
        # mask1=np.array(masks[i][0])[0::2]
        # #mask1=mask1.reshape((poly_len//2,1))
        # mask2=np.array(masks[i][0])[1::2]
        # #mask2 = mask2.reshape((poly_len // 2, 1))
        # mask=np.stack((mask1,mask2),axis=1)
        # mask_list.append(mask)

        masks_=polygons_to_mask(masks[i],maxh,maxw)
        masks_=paddle.to_tensor(masks_)
        masks_=paddle.unsqueeze(masks_,0)

        mask_list.append(masks_)

  # masks = np.stack(masks).reshape(-1, height, width)
    num_bbox = bboxes.shape[0]
    fake_inds = paddle.arange(num_bbox)
    fake_inds=paddle.to_tensor(fake_inds,dtype=bboxes.dtype,place=device)[:, None]
    rois = paddle.concat([fake_inds, bboxes], axis=1)  # Nx5
    rois = paddle.to_tensor(rois,place=device)
    targets_list=[]
    if num_bbox > 0:
        for i in range (num_bbox):
            num_bbox=paddle.to_tensor(num_bbox)
            gt_masks_th = paddle.to_tensor(mask_list,dtype=rois.dtype,place=device)
            gt_masks_th=paddle.index_select(gt_masks_th,inds,0)

            gt_masks_i=paddle.unsqueeze(gt_masks_th[i],0)
            roi=bboxes[i]
            targets = roi_align(gt_masks_i, roi[None,:], out_shape,
                            1.0, 0, paddle.ones([1],dtype=paddle.int32)).squeeze(1)
            targets_list.append(targets)
        targets_list=paddle.to_tensor(targets_list)
        resized_masks = (targets_list >= 0.5).cpu().numpy()
    else:
        resized_masks = []
    return resized_masks
