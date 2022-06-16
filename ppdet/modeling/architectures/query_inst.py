from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ppdet.modeling.heads.roi_extractor import RoIAlign
from ..transformers import bbox_xyxy_to_cxcywh
from .. bbox_utils import bbox2roi,bbox2result
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle
import paddle.nn as nn
from ..assigners import HungarianAssigner,PseudoSampler
from ..proposal_generator import MaskAssigner
from ..SingleRoIExtractor import SingleRoIExtractor

@register
class QueryInst(BaseArch):
    r"""Implementation of `QueryInst: Parallelly Supervised Mask Query for
     Instance Segmentation <https://arxiv.org/abs/2105.01928>`, based on
     SparseRCNN detector. """
    __category__ = 'architecture'

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 neck=None,
                 mask_head=None,
                 mask_post_process=None,
                 num_stage=6,

                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=RoIAlign.__dict__,
                 mask_roi_extractor=RoIAlign.__dict__,
                 pos_weight=[1,1,1,1,1,1],
                 mask_size=[28,28,28,28,28,28],
                 num_classes=80,
                 mask_resolution=28,
                 num_proposals=100,
                 bbox_roi_out_c=256,
                 bb__roi_f_S=[4, 8, 16, 32],

    ):
        super(QueryInst, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head=nn.LayerList()
        self.mask_head = nn.LayerList()
        for i in  range (num_stage):
            self.bbox_head.append(bbox_head)
            self.mask_head.append(mask_head)
        self.neck = neck

        self.mask_post_process = mask_post_process
        self.with_mask = mask_head is not None
        self.num_stage=num_stage
        self.bbox_roi_extractor= RoIAlign(**bbox_roi_extractor)
        #self.bbox_roi_extractor=SingleRoIExtractor(out_channels= bbox_roi_out_c,featmap_strides=bb__roi_f_S,layer=bbox_roi_extractor)
        self.mask_roi_extractor=RoIAlign(**mask_roi_extractor)
        #self.mask_roi_extractor = SingleRoIExtractor(out_channels=bbox_roi_out_c, featmap_strides=bb__roi_f_S,layer=mask_roi_extractor)
        self.stage_loss_weights=stage_loss_weights
        self.bbox_assigner=HungarianAssigner()
        self.bbox_sampler=PseudoSampler()
        self.pos_weight=pos_weight
        self.mask_size=mask_size
        self.mask_asssigner=MaskAssigner(num_classes=num_classes,mask_resolution=mask_resolution)
        self.num_proposals=num_proposals
    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)
        out_shape = neck and out_shape or bbox_head.get_head().out_shape
        kwargs = {'input_shape': out_shape}
        mask_head = cfg['mask_head'] and create(cfg['mask_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
            "mask_head": mask_head,
        }

    def forward_train(self, **kwargs):
        """

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #input:['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd']

        num_imgs=len(self.inputs["im_id"])

        all_stage_bbox_results = []
        x = self.backbone(self.inputs)
        x=self.neck(x)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, self.inputs)
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        num_proposals = proposal_boxes.shape[1]
        imgs_whwh = paddle.tile(imgs_whwh,[1, num_proposals, 1])
        object_feats = proposal_features
        all_stage_loss = {}
        for stage in range (self.num_stage):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage,x, rois,proposal_list, object_feats,
                                             )
            all_stage_bbox_results.append(bbox_results)
            # if gt_bboxes_ignore is None:
            #     # TODO support ignore
            #     gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(num_imgs):
                num_gts=len(self.inputs["gt_bbox"])
                normolize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /imgs_whwh[i])
                assign_result = self.bbox_assigner(normolize_bbox_ccwh, cls_pred_list[i], self.inputs["gt_bbox"][i],self.inputs["gt_class"][i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result, proposal_list[i], self.inputs["gt_bbox"][i])
                sampling_results.append(sampling_result)
            weight = paddle.ones([num_gts]) * self.pos_weight[stage]
            bbox_targets = self.bbox_head[i].get_targets(
                sampling_results, self.inputs["gt_bbox"][i], self.inputs["gt_class"][i],weight,
                True)

            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            object_feats = bbox_results['object_feats']

            single_stage_loss = self.bbox_head[i].loss(
                cls_score.reshape([-1, cls_score.shape[-1]]),
                decode_bbox_pred.reshape([-1, 4]),
                *bbox_targets,
                imgs_whwh=imgs_whwh)

            if self.with_mask:
                mask_results = self._mask_forward_train( stage,x, bbox_results['attn_feats'],
                                                        sampling_results, self.inputs["gt_poly"],self.mask_size[stage])
                single_stage_loss['loss_mask'] = mask_results['loss_mask']

            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                                        self.stage_loss_weights[i]

        return all_stage_loss

    def _bbox_forward(self, stage,x, rois,proposal_list, object_feats):
        '''Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].'''
        num_imgs = len(self.inputs["im_id"])
        img_metas=self.inputs
        num_proposal=paddle.to_tensor(len(proposal_list[0]))
        roi_nums=paddle.tile(num_proposal,paddle.to_tensor(num_imgs))

        bbox_feats =self.bbox_roi_extractor(x,proposal_list,roi_nums)
        #bbox_feats = self.bbox_roi_extractor(x, rois,proposal_list)
        cls_score, bbox_pred, object_feats, attn_feats = self.bbox_head[stage](bbox_feats,
                                                                   object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            paddle.zeros([len(rois)]),  # dummy arg
            bbox_pred.reshape([-1, bbox_pred.shape[-1]]),
            [paddle.zeros([object_feats.shape[1]]) for _ in range(num_imgs)],
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=paddle.concat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _mask_forward(self, stage, x, rois,bb_list, attn_feats):
        num_imgs = len(self.inputs["im_id"])
        img_metas = self.inputs
        num_proposal=[]
        for i in range (num_imgs):
            num_proposal.append(len(bb_list[i]))
        num_proposal = paddle.to_tensor(num_proposal)


        mask_feats = self.mask_roi_extractor( x,bb_list,num_proposal)
        #mask_feats = self.mask_roi_extractor(x, rois,bb_list)
        # do not support caffe_c4 model anymore
        mask_pred = self.mask_head[stage](mask_feats, attn_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage,x, attn_feats, sampling_results, gt_masks,mask_szie):

        if sum([len(gt_mask) for gt_mask in gt_masks]) == 0:
            print('Ground Truth Not Found!')
            loss_mask = sum([_.sum() for _ in self.mask_head.parameters()]) * 0.
            return dict(loss_mask=loss_mask)
        pos_rois = bbox2roi([res["pos_bboxes"] for res in sampling_results])
        pos_list=[res["pos_bboxes"] for res in sampling_results]
        feats_list=[]
        for (feats, res) in zip(attn_feats, sampling_results):
            if len(res["pos_inds"])==1:
                feats=paddle.unsqueeze(feats[res["pos_inds"]],0)
            else:
                feats=feats[res["pos_inds"]]
            feats_list.append(feats)
        attn_feats =paddle.concat(feats_list)

        mask_results = self._mask_forward(stage,x, pos_rois,pos_list, attn_feats)
        mask_size_list=[]
        tgt_gt_inds=[sampling_result["pos_assigned_gt_inds"]for sampling_result in sampling_results ]
        tgt_gt_label = [sampling_result["pos_gt_labels"] for sampling_result in sampling_results]
        for i in range(len(gt_masks)):
            mask_size_list.append(mask_szie)
        mask_targets = self.mask_head[stage].get_targets(sampling_results, gt_masks,mask_size_list,self.inputs)
        #rois, rois_num, tgt_classes, mask_targets, mask_index, tgt_weights=self.mask_asssigner(pos_list, tgt_gt_inds, tgt_gt_label,self.inputs)
        pos_labels =  paddle.concat([res['pos_gt_labels'] for res in sampling_results])

        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def simple_test(self):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        rescale=True
        img=self.inputs["image"]
        x = self.backbone(self.inputs)
        x=self.neck(x)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, self.inputs)
        num_imgs = len(self.inputs["im_id"])
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple(each for each in self.inputs['im_shape'])
        #scale_factors = tuple(meta['scale_factor'] )
        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}

        object_feats = proposal_features
        for stage in range(self.num_stage):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, proposal_list, object_feats,
                                              )
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

            if self.with_mask:
                rois = bbox2roi(proposal_list)
                mask_results = self._mask_forward(stage, x, rois, proposal_list,bbox_results['attn_feats'])
                mask_results['mask_pred'] = mask_results['mask_pred'].reshape([
                    num_imgs, -1, *mask_results['mask_pred'].shape[1:]]
                )

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = nn.functional.sigmoid(cls_score)
        else:
            cls_score =nn.functional.softmax(cls_score)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = paddle.topk(cls_score_per_img.flatten(
                0, 1),self.num_proposals, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                      num_classes]
            if rescale:
                scale_factors = self.inputs['scale_factor'][img_id]
                scale_factors=paddle.concat([scale_factors,scale_factors],axis=0)
                bbox_pred_per_img /= scale_factors
            det_bboxes.append(
                paddle.concat([bbox_pred_per_img, scores_per_img[:, None]], axis=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    paddle.to_tensor(scale_factor).cuda()
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_imgs):
                mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                mask_pred_per_img = mask_pred_per_img[:, None, ...].tile([1, num_classes, 1, 1])
                segm_result = self.mask_head[-1].get_seg_masks(
                    mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                     ori_shapes[img_id], scale_factors[img_id],
                    rescale)
                segm_results.append(segm_result)

            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.aug_test_rpn(x, img_metas)
        results = self.roi_head.aug_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            aug_imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs

    def _forward(self):
        pass

    def get_loss(self,):
        all_stage_loss = self.forward_train()
        loss = {}
        loss_list=list(all_stage_loss.values())
        total_loss = paddle.add_n(list(all_stage_loss.values()))
        loss.update({'loss': total_loss})
        return loss
    def get_pred(self,):

        pred=self.simple_test()

        return pred

