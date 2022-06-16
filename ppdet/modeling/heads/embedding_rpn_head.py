
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from ppdet.core.workspace import register
from ..transformers import bbox_cxcywh_to_xyxy


@register
class EmbeddingRPNHead(nn.Layer):
    def __init__(self,num_proposals=100,proposal_feature_channel=256):
        super().__init__()
        self.num_proposal=num_proposals
        self.proposal_feature_channel=proposal_feature_channel
        self._init_layers()


    def _init_layers(self):
        #初始化建议框与特征
        self.init_proposal_bboxes=nn.Embedding(self.num_proposal,4,)
        self.init_proposal_features=nn.Embedding(self.num_proposal,self.proposal_feature_channel)

    def init_weight(self):
        super().init_weight()
        init_1=nn.initializer.Constant(0.5)
        init_2 = nn.initializer.Constant(1)
        init_1(self.init_proposal_bboxes.weight[:,:2])
        init_2(self.init_proposal_features.weight[:,2:])


    def _decode_init_proposals(self,body_feat,inputs):
        proposals=self.init_proposal_bboxes.weight.clone()
        imgs=body_feat
        proposals=bbox_cxcywh_to_xyxy(proposals)
        num_img=len(imgs[0])
        imgs_whwh=[]
        for i in range(num_img):
            h,w=inputs['im_shape'][i]
            img_whwh=paddle.concat([w,h,w,h])
            img_whwh=paddle.to_tensor([img_whwh])
            imgs_whwh.append(img_whwh)
        imgs_whwh=paddle.concat(imgs_whwh,axis=0)
        imgs_whwh=imgs_whwh[:,None,:]


        proposals=proposals*imgs_whwh

        init_proposal_features=self.init_proposal_features.weight.clone()
        feat_h,feat_w=init_proposal_features.shape
        init_proposal_features=init_proposal_features[None].expand((num_img, feat_h,feat_w))

        return proposals, init_proposal_features, imgs_whwh

    def forward_dummy(self,body_feat,inputs):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(body_feat,inputs)

    def forward_train(self,body_feat,inputs):
        """Forward function in training stage."""
        return self._decode_init_proposals(body_feat,inputs)

    def simple_test_rpn(self,body_feat, inputs):
        """Forward function in testing stage."""
        return self._decode_init_proposals(body_feat,inputs)

    def simple_test(self, inputs):
        """Forward function in testing stage."""
        raise NotImplementedError

    def aug_test_rpn(self, inputs):
        raise NotImplementedError(
            'EmbeddingRPNHead does not support test-time augmentation')


    def forward(self,body_feat,inputs):
        if self.training:
            proposals, init_proposal_features, imgs_whwh=self.forward_train(body_feat,inputs)
            return  proposals, init_proposal_features, imgs_whwh
        else:
            proposals, init_proposal_features, imgs_whwh=self.simple_test_rpn(body_feat,inputs)
            return proposals, init_proposal_features, imgs_whwh





