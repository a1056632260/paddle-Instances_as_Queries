import paddle
import paddle.nn as nn
from .heads import RoIAlign
from ppdet.modeling import ops

class SingleRoIExtractor(nn.Layer):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 layer=RoIAlign.__dict__
                ):
        super(SingleRoIExtractor, self).__init__()
        self.finest_scale = finest_scale
        self.roi_layers = RoIAlign(**layer)
        self.resolution=layer["resolution"]
        self.spatial_scale=layer["spatial_scale"]
        self.aligned=layer["aligned"]
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = paddle.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = paddle.floor(paddle.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clip(min=0, max=num_levels - 1).astype(paddle.int64)

        return target_lvls


    def forward(self, feats, rois,pro_list, roi_scale_factor=None):
        """Forward function."""
        out_size = (self.roi_layers.resolution,self.roi_layers.resolution)
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])

        roi_feats = paddle.zeros([rois.shape[0], self.out_channels, *out_size])


        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            num_rois=paddle.to_tensor(mask.shape)
            if inds.numel() > 0:
                rois_ = rois[inds]
                rois_=rois_[:,1:]


                roi_feats_t = ops.roi_align(
                feats[i],
                rois_,
                self.resolution,
                self.spatial_scale[i],
                rois_num=num_rois,
                aligned=self.aligned)

                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = paddle.stack((rois[:, 0], x1, y1, x2, y2), axis=-1)
        return new_rois
