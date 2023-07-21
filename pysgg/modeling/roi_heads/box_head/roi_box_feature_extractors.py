# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from pysgg.modeling import registry
from pysgg.modeling.backbone import resnet
from pysgg.modeling.make_layers import group_norm
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.poolers import Pooler


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

        if cfg.MODEL.RELATION_ON:
            # for the following relation head, the features need to be flattened
            pooling_size = 2
            self.adptive_pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
            input_size = self.out_channels * pooling_size ** 2
            representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
            use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

            if half_out:
                out_dim = int(representation_size / 2)
            else:
                out_dim = representation_size

            self.fc7 = make_fc(input_size, out_dim, use_gn)
            self.resize_channels = input_size
            self.flatten_out_channels = out_dim

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x

    def forward_without_pool(self, x):
        x = self.head(x)
        return self.flatten_roi_features(x)

    def flatten_roi_features(self, x):
        x = self.adptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc7(x))
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("VETOFeatureExtractor")
class VETOFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        super(VETOFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION #different pool size
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        """
        pooler2 = Pooler(
            output_size=(4, 4),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        """
        self.pooler = pooler
        #self.pooler2 = pooler2
        self.out_channels = 256

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM #cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        out_dim = representation_size
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

        #self.fc6_d = make_fc(4096, representation_size, use_gn) #make_fc(12544, representation_size, use_gn)
        #self.fc7_d = make_fc(representation_size, out_dim//2, use_gn)

    def forward(self, x, proposals, depth_features=None):
        d = None
        x_2d = None
        d_2d = None
        x_1d = None
        d_1d = None
        if depth_features is not None:
            x_2d, d_2d = self.pooler(x, proposals, depth_features=depth_features)

            #x, d = self.pooler2(x, proposals, depth_features=depth_features)
        else:
            x = self.pooler(x, proposals, depth_features=depth_features)
        #x_1d = x.view(x.size(0), -1)
        #x_1d = F.relu(self.fc6(x_1d))
        #x_1d = F.relu(self.fc7(x_1d))
        #d_1d = d.view(d.size(0), -1)
        #d_1d = F.relu(self.fc6(d_1d))
        #d_1d = F.relu(self.fc7(d_1d))

        return x_2d, d_2d, x_1d, d_1d

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x



@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size

        self.fc7 = make_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, x, proposals, depth_features=None):
        d = None
        if depth_features is not None:
            x, d = self.pooler(x, proposals, depth_features=depth_features)
        else:
            x = self.pooler(x, proposals, depth_features=depth_features)
        if d is None:
            x = x.view(x.size(0), -1)

            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
        return x, d

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_depth")
class FPN2MLPFeatureExtractor_depth(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        super(FPN2MLPFeatureExtractor_depth, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.dfc6 = make_fc(input_size, representation_size, use_gn)
        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size

        self.fc7 = make_fc(representation_size, 4096, use_gn)
        self.dfc7 = make_fc(representation_size, 4096, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, x, proposals, depth_features=None):
        d = None
        if depth_features is not None:
            x, d = self.pooler(x, proposals, depth_features=depth_features)
        else:
            x = self.pooler(x, proposals, depth_features=depth_features)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        if d is not None:
            d = d.view(d.size(0), -1)

            d = F.relu(self.dfc6(d))
            d = F.relu(self.dfc7(d))

        return x, d, None , None

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
    if for_relation and cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR_MINI is not None and cfg.DATASETS.USE_DEPTH:
        feature_extractor = cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR_MINI
    else:
        feature_extractor = cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        feature_extractor
    ]
    return func(cfg, in_channels, half_out, cat_all_levels, for_relation)
