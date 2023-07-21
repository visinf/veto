# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_transforms(cfg, is_train=True):
    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255)
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST

    transform = T.Compose(
        [
            #T.SquarePad(),
            T.Resize(min_size, max_size),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    if cfg.DATASETS.USE_DEPTH:
        transform_depth = T.Compose(
            [
                #T.SquarePad(single_channel=True),
                T.Resize(min_size, max_size),
                T.ToTensor(),
                T.DepthNormalize(),
            ]
        )
    else:
        transform_depth = None
    return transform, transform_depth
