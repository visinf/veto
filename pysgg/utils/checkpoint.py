# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from pysgg.utils.c2_model_loading import load_c2_format
from pysgg.utils.imports import import_file
from pysgg.utils.model_serialization import load_state_dict
from pysgg.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
            custom_scheduler=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.custom_scheduler = custom_scheduler

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and not self.custom_scheduler:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, with_optim=True, update_schedule=False, load_mapping={}):
        if not f:
            if self.has_checkpoint():
                # override argument with existing checkpoint
                self.logger.info("Override argument with existing checkpoint")
                f = self.get_checkpoint_file()
            else:
                # no checkpoint could be found
                self.logger.info("No checkpoint found. Initializing model from scratch")
                return {}
        self.logger.info("Loading parameters from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_mapping)
        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                if update_schedule:
                    self.scheduler.last_epoch = checkpoint["iteration"]
                else:
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def load_weight_partially(self, f: str):
        state_dict = self._load_file(f)['model']
        own_state = self.model.state_dict()
        own_model_load_flag = {}
        for k, v in own_state.items():
            own_model_load_flag[k] = False
        for name, param in state_dict.items():
            if name.startswith('module'):
                name = name.strip('module.')
            try:
                if name not in own_state:
                    self.logger.info('[Missed]: {}'.format(name))
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
                own_model_load_flag[name] = True
                self.logger.info("[Loaded]: {}".format(name))
            except RuntimeError:
                self.logger.info('[Missed] Size Mismatch... : {}'.format(name))
        self.logger.info("non loaded module of current model: ")
        for k, v in own_model_load_flag.items():
            if not own_model_load_flag[k]:
                self.logger.info(f"{k}")

        self.logger.info("load the pretrain model %s" % f)
        return own_model_load_flag

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, load_mapping):
        load_state_dict(self.model, checkpoint.pop("model"), load_mapping, )


class DetectronCheckpointer(Checkpointer):
    def __init__(
            self,
            cfg,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
            custom_scheduler=False,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger, custom_scheduler
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "pysgg.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def clip_grad_norm(named_parameters, max_norm, logger, clip=False, verbose=False):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    if verbose:
        logger.info('---Total norm {:.5f} clip coef {:.5f}-----------------'.format(total_norm, clip_coef))
        for name, norm in param_to_norm.items():
            logger.info("{:<50s}: {:f}, ({})".format(name, norm, param_to_shape[name]))
        logger.info('-------------------------------')

    return total_norm

def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()

    # -- (ADDED) added a visual separator for better readability
    print("\n==================================\n"
          "Loading checkpoint parameters...\n")

    for name, param in state_dict.items():
        if name.find('features') != -1:
            name = name.replace( 'features', 'backbone.body.conv_body')
        elif name.find('rpn_head.conv.0') != -1:
            name = name.replace('rpn_head.conv.0', 'rpn.head.conv')
        elif name.find('score_fc') != -1:
            name = name.replace('score_fc', 'roi_heads.box.predictor.cls_score')
        elif name.find('bbox_fc') != -1:
            name = name.replace('bbox_fc', 'roi_heads.box.predictor.bbox_pred')
        elif name.find('roi_fmap.0') != -1:
            name = name.replace('roi_fmap.0', 'roi_heads.box.feature_extractor.fc6')
        elif name.find('roi_fmap.3') != -1:
            name = name.replace('roi_fmap.3', 'roi_heads.box.feature_extractor.fc7')
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
            print("Successfully loaded {} with size {}".format(name, param.size()))
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("\n*** We couldn't find {}".format(','.join(missing)))
        mismatch = True

    # -- (ADDED) added a visual separator for better readability
    print("==================================\n")
    return not mismatch