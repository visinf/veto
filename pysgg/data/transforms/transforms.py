# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
from PIL import Image, ImageOps
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target



class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

RandomVerticalFlip(prob=5, )

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class DepthNormalize(object):
    """
    Per depth-image normalization
    zero mean and unit standard deviation
    """

    def __call__(self, depth_img, target):
        """
        :param depth_img: depth image tensor
        :return: depth_map (normalized)
        """
        # -- Convert depth tensor to float type
        depth_img = depth_img.float()
        # -- Determine the dimension of the image
        _, height, width = depth_img.size()
        # -- QUEST @Max what does zero_avoid do?
        zero_avoid = 1.0 / np.sqrt(height * width)
        # -- Do the normalization
        depth_map = depth_img - depth_img.mean()
        temp = np.maximum(depth_map.std(), zero_avoid)
        temp = temp.to(depth_map.dtype)
        depth_map /= temp
        return depth_map, target

class SquarePad(object):
    def __init__(self, single_channel=False):
        # -- Set the image mean (fill the padding with this number)
        if not single_channel:
            self.mean = (int(0.485 * 256), int(0.456 * 256), int(0.406 * 256))
        else:
            # -- You may want to change this number with the real mean
            self.mean = 32767

    def __call__(self, img, target):
        w, h = img.size
        # -- replace the filling value
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=self.mean)
        return img_padded, target

"""
class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.size = min_size
        #self.max_size = max_size
    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target
"""