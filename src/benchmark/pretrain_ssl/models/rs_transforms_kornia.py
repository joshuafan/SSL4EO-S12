"""
Kornia reimplementation of augmentations.
Input images are usually Tensors of shape [C,H,W], dtype=torch.float32.
Values are assumed to be between 0 and 1.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import torch
from torch import Tensor
import random
import kornia as K
from kornia.augmentation import AugmentationBase2D
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D




class RandomBrightness(IntensityAugmentationBase2D):
    """Random Brightness from https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/benchmark/pretrain_ssl/models/rs_transforms_float32.py
    """
    def __init__(
        self, 
        brightness: float = 0.4,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.brightness = brightness

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, Tensor]:
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        return dict(brightness=s)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        img = input * params['brightness']
        return img


class RandomContrast(IntensityAugmentationBase2D):
    """Random Contrast from https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/benchmark/pretrain_ssl/models/rs_transforms_float32.py
    
    Assumes input is of shape (B, C, H, W) or (C, H, W)
    """
    def __init__(
        self,
        contrast: float = 0.4,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.contrast = contrast

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, Tensor]:
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        return dict(contrast=s)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        mean = torch.mean(input, axis=(-1, -2), keepdim=True)
        img = ((input - mean) * params['contrast'] + mean)
        return img


class ToGray(IntensityAugmentationBase2D):
    """ToGray from https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/benchmark/pretrain_ssl/models/rs_transforms_float32.py
    
    Assumes input is of shape (B, C, H, W) or (C, H, W)
    """
    def __init__(
        self,
        out_channels,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.out_channels = out_channels

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:        
        gray_img = torch.mean(input, axis=-3)
        if len(input.shape) == 4:
            gray_img = torch.tile(gray_img, (1, self.out_channels, 1, 1))
        else:
            gray_img = torch.tile(gray_img, (self.out_channels, 1, 1))
        return gray_img


class RandomChannelDrop(IntensityAugmentationBase2D):
    r"""
    RandomChannelDrop from

    Assumes input is of shape (B, C, H, W) or (C, H, W)
    """
    def __init__(
        self,
        min_n_drop: int = 1,
        max_n_drop: int = 8,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, float]:
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels_to_drop = np.random.choice(range(input_shape[-3]), size=n_channels, replace=False)
        return dict(channels_to_drop=channels_to_drop)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """
        `input` is assumed to be of shape [B x C x H x W], where B is the batch size and C is the number of channels
        """
        for c in params["channels_to_drop"]:
            if len(input.shape) == 4:
                input[:, c, :, :] = 0
            else:
                input[c, :, :] = 0  
        return input


