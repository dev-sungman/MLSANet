import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import random
import math

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageFilter

class BaseTransform(ABC):
    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob))(img)

    @abstractmethod
    def transform(self, img):
        pass

class GaussianBlur(BaseTransform):
    def transform(self, img):
        kernel_size = self.mag
        gBlur = ImageFilter.GaussianBlur(kernel_size)
        return img.filter(gBlur)

# ref: https://github.com/proceduralia/pytorch-neural-enhance
class UnsharpMask(BaseTransform):
    def transform(self, img):
        result = np.empty_like(img, dtype=np.float)
        for c in range(img.shape[-1]):
            result[..., c] = 
