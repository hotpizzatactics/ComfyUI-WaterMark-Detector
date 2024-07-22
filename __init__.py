from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "CLAHEEnhancement": CLAHEEnhancement,
    "HighPassFilter": HighPassFilter,
    "EdgeDetection": EdgeDetection,
    "CombineEnhancements": CombineEnhancements,
    "AdaptiveThresholding": AdaptiveThresholding,
    "MorphologicalOperations": MorphologicalOperations,
    "GrayColorEnhancement": GrayColorEnhancement
    "ImprovedGrayColorEnhancement": ImprovedGrayColorEnhancement,
    "TextureEnhancement": TextureEnhancement,
    "DenoisingFilter": DenoisingFilter,
    "FlexibleCombineEnhancements": FlexibleCombineEnhancements
}

__all__ = ['NODE_CLASS_MAPPINGS']


