from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "CLAHEEnhancement": CLAHEEnhancement,
    "HighPassFilter": HighPassFilter,
    "EdgeDetection": EdgeDetection,
    "CombineEnhancements": CombineEnhancements,
    "AdaptiveThresholding": AdaptiveThresholding,
    "MorphologicalOperations": MorphologicalOperations,
    "GrayColorEnhancement": GrayColorEnhancement,
    "ImprovedGrayColorEnhancement": ImprovedGrayColorEnhancement,
    "TextureEnhancement": TextureEnhancement,
    "DenoisingFilter": DenoisingFilter,
    "FlexibleCombineEnhancements": FlexibleCombineEnhancements
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLAHEEnhancement": "CLAHE Enhancement",
    "HighPassFilter": "High Pass Filter",
    "EdgeDetection": "Edge Detection",
    "CombineEnhancements": "Combine Enhancements",
    "AdaptiveThresholding": "Adaptive Thresholding",
    "MorphologicalOperations": "Morphological Operations",
    "GrayColorEnhancement": "Gray Color Enhancement",
    "ImprovedGrayColorEnhancement": "Improved Gray Color Enhancement",
    "TextureEnhancement": "Texture Enhancement",
    "DenoisingFilter": "Denoising Filter",
    "FlexibleCombineEnhancements": "Flexible Combine Enhancements"
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]