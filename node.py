import torch
import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class CLAHEEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "clip_limit": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "grid_size": ("INT", {"default": 8, "min": 2, "max": 16, "step": 1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/watermark"

    def enhance(self, image, clip_limit, grid_size):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            result.append(torch.from_numpy(enhanced.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class HighPassFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "cutoff_freq": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 1.0})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter"
    CATEGORY = "image/watermark"

    def filter(self, image, cutoff_freq):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            f = fft2(gray)
            fshift = fftshift(f)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 0
            fshift = fshift * mask
            f_ishift = ifftshift(fshift)
            img_back = np.abs(ifft2(f_ishift))
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
            enhanced = np.repeat(img_back[:,:,np.newaxis], 3, axis=2)
            result.append(torch.from_numpy(enhanced.astype(np.float32)))
        return (torch.stack(result),)

class EdgeDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                             "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "image/watermark"

    def detect(self, image, low_threshold, high_threshold):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            edges_rgb = np.repeat(edges[:,:,np.newaxis], 3, axis=2)
            result.append(torch.from_numpy(edges_rgb.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class CombineEnhancements:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"original": ("IMAGE",), 
                             "enhanced1": ("IMAGE",), 
                             "enhanced2": ("IMAGE",), 
                             "enhanced3": ("IMAGE",),
                             "weight1": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "weight2": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "weight3": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "image/watermark"

    def combine(self, original, enhanced1, enhanced2, enhanced3, weight1, weight2, weight3):
        result = []
        for orig, enh1, enh2, enh3 in zip(original, enhanced1, enhanced2, enhanced3):
            combined = orig * (1 - weight1 - weight2 - weight3) + enh1 * weight1 + enh2 * weight2 + enh3 * weight3
            result.append(torch.clamp(combined, 0, 1))
        return (torch.stack(result),)

class AdaptiveThresholding:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
                             "C": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "threshold"
    CATEGORY = "image/watermark"

    def threshold(self, image, block_size, C):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
            thresh_rgb = np.repeat(thresh[:,:,np.newaxis], 3, axis=2)
            result.append(torch.from_numpy(thresh_rgb.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class MorphologicalOperations:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "operation": (["dilate", "erode", "open", "close"],),
                             "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "morph"
    CATEGORY = "image/watermark"

    def morph(self, image, operation, kernel_size):
        result = []
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            if operation == "dilate":
                morphed = cv2.dilate(img_np, kernel, iterations=1)
            elif operation == "erode":
                morphed = cv2.erode(img_np, kernel, iterations=1)
            elif operation == "open":
                morphed = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
            elif operation == "close":
                morphed = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            result.append(torch.from_numpy(morphed.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class ImprovedGrayColorEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "lower_gray": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                             "upper_gray": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                             "boost_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1}),
                             "sharpen_amount": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_gray"
    CATEGORY = "image/watermark"

    def enhance_gray(self, image, lower_gray, upper_gray, boost_factor, sharpen_amount):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Create a mask for gray areas using A and B channels
            gray_mask = cv2.inRange(lab, (0, 128-10, 128-10), (255, 128+10, 128+10))
            
            # Boost the L channel of gray areas
            l[gray_mask > 0] = np.clip(l[gray_mask > 0] * boost_factor, 0, 255)
            
            # Merge the channels back
            lab_boosted = cv2.merge([l, a, b])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab_boosted, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
            
            result.append(torch.from_numpy(enhanced.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class TextureEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "frequency_range": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                             "boost_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_texture"
    CATEGORY = "image/watermark"

    def enhance_texture(self, image, frequency_range, boost_factor):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Perform FFT
            f = fft2(gray)
            fshift = fftshift(f)
            
            # Create a band-pass filter
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            mask[crow-frequency_range:crow+frequency_range, ccol-frequency_range:ccol+frequency_range] = 1
            mask[crow-frequency_range//2:crow+frequency_range//2, ccol-frequency_range//2:ccol+frequency_range//2] = 0
            
            # Apply filter and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.abs(ifft2(f_ishift))
            
            # Normalize and boost
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
            img_back = np.clip(img_back * boost_factor, 0, 1)
            
            # Combine with original image
            enhanced = img_np.astype(np.float32) / 255.0
            enhanced += np.repeat(img_back[:,:,np.newaxis], 3, axis=2)
            enhanced = np.clip(enhanced, 0, 1)
            
            result.append(torch.from_numpy(enhanced.astype(np.float32)))
        return (torch.stack(result),)

class DenoisingFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 10, "min": 0, "max": 20, "step": 0.1}),
                             "color_strength": ("FLOAT", {"default": 10, "min": 0, "max": 20, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/watermark"

    def denoise(self, image, strength, color_strength):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, strength, color_strength, 7, 21)
            result.append(torch.from_numpy(denoised.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class FlexibleCombineEnhancements:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"original": ("IMAGE",)},
                "optional": {f"enhanced{i}": ("IMAGE",) for i in range(1, 6)},
                "hidden": {f"weight{i}": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}) for i in range(1, 6)}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "image/watermark"

    def combine(self, original, **kwargs):
        enhanced_images = [kwargs.get(f"enhanced{i}") for i in range(1, 6) if f"enhanced{i}" in kwargs]
        weights = [kwargs.get(f"weight{i}", 0.2) for i in range(1, 6) if f"enhanced{i}" in kwargs]
        
        result = []
        for imgs in zip(original, *enhanced_images):
            combined = imgs[0] * (1 - sum(weights))
            for img, weight in zip(imgs[1:], weights):
                combined += img * weight
            result.append(torch.clamp(combined, 0, 1))
        return (torch.stack(result),)

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
