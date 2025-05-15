import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

class PercentileNormalize(ImageOnlyTransform):
    """Normalize image by percentiles. Works both as standalone and with albumentations."""
    
    def __init__(
        self,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.available_keys = {'image'}
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Standalone usage (for GUI and direct calls)."""
        return self.apply(image)
    
    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Albumentations interface."""
        p_low = np.percentile(img, self.lower_percentile)
        p_high = np.percentile(img, self.upper_percentile)
        image_clipped = np.clip(img, p_low, p_high)
        
        mean = np.mean(image_clipped)
        std = np.std(image_clipped)
        
        normalized = (image_clipped - mean) / (std + 1e-8)
        return normalized
    
    def get_transform_init_args_names(self) -> tuple:
        """Required by albumentations."""
        return ('lower_percentile', 'upper_percentile', 'p')
        