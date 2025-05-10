
import numpy as np

class PercentileNormalize():
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
    def __call__(self, image):
        p_low = np.percentile(image, self.lower_percentile)
        p_high = np.percentile(image, self.upper_percentile)
        image_clipped = np.clip(image, p_low, p_high)
        
        mean = np.mean(image_clipped)
        std = np.std(image_clipped)
        
        normalized = (image_clipped - mean) / (std + 1e-8)
        return normalized
        