import cv2
import numpy as np

def enhance_low_light(img: np.ndarray, gamma: float = 1.5, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    """
    Brightens dark images using CLAHE + Gamma correction.
    """
  
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)

   
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)

    
    img_lab2 = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(img_lab2, cv2.COLOR_LAB2RGB)

    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    img3 = cv2.LUT(img2, table)

    return img3
