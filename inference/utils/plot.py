import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def _build_masked_image(img,mask,return_pil=True,fill_contour=True,cmap=[255,10,10],thickness=2,return_pair=False):

    img = np.array(img)
    img0 = img.copy()

    
    mask = np.argmax(mask,axis=2)
    try:
        _,contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    except:
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(img, contour, -1, tuple(cmap), thickness)

    if fill_contour==True:
        cmap = np.array([[0,0,0],cmap])
        kps = np.argwhere(mask)
        img = np.array(img)
        color = cmap[mask]
        img = (0.7*img + 0.3*color).astype(np.uint8)
    
    if return_pair:
        return _build_img_mask_pair(img0,img,return_pil)
    else:
        if return_pil:
            return Image.fromarray(img)
        else:
            return img



def _build_img_mask_pair(img,mask,return_pil=True):
    merged = np.hstack([img,mask])
    merged = Image.fromarray(merged)
    if return_pil:
        return merged
    else:
        return np.array(merged)
