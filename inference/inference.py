from inference.utils.plot import _build_masked_image
from model.corrmodels import Model 
from PIL import Image
import numpy as np

class InferenceModel:
    def __init__(self,model=None, model_dir=None,**kwargs):
        
        if isinstance(model_dir,str):
            self.model = Model(model_dir=model_dir,predictions=False)
        else:
            self.model = model
        self.return_pil=True
        self.fill_contour=False
        self.cmap=(0,0,255)
        self.thickness=1
        self.return_pair=False
        self.return_mask = False
        
        
        if isinstance(kwargs,dict):
            self._setparams(kwargs)

    def _setparams(self,kwargs):
        for k,v in kwargs.items():
            self.__dict__[k]=v

    def run(self,image,**kwargs):
        """
        image: PIL object
        """
        if isinstance(kwargs,dict):
            self._setparams(kwargs)
        W,H = image.size
        
        im,mask = self.model.run(image) 

        if self.return_mask:
            out = np.argmax(mask,axis=2)
            out = Image.fromarray(out.astype(np.uint8))
            
        
        else:
            print(3)
            out = _build_masked_image(im,mask,\
                        return_pil=self.return_pil,\
                        fill_contour=self.fill_contour,\
                        cmap=self.cmap,\
                        thickness=self.thickness,\
                        return_pair=self.return_pair)
        print(4)
        w,h = out.size

        if self.return_pair:
            print(5)
            W = int(2*W)
        if (H!=h) or (W!=w):
            print(6)
            out = out.resize((W,H))
            print(7)
        return None,out