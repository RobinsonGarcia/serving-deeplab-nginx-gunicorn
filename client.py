from __future__ import print_function
import requests
import json
import cv2
from PIL import Image
import io 
import os
import matplotlib.pyplot as plt
import numpy as np

class client:
    def __init__(self,addr = 'http://192.168.99.101:3000',**params):
        
        self.predict_url = addr + '/predict'
        
        if isinstance(params,dict):
            self._setparams(params)

    def _setparams(self,params):
        for k,v in params.items():
            self.__dict__[k]=v
    def check_filetype(self,file):
        filename = os.path.split(file)[-1]
        filetype = filename.split('.')[-1]
        if  filetype != "png":
            tmp = Image.open(file)
            tmp.save(filename+".png")
            return filename+".png"
        else:
            return file
    
    def run(self,file,**params):
        file = self.check_filetype(file)
            
        if isinstance(params,dict):
            self._setparams(params)

        img = cv2.imread(file)
        # prepare headers for http request
        content_type = 'image/png'
        headers = {'content-type': content_type,
                   'filename':os.path.split(file)[-1].split(".")[0],'json':json.dumps(params)}

        # encode image as jpeg
        _, img_encoded = cv2.imencode('.png', img)
        # send http request with image and receive response
        response = requests.post(self.predict_url, data=img_encoded.tostring(), headers=headers)
        
        # decode response
        return Image.open(io.BytesIO(response.content))

