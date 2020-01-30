import os
import numpy as np
import tensorflow as tf
from PIL import Image
import time

_FROZEN_GRAPH_PATH = os.path.join('model', 'frozen_graph','frozen_inference_graph.pb')

class Model(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    
    INPUT_SIZE = 513

    def __init__(self, model_dir=None,verbose=False,predictions=True):

        """Creates and loads pretrained deeplab model."""
        if not isinstance(model_dir,str):
            model_dir = _FROZEN_GRAPH_PATH
        self.graph = tf.Graph()
        self.verbose = verbose
        self.INPUT_SIZE = 513
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        if not predictions:
            self.OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0'
        
        graph_def = None
        model_filename = model_dir

       
        with tf.gfile.GFile(model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        
        config = tf.compat.v1.ConfigProto(device_count={'GPU':0})
 

        self.sess = tf.Session(graph=self.graph,config=config)

    
        
    def run(self, image):
            """Runs inference on a single image.

            Args:
                image: A PIL.Image object, raw input image.

            Returns:
                resized_image: RGB image resized from original input image.
                seg_map: Segmentation map of `resized_image`.
            """
          
            target_size = (self.INPUT_SIZE,self.INPUT_SIZE)


            resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            if self.verbose: print('Image resized')
            start_time = time.time()
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
            if self.verbose:print('Image processing finished')
            if self.verbose:print('Elapsed time : ' + str(time.time() - start_time))
            seg_map = batch_seg_map[0]

            return resized_image, seg_map
    
    def run_batch(self, images):
            """Runs inference on batch.

            Args:
                image: Numpy array with dims (N x H x W x C).

            Returns:
                resized_image: RGB image resized from original input image.
                seg_map: Segmentation map of `resized_image`.
            """
         
            start_time = time.time()
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [images]})
            if self.verbose:print('Image processing finished')
            if self.verbose:print('Elapsed time : ' + str(time.time() - start_time))
            seg_map = batch_seg_map[0]
            return seg_map
