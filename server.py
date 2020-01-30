#!/usr/bin/env python3
 
import logging

from logging.handlers import RotatingFileHandler
from flask import Flask, request, Response, send_file, send_from_directory, abort, render_template
import json
import jsonpickle
import numpy as np
import cv2
import base64
import os
from PIL import Image
#from model.corrmodels import Model
from inference.inference import InferenceModel as Model

MODE = os.environ.get('MODE')
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER,os.environ.get('MODEL_PATH'))
IMAGE_FOLDER = os.path.join(APP_FOLDER,'static','images')
PREDICTIONS_FOLDER = os.path.join(APP_FOLDER,'static','predictions')

# Initialize the Flask application
app = Flask(__name__)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['MODEL_PATH'] = MODEL_PATH
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER
app.config['MODE'] = MODE

# Load the inference model
try:
    model = Model(model_dir = app.config['MODEL_PATH'])
    app.logger.info("Model loaded")
except Exception as e:
    
    app.logger.error("Failed to load model")
    raise str(e)

@app.route("/")
@app.route("/index")
def main(name=None):
    return render_template('index.html', name=name)

@app.route("/get-image/<image_name>")
def get_image(image_name):
    try:
        return send_from_directory(app.config["IMAGE_FOLDER"], filename=image_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)

@app.route("/get-image/<image_name>")
def get_mask(image_name):
    try:
        return send_from_directory(app.config["PREDICTIONS_FOLDER"], filename=image_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)


@app.route('/predict', methods=['POST'])
def predict():
    r = request
    
    # log header info
    app.logger.info("headers{}".format(r.headers))
    app.logger.info(r.headers['json'])

    params = dict(json.loads(r.headers['json']))
    app.logger.info(params)

    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # convert from BGR to RGB, and read size info
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    height,width = img.size
    app.logger.info("running inference...")

    # run the segmentation model
    _, mask = model.run(img,**params)

    if params["return_mask"]==True:
        mask = mask.resize((width,height),Image.BICUBIC)

    app.logger.info("saving outputs...")
    
    # save the loaded image and the model output
    filename = os.path.join(app.config['IMAGE_FOLDER'],\
        r.headers['filename']+'.png')
    img.save(filename)

    maskname = os.path.join(app.config['PREDICTIONS_FOLDER'],\
        r.headers['filename']+'.png')
    mask.save(maskname)

    # return the model output
    return send_from_directory(app.config['PREDICTIONS_FOLDER']\
        , filename=os.path.split(maskname)[-1], as_attachment=True)
  
if __name__=="__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000, debug=True)

#https://pythonhosted.org/Flask-Bootstrap/basic-usage.html
