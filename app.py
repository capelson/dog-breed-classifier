from __future__ import division, print_function

import sys
import os
import glob
import re
import json
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), 'model.json')
WEIGHTS_PATH = os.path.join(os.getcwd(), 'model_weights.hdf5')

with open( 'labels.json', 'r') as JSON:
       LABELS = json.load(JSON)


def load_model():
    json_file = open(MODEL_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(WEIGHTS_PATH)
    print("Loaded model from disk\n Check http://localhost:5000/")
    return loaded_model

model = load_model()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(240,240))

    # preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.

    preds = model.predict(x)
    return preds

def get_breed_name(breed_code):
    return breed_code.split('-')[1]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        pred_class = get_breed_name(LABELS[str(np.argmax(preds))])
        return str(pred_class)

    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()