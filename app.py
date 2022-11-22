from flask import Flask, jsonify, request, flash, redirect, url_for
from flask_cors import CORS
import matplotlib.image as img
import joblib
import os
import sklearn
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array, array_to_img
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np 
import pandas as pd


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'picture'


def return_image_embedding(img_path):
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    curr_df = pd.DataFrame(preds[0]).T
    return curr_df

def save_image(image):
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
@app.route("/")
def home():
  return 'API up and running'

@app.route("/decisiontree", methods=['POST'])
def predict_decisiontree():
    if request.method == "POST":
        if 'file' not in request.files:
            return 'Error: No file part'
        image = request.files['file']
        if(image.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
            save_image(image)
            img_path = os.path.realpath(app.config['UPLOAD_FOLDER']+'/'+image.filename)
            nueva = return_image_embedding(img_path)
            model = joblib.load('decisiontree.pkl')
            prediction = model.predict(nueva)
            return jsonify({'prediction': str(prediction)})          
        else:
            return 'Error: File type not supported'

if __name__ == '__main__':
  app.run()