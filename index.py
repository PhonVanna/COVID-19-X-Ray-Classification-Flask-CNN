import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from tensorflow.keras.models import  Model, load_model
from tensorflow.keras.preprocessing import image

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = { 'jpg', 'png', 'jpeg' }
my_model = load_model("model/covid_model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def test_model(image_for_testing):
    test_image=image.load_img(image_for_testing,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=my_model.predict_classes(test_image)
    Catagories=['Person has no Covid-19','Person has Covid-19']
    result_name = result = Catagories[int(result[0][0])]
    return result_name


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result_name = test_model(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("home.html", name=filename, result = result_name)
    return render_template("home.html", name='', result='')
    

if __name__ == '__main__':
    app.run(debug=True)