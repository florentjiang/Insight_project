from flask import Flask, flash, render_template, request
from werkzeug.utils import secure_filename
from textstat.textstat import textstat
from textblob import TextBlob
from keras.preprocessing.sequence import pad_sequences
import pickle
import math
import os
import cv2
import pandas as pd
import numpy as np
from keras import layers
from keras import Input
from keras.models import Model
from keras import models
from post2success import config
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create the application object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET', 'POST'])
def index():
    tb._SYMBOLIC_SCOPE.value = True
    tokenizer = pickle.load(open(os.path.join(config.model_dir,'deep_learning_tokenizer'), 'rb'))
    model_like = pickle.load( open(os.path.join(config.model_dir,'deep_learning_like_model.pkl'), 'rb'))
    model_comment = pickle.load(open(os.path.join(config.model_dir,'deep_learning_comment_model.pkl'), 'rb'))
    model_share = pickle.load(open(os.path.join(config.model_dir,'deep_learning_share_model.pkl'), 'rb'))
    image_predictor = pickle.load(open(os.path.join(config.model_dir,'deep_learning_image_predictor.pkl'), 'rb'))
    text_predictor = pickle.load(open(os.path.join(config.model_dir,'deep_learning_text_predictor.pkl'), 'rb'))
    if request.method == "POST":
      # extract uploaded image file and text
      if 'file' not in request.files:
        return render_template('index.html',image_path=None)
      file = request.files['file']
      post = request.form["message"]
      if post == "" and file.filename == "":
        return render_template('index.html',image_path=None)
      if file and allowed_file(file.filename):
        # resize the image to 100*100 pixels
        pixel_x = 100
        pixel_y = 100
        dim = (pixel_x, pixel_y)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        im = cv2.imread(full_path)
        X_image_train = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
      else:
        full_path=''
        X_image_train = [[[0]*3]*100]*100
      sentiment = round(TextBlob(post).sentiment.polarity,3)
      readability = round(textstat.gunning_fog(post),3)
      # tokenize text
      maxlen = 30
      sequences = tokenizer.texts_to_sequences([post])
      X_text_train = pad_sequences(sequences, maxlen=maxlen)
      X=[[X_image_train],X_text_train]
      # predict number of likes, comments and shares
      output_like = int(math.exp(model_like.predict(X)[0]))
      output_comment = int(math.exp(model_comment.predict(X)[0]))
      output_share = int(math.exp(model_share.predict(X)[0]))
      # calculate the image score and text score using a pre-trained model
      image_score = np.dot(model_like.layers[8].get_weights()[0][:1024].reshape(1,-1),image_predictor.predict([[X_image_train]]).reshape(-1,1))[0][0]
      text_score = np.dot(model_like.layers[8].get_weights()[0][1024:].reshape(1,-1),text_predictor.predict([X_text_train]).reshape(-1,1))[0][0]
      return render_template('results.html',image_path=full_path, post=post, output_like=output_like,output_comment=output_comment,output_share=output_share,readability = readability,sentiment=sentiment, image_score = image_score, text_score = text_score)
    else:
      return render_template('index.html',image_path=None)
# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=False) 