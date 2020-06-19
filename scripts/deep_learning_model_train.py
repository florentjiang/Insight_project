from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras import Input
from keras.models import Model
from keras import models
from keras.applications import VGG16
from keras.optimizers import Adam
import os
import numpy as np
import cv2
import pandas as pd
import pickle
from post2success import config

def data_processor (num_of_samples=10, maxlen=30, max_words=10000, embedding_dim=50, pixel_x=100, pixel_y=100):
    # loading data
    # X_image_train : image input
    # X_text_train  : text input
    # texts         : raw text
    # y_train_like  : nb of likes output
    # y_comment_like  : nb of comments output
    # y_share_like  : nb of shares output
    dim = (pixel_x, pixel_y)
    X_image_train = []
    y_train_like = []
    y_train_comment = []
    y_train_share = []
    texts = []
    data_raw=pd.read_csv(config.movie_list_path)
    post_list_all_reload = pickle.load( open(config.post_path, "rb") )
    for item in data_raw['page_username']:
        for movie in post_list_all_reload:
            if (movie['username']==item):
                print('Loading data for movie: ',item)
                for post in movie['posts']:
                    if post['image'] and post['post_id']:
                        fname=movie['username']+'_'+post['post_id']+'.jpg'
                        fpath = os.path.join(config.photo_dir,fname)
                        try:
                            im = cv2.imread(fpath)
                            im_resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                            X_image_train.append(im_resized) # loading image data
                            texts.append(post['text']) # loading text data
                            y_train_like.append(post['likes']+0.1) # loading nb of likes output
                            y_train_comment.append(post['comments']+0.1) # loading nb of comments output
                            y_train_share.append(post['shares']+0.1) # loading nb of shares output
                        except Exception:
                            pass
    # prepare the text input
    # tokenize raw text into X_text_train
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    X_text_train = pad_sequences(sequences, maxlen=maxlen)

    # NLP part: using pretrained word embedding model
    # load the pretrained GloVe coefficient
    embeddings_index = {}
    f = open(os.path.join(config.model_dir, 'glove.6B.50d.txt'),encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return X_image_train, X_text_train, y_train_like, y_train_comment, y_train_share, embedding_matrix


def deep_learning_model_train(epochs=40, batch_size=512):
    # loading input data
    pixel_x = 100
    pixel_y = 100
    max_words = 10000
    embedding_dim = 50
    X_image_train, X_text_train, y_train_like, y_train_comment, y_train_share, embedding_matrix = data_processor (max_words=max_words,embedding_dim=embedding_dim, pixel_x=pixel_x, pixel_y=pixel_y)
    # Build the deep learning model
    # NLP part: build the deep learning model for text input
    text_input = Input(shape=(None,), dtype='int32', name='text')
    embedded_text = layers.Embedding(max_words, embedding_dim)(text_input)
    encoded_text = layers.LSTM(16)(embedded_text)

    # Computer vision part: build the deep learning model for image input
    image_input = Input(shape=(pixel_x, pixel_y, 3), name='image')
    vgg16 = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(pixel_x, pixel_y, 3))(image_input)
    x = layers.Flatten()(vgg16) 
    x = layers.Dense(1024, activation='relu')(x)

    # Concatenate NLP output and computer vision output
    # build the output layer for regression

    concatenated = layers.concatenate([x, encoded_text], axis=-1)
    output = layers.Dense(1, activation="linear")(concatenated)
    opt = Adam(lr=2e-3)
    for y_train ,filename in zip([y_train_like, y_train_comment, y_train_share],['deep_learning_like_model.pkl','deep_learning_comment_model.pkl','deep_learning_share_model.pkl']):
        model = Model([image_input, text_input], output)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
        

        # freeze parameters in pretrained models
        for layer in model.layers[1].layers: # freeze VGG16 coefficient
            layer.trainable = False
        model.layers[4].set_weights([embedding_matrix])
        model.layers[4].trainable = False # freeze GloVe word embedding

        # train and save model
        model.fit([X_image_train, X_text_train], y_train, epochs=epochs, batch_size=batch_size)
        pickle.dump(model, open(os.path.join(config.model_dir,filename), 'wb'))

if __name__ == "__main__":
    deep_learning_model_train()