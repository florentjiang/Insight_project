import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from post2success import config
import pickle
import os

def linear_model_train():
# data loading and cleaning
    data_raw=pd.read_csv(config.post_target_path)
    data=data_raw.loc[data_raw['page_post']==1].dropna()
    data['sentiment_sq']=data['sentiment']*data['sentiment']
    data['ln_word']=np.log(data['word'])
    X=data[['sentiment','sentiment_sq','readability','link','photo','video','ln_word']].values

    y_nb_like=np.log(data['likescount']+1)
    y_nb_comment=np.log(data['commentscount']+1)
    y_nb_share=np.log(data['sharescount']+1)

# model training
    model_like = LinearRegression().fit(X, y_nb_like)
    model_comment = LinearRegression().fit(X, y_nb_comment)
    model_share = LinearRegression().fit(X, y_nb_share)

# model saving
    filename = 'linear_like_model.pkl'
    pickle.dump(model_like, open(os.path.join(config.model_dir,filename), 'wb'))
    filename = 'linear_comment_model.pkl'
    pickle.dump(model_comment, open(os.path.join(config.model_dir,filename), 'wb'))
    filename = 'linear_share_model.pkl'
    pickle.dump(model_share, open(os.path.join(config.model_dir,filename), 'wb'))

# train the linear regression model
if __name__ == "__main__":
    linear_model_train()
