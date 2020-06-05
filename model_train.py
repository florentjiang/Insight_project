import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import pickle



# data loading and cleaning
data_raw=pd.read_csv('linear.csv')
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
model_path="./web-app/"
filename = 'like_model.sav'
pickle.dump(model_like, open(model_path+filename, 'wb'))
filename = 'comment_model.sav'
pickle.dump(model_comment, open(model_path+filename, 'wb'))
filename = 'share_model.sav'
pickle.dump(model_share, open(model_path+filename, 'wb'))