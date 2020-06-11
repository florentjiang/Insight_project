import urllib.request
import pandas as pd
import numpy as np
import pickle
import socket

# read movie channel username stored in csv file
# read posts stored in pickle file
input_path = './posts/'
model_input_path = './models/'
photo_output_path = './photos/'
movie_list=pd.read_csv(input_path+'page_username.csv')
post_list_all_reload = pickle.load( open(model_input_path+'fb_post_list_1000pages.p', "rb") )

# for every post containing a image link, scraping the photo and storing
# local as username_postid.jpg
for item in movie_list['page_username']:
    print('Checking username: '+item)
    for movie in post_list_all_reload:
        if (movie['username']==item):
            for post in movie['posts']:
                if post['image'] and post['post_id']:
                    photo_url = post['image']
                    local_address = photo_output_path+item+'_'+post['post_id']+'.jpg'
                    try:
                        socket.setdefaulttimeout(1)
                        urllib.request.urlretrieve(photo_url, local_address)
                    except Exception as e:
                        print("error")