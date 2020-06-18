import urllib.request
import pandas as pd
import numpy as np
import pickle
import socket
from post2success import config

# read movie channel username stored in csv file
# read posts stored in pickle file

def photo_scraper():
# for every post containing a image link, scraping the photo and storing
# locally as username_postid.jpg
    movie_list=pd.read_csv(config.movie_list_path)
    post_list_all_reload = pickle.load( open(config.post_path, "rb") )
    for item in movie_list['page_username']:
        print('Checking username: '+item)
        for movie in post_list_all_reload:
            if (movie['username']==item):
                for post in movie['posts']:
                    if post['image'] and post['post_id']:
                        photo_url = post['image']
                        local_address = str(config.photo_dir)+item+'_'+post['post_id']+'.jpg'
                        try:
                            socket.setdefaulttimeout(10)
                            urllib.request.urlretrieve(photo_url, local_address)
                        except Exception as e:
                            print("error for post id: "+post['post_id'])
    return
# run the photo scraper
if __name__ == "__main__":
    photo_scraper()
