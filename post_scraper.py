import pandas as pd
from facebook_scraper import get_posts
import datetime
import pickle

# read movie channel username stored in csv file
input_path = './posts/'
model_output_path = './models/'
movie_list = pd.read_csv(input_path+'page_username.csv')
start_date = datetime.datetime(2011, 1, 1)
end_date = datetime.datetime(2014, 5, 8)
post_list_all=[]

# for every movie in the list, scraping posts during studied periods
for item in movie_list['page_username']:
    post_user = {}
    post_list_user = []
    post_user['username'] = item
    print('Checking username: '+item)
    for post in get_posts(item,pages=1000):
        if post is not None:
            if (post['time'] is None) or (post['time'] > start_date and post['time'] < end_date):
                post_list_user.append(post)
    post_user['posts'] = post_list_user
    post_list_all.append(post_user)

pickle.dump(post_list_all,open(model_output_path+'fb_post_list_1000pages.p', "wb" ))