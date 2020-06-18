import pandas as pd
from facebook_scraper import get_posts
from post2success import config
import pickle

# read movie channel username stored in csv file
# for every movie in the list, scraping posts during studied periods
def post_scraper():
    post_list_all=[]
    movie_list = pd.read_csv(config.movie_list_path)
    for item in movie_list['page_username']:
        post_user = {}
        post_list_user = []
        post_user['username'] = item
        print('Checking username: '+item)
        for post in get_posts(item,pages=1000):
            if post is not None:
                if (post['time'] is None) or (post['time'] > config.start_date and post['time'] < config.end_date):
                    post_list_user.append(post)
        post_user['posts'] = post_list_user
        post_list_all.append(post_user)
    pickle.dump(post_list_all,open(config.post_path, "wb" ))
    return

# run the post scraper
if __name__ == "__main__":
        post_scraper()
