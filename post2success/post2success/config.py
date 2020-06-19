from pathlib import Path
import datetime
import os

post_dir = Path('d:/Gdrive/Insight/Insight_project/data/posts/')
movie_list_path = os.path.join(post_dir,'page_username.csv')
post_target_path = os.path.join(post_dir,'post_target.csv')
post_path = os.path.join(post_dir,'fb_post_list.pkl')

photo_dir = Path('d:/Gdrive/Insight/Insight_project/data/photos/')
model_dir = Path('d:/Gdrive/Insight/Insight_project/models/')
start_date = datetime.datetime(2011, 1, 1)
end_date = datetime.datetime(2014, 5, 8)
