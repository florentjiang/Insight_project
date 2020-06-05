from flask import Flask, redirect, url_for, render_template, request
from textstat.textstat import textstat
from textblob import TextBlob
import pickle
import math

# Create the application object
app = Flask(__name__)

# Get cleaned text of the post from url
def getTextFromHTMLLink(url):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    rows = soup.find_all('p')
    str_cells = str(rows)
    cleantext = BeautifulSoup(str_cells, "lxml").get_text()
    return cleantext

# Function to count words
def countWords(s):
  words=s.split()
  return len(words)

@app.route('/',methods=['GET', 'POST'])
def login():
    if request.method == "POST":
      url = request.form["nm"]
      if url == "":
        return render_template('index.html')
      model_path="./"
      model_like = pickle.load(open(model_path+'like_model.sav', 'rb'))
      model_comment = pickle.load(open(model_path+'comment_model.sav', 'rb'))
      model_share = pickle.load(open(model_path+'share_model.sav', 'rb'))
      post=getTextFromHTMLLink(url)
      print(post)
      X=[[TextBlob(post).sentiment.polarity,
      TextBlob(post).sentiment.polarity*TextBlob(post).sentiment.polarity,
      textstat.gunning_fog(post),
      1,1,0,countWords(post)]]
      output_like = int(math.exp(model_like.predict(X)[0]))
      output_comment = int(math.exp(model_comment.predict(X)[0]))
      output_share = int(math.exp(model_share.predict(X)[0]))
      return redirect(url_for("user",output_like=output_like,output_comment=output_comment,output_share=output_share))
    else:
      return render_template('index.html')
@app.route("/<output_like>/<output_comment>/<output_share>")
def user(output_like=None,output_comment=None,output_share=None):
    return f"<p> The predicted weekly likes: {output_like}</p><p> The predicted weekly comment: {output_comment}</p><p> The predicted weekly share: {output_share}</p>"

# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=False) #will run locally http://127.0.0.1:5000/