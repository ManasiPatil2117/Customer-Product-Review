from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
import requests
import json
from bs4 import BeautifulSoup

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')

    # Get link
    text1 = request.form['text1'].strip()

    try:
        # Get reviews from a link-
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}

        s = requests.Session()
        res = s.get(text1, headers=headers, verify=False)

        if res.status_code != 200:
            error = f"Invalid URL - Status Code: {res.status_code}"
            return render_template('error.html', error=error)

        soup = BeautifulSoup(res.text, "lxml")

        script = None
        for s in soup.find_all("script"):
            if 'pdpData' in s.text:
                script = s.get_text(strip=True)
                break

        PDPData = json.loads(script[script.index('{'):])
        reviewStr = ''
        error = ''

        # Check if having ratings or not-
        if len(PDPData["pdpData"]["ratings"]["reviewInfo"]["topReviews"]) != 0:

            # Check if reviews section is expanded or not-
            if "reviewsData" in PDPData:
                # Retrive each review and combine them in a reviewStr-

                for i in PDPData["reviewsData"]["reviews"]:
                    reviewStr += i["review"] + ' '

                if "reviewsMetaData" in PDPData["reviewsData"]:
                    for i in PDPData["reviewsData"]["reviewsMetaData"]["topImageReviewEntries"]:
                        reviewStr += i["review"] +' '
                
                # print(reviewStr)

            else:
                error = "Please extend the review list and then provide the link"
                return render_template('error.html', error=error)

        else:
            error = "No rating for given product"
            return render_template('error.html', error=error)

        text_final = ''.join(c for c in reviewStr if not c.isdigit())
        # remove stopwords

        processed_doc1 = ' '.join(
            [word for word in text_final.split() if word not in stop_words])

        # remove punctuations
        #text3 = ''.join(c for c in text2 if c not in punctuation)

        sa = SentimentIntensityAnalyzer()
        dd = sa.polarity_scores(text=processed_doc1)
        compound = round((1 + dd['compound'])/2, 2)
        return render_template('result.html', final=compound, text1=text_final, text2=dd['pos'], text5=dd['neg'], text4=compound, text3=dd['neu'])

    except requests.exceptions.RequestException as e:
        error = f"Error fetching the URL: {str(e)}"
        return render_template('error.html', error=error)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
