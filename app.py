
from flask import Flask, request, render_template
import requests
import pandas as pd
from bs4 import BeautifulSoup
import string
import re
import unicodedata
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import string

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    

    # Get link
    reviews_url = request.form['productLink'].strip()
    print("URL --- " +reviews_url )
    try:
        # Get reviews from a link-
        headers = {
            'authority': 'www.amazon.com',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}

        s = requests.Session()
        # res = s.get(reviews_url, headers=headers, verify=False)

        len_page = 10


        def reviewsHtml(url, len_page):
            
            
            soups = []
            
            
            for page_no in range(1, len_page + 1):
                
            
                params = {
                    'ie': 'UTF8',
                    'reviewerType': 'all_reviews',
                    'filterByStar': 'critical',
                    'pageNumber': page_no,
                }
                
                
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    error = f"Invalid URL - Status Code: {response.status_code}"
                    return render_template('error.html', error=error)
                
                soup = BeautifulSoup(response.text, 'lxml')
                soups.append(soup)
                
            return soups


        def getReviews(html_data):

            
            data_dicts = []
            boxes = html_data.select('div[data-hook="review"]')
            
            
            for box in boxes:
                
            
                try:
                    name = box.select_one('[class="a-profile-name"]').text.strip()
                except Exception as e:
                    name = 'N/A'

                try:
                    stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
                except Exception as e:
                    stars = 'N/A'   

                try:
                    title = box.select_one('[data-hook="review-title"]').text.strip()
                except Exception as e:
                    title = 'N/A'

                try:
                    datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
                    date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
                except Exception as e:
                    date = 'N/A'

                try:
                    description = box.select_one('[data-hook="review-body"]').text.strip()
                except Exception as e:
                    description = 'N/A'

                
                data_dict = {
                    'Name' : name,
                    'Stars' : stars,
                    'Title' : title,
                    'Date' : date,
                    'Text' : description
                }

                
                data_dicts.append(data_dict)
            
            return data_dicts



        html_datas = reviewsHtml(reviews_url, len_page)

        reviews = []

        for html_data in html_datas:
            
        
            review = getReviews(html_data)
            reviews += review
        df_reviews = pd.DataFrame(reviews)

        print(df_reviews)
        df_reviews.to_csv('./input/Reviews.csv', index=False)
       
        # Read the Dataset
        df=pd.read_csv('./input/Reviews.csv')
        df.head()
        df.info()
        df['Text']

        # Clean the Dataset
        # First lets remove Punctuations from the Reviews
        def punctuation_removal(messy_str):
            clean_list = [char for char in messy_str if char not in string.punctuation]
            clean_str = ''.join(clean_list)
            return clean_str

        df['Text'] = df['Text'].apply(punctuation_removal)
        print("NOT HERE")
        # lets make a function to remove Numbers from the reviews
        def drop_numbers(list_text):
            list_text_new = []
            for i in list_text:
                if not re.search(r'\d', i):
                    list_text_new.append(i)
            return ''.join(list_text_new)

        df['Text'] = df['Text'].apply(drop_numbers)

        # lets show the Top 10 Reviews after Removal of Punctuations and Numbers
        df['Text'].head(10)

        ### Removing Accented Characters

        
        # lets create a function to remove accented characters
        def remove_accented_chars(text):
            new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            return new_text

        # lets apply the function
        df['Text'] = df.apply(lambda x: remove_accented_chars(x['Text']), axis = 1)

        # Create a function to remove special characters
        def remove_special_characters(text):
            pat = r'[^a-zA-z0-9]' 
            return re.sub(pat, ' ', text)
        
        # lets apply this function
        df['Text'] = df.apply(lambda x: remove_special_characters(x['Text']), axis = 1)

        # Feature Engineering

        # lets check if the dataset has any Missing Values
        df.isnull().sum()

        # Lets calculate the length of the Reviews
        df['length'] = df['Text'].apply(len)

        #Text Polarity
        # It is the expression that determines the sentimental aspect of an opinion. In textual data, the result of sentiment analysis can be determined for each entity in the sentence, document or sentence. The sentiment polarity can be determined as positive, negative and neutral.

        # Lets calculate the Polarity of the Reviews
        def get_polarity(text):
            textblob = TextBlob(str(text.encode('utf-8')))
            pol = textblob.sentiment.polarity
            return pol

        # lets apply the function
        df['polarity'] = df['Text'].apply(get_polarity)

        # Text Subjectivity
        # In natural language, subjectivity refers to expression of opinions, evaluations, feelings, and speculations and thus incorporates sentiment. Subjective text is further classified with sentiment or polarity.

        # Lets calculate the Subjectvity of the Reviews
        def get_subjectivity(text):
            textblob = TextBlob(str(text.encode('utf-8')))
            subj = textblob.sentiment.subjectivity
            return subj

        # lets apply the Function
        df['subjectivity'] = df['Text'].apply(get_subjectivity)

        ## lets summarize the Newly Created Features
        df[['length','polarity','subjectivity']].describe()

        # calculating the Character Count in the Reviews
        df['char_count'] = df['Text'].apply(len)

        # calculating the Word Count
        df['word_count'] = df['Text'].apply(lambda x: len(x.split()))

        # Calculating the Word Density
        df['word_density'] = df['char_count'] / (df['word_count']+1)

        # importing the List of Punctuations
        
        punctuation = string.punctuation

        # Calculating the Punctuation Count
        df['punctuation_count'] = df['Text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 

        ## lets summarize the Newly Created Features
        df[['char_count','word_count','word_density','punctuation_count']].describe()

        # Make Visulization
        # Lets calculate the Polarity of the Reviews
        def get_polarity(text):
            textblob = TextBlob(str(text))
            pol = textblob.sentiment.polarity
            if(pol==0):
                return "Neutral"
            elif(pol>0 and pol<=0.3):
                return "Weakly Positive"
            elif(pol>0.3 and pol<=0.6):
                return "Positive"
            elif(pol>0.6 and pol<=1):
                return "Strongly Positive"
            elif(pol>-0.3 and pol<=0):
                return "Weakly Negative"
            elif(pol>-0.6 and pol<=-0.3):
                return "Negative"
            elif(pol>-1 and pol<=-0.6):
                return "Strongly Negative"
            
        df['polarity'] = df['Text'].apply(get_polarity)

        df['polarity'].value_counts()
        neutral = 0
        wpositive = 0
        spositive = 0
        positive = 0
        negative = 0
        wnegative = 0
        snegative = 0
        polarity = 0
        for i in range(0,22):
            textblob = TextBlob(str(df['Text'][i]))
            polarity+= textblob.sentiment.polarity
            pol = textblob.sentiment.polarity
            if (pol == 0):  # adding reaction of how people are reacting to find average later
                neutral += 1
            elif (pol > 0 and pol <= 0.3):
                wpositive += 1
            elif (pol > 0.3 and pol <= 0.6):
                positive += 1
            elif (pol > 0.6 and pol <= 1):
                spositive += 1
            elif (pol > -0.3 and pol <= 0):
                wnegative += 1
            elif (pol > -0.6 and pol <= -0.3):
                negative += 1
            elif (pol > -1 and pol <= -0.6):
                snegative += 1

        # input for term to be searched and how many Review to search
        searchTerm = "22"
        NoOfTerms = 22

        # finding average reaction
        polarity = polarity / NoOfTerms
        polarity

        # To calculate the Prsentage
        def percentage(part, whole):
            temp = 100 * float(part) / float(whole)
            return format(temp, '.2f')

        # finding average of how people are reacting
        positive = percentage(positive, NoOfTerms)
        wpositive = percentage(wpositive, NoOfTerms)
        spositive = percentage(spositive, NoOfTerms)
        negative = percentage(negative, NoOfTerms)
        wnegative = percentage(wnegative, NoOfTerms)
        snegative = percentage(snegative, NoOfTerms)
        neutral = percentage(neutral, NoOfTerms)

        # printing out data
        print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " Review.")
        print()
        print("-----------------------------------------------------------------------------------------")
        print()
        print("General Report: ")

        if (polarity == 0):
            print("Neutral")
        elif (polarity > 0 and polarity <= 0.3):
            print("Weakly Positive")
        elif (polarity > 0.3 and polarity <= 0.6):
            print("Positive")
        elif (polarity > 0.6 and polarity <= 1):
            print("Strongly Positive")
        elif (polarity > -0.3 and polarity <= 0):
            print("Weakly Negative")
        elif (polarity > -0.6 and polarity <= -0.3):
            print("Negative")
        elif (polarity > -1 and polarity <= -0.6):
            print("Strongly Negative")

        print()
        print("------------------------------------------------------------------------------------------")
        # compound = round((1 + dd['compound'])/2, 2)
        return render_template('result.html', positive=positive, negative=negative, wpositive=wpositive, neutral=neutral, spositive = spositive, wnegative=wnegative, snegative=snegative)

    except requests.exceptions.RequestException as e:
        error = f"Error fetching the URL: {str(e)}"
        return render_template('error.html', error=error)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
