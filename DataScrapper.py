
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}



reviews_url = 'https://www.amazon.com/Legendary-Whitetails-Journeyman-Jacket-Tarmac/product-reviews/B013KW38RQ/'
#reviews_url = input("ENTER URL: ")

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
            'Description' : description
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
df_reviews.to_csv('reviews.csv', index=False)