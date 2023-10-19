# Customer Product Review Sentiment Analysis

## Project Overview

The "Customer Product Review Sentiment Analyzer" project is a Python application that performs sentiment analysis on Amazon product reviews. The application scrapes data from Amazon product pages and provides sentiment analysis results in categories such as "Strongly Positive," "Positive," "Weakly Positive," "Neutral," "Strongly Negative," "Negative," and "Weakly Negative." This project is useful for understanding customer sentiments towards products on Amazon.


## Features

- Scrape Amazon product reviews: The application collects product reviews from Amazon web pages using BeautifulSoup.
- Sentiment Analysis: It performs sentiment analysis on the collected reviews using TextBlob library and categorizes them into various sentiment levels.
- User-Friendly Results: Results are presented in a form of pie chart and a table.

## Prerequisites

- Python 3.7 or higher
- Python libraries used (list dependencies)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/ManasiPatil2117/Customer-Product-Review-Sentiment-Analyzer.git

2. Install the required files:

   ```bash
   pip install -r requirements.txt

3. Run the application:

   ```bash
   python app.py

## Usage
   1. Enter the URL of the Amazon product review page you want to analyze.
   2. The application will scrape the reviews and store it locally in a CSV file. It will then preprocess the data from CSV and perform sentiment analysis on it.
   3. The results will be displayed, showing the distribution of sentiments in the reviews.

## Contributing
Contributions to this project are welcome. 

To contribute:
   - Fork the repository.
   - Create a new branch for your feature or bug fix.
   - Make your changes and test them.
   - Submit a pull request with a clear description of your changes.


