import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')

# Microsoft ticker
microsoft_tickers = ['MSFT']

# Date range
start_date = '2025-01-02'
end_date = '2025-05-30'

# API endpoint
url = 'https://api.marketaux.com/v1/news/all'

# Store all records
records = []

# Fetch only Microsoft headlines
for symbol in microsoft_tickers:
    print(f"Fetching news for {symbol}...")
    page = 1
    while True:
        params = {
            'api_token': API_TOKEN,
            'symbols': symbol,
            'filter_entities': 'true',
            'limit': 100,
            'page': page,
            'published_after': start_date,
            'published_before': end_date
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching page {page} for {symbol}: {response.status_code}")
            break

        data = response.json()
        articles = data.get('data', [])

        if not articles:
            break  # No more articles

        for article in articles:
            headline = article.get('title')
            date = article.get('published_at')
            entities = article.get('entities', [])

            # Confirm article is about the correct Microsoft ticker
            matched_symbols = [e.get('symbol') for e in entities if e.get('symbol') == symbol]

            for match in matched_symbols:
                records.append({
                    'headline': headline,
                    'date': date,
                    'company': match
                })

        page += 1
        time.sleep(0.5)  # To avoid rate limits

# Create DataFrame
df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', ascending=False, inplace=True)
df.drop_duplicates(subset=['headline', 'date'], inplace=True)

df.to_csv('../../data/scraped-data/scraped-microsoft-headlines.csv')
