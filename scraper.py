import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_coffee_reviews(num_pages=5):
    reviews = []
    base_url = "https://www.coffeereview.com/review/page/{}/"
    
    for page in range(1, num_pages + 1):
        try:
            response = requests.get(base_url.format(page))
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all review containers
            review_elements = soup.find_all('article', class_='review')
            
            for review in review_elements:
                title = review.find('h2').text.strip() if review.find('h2') else ''
                rating = review.find('div', class_='rating').text.strip() if review.find('div', class_='rating') else ''
                text = review.find('div', class_='entry-content').text.strip() if review.find('div', class_='entry-content') else ''
                
                reviews.append({
                    'title': title,
                    'rating': rating,
                    'text': text
                })
            
            time.sleep(1)  # Be respectful with scraping
            
        except Exception as e:
            print(f"Error on page {page}: {str(e)}")
            continue
    
    return pd.DataFrame(reviews)