import tweepy
import pandas as pd
from datetime import datetime

# Twitter API credentials
API_KEY = 'Pla9hWD0iM3BFErKGVmwF1Plo'
API_KEY_SECRET = 'h2Amv1WWAnrzHmIecZYkEMdMJp1LEGCuZ2y661TJUkxeS4SZcb'
ACCESS_TOKEN = '1591906146910339074-36xM1vzQ2VM0R0nsGUDWh7GwhvSf43'
ACCESS_TOKEN_SECRET = 'B5Bn4QdgdmuH1QQltQkV7kOVc24xqKaj4Pc4IJFQh9nQF'

# Authenticate with Twitter API
def authenticate_twitter():
    auth = tweepy.OAuth1UserHandler(API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

# Scrape tweets based on a query
def scrape_tweets(api, query, num_tweets):
    tweets = []
    try:
        # Use Tweepy's Cursor to paginate through tweets
        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang='en').items(num_tweets):
            tweets.append({
                'target': 2,  # Placeholder for sentiment (2 = neutral)
                'ids': tweet.id,
                'date': tweet.created_at.strftime('%a %b %d %H:%M:%S %Z %Y'),  # Format date
                'flag': query if query else 'NO_QUERY',  # Use query as flag
                'user': tweet.user.screen_name,
                'text': tweet.full_text
            })
        print(f"Scraped {len(tweets)} tweets.")
    except tweepy.TweepyException as e:
        print(f"Error: {e}")
    return tweets

# Save scraped tweets to a CSV file
def save_tweets_to_csv(tweets, file_path):
    df = pd.DataFrame(tweets)
    df.to_csv(file_path, index=False)
    print(f"Tweets saved to {file_path}")

# Main function to scrape tweets
def scrape_and_save_tweets():
    # Authenticate with Twitter API
    api = authenticate_twitter()

    # Define search query and number of tweets to scrape
    query = ""  # Replace with your desired query
    num_tweets = 1000  # Number of tweets to scrape

    # Scrape tweets
    tweets = scrape_tweets(api, query, num_tweets)

    # Save tweets to CSV
    save_tweets_to_csv(tweets, 'scraped_tweets.csv')

# Run the scraper
if __name__ == "__main__":
    scrape_and_save_tweets()