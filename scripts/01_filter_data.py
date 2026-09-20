import pandas as pd
import os

# Define our file paths
# Assuming you run this script from the root of your project folder
RAW_DATA_PATH = os.path.join("data", "raw", "twcs.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "apple_support_conversations.csv")

print("Loading data... (This might take a minute for 1M+ rows)")
# 1. Load the dataset
df = pd.read_csv(RAW_DATA_PATH) 

print("Splitting data...")
# 2. Get only the tweets written by AppleSupport (The answers)
apple_responses = df[df['author_id'] == 'AppleSupport']

# 3. Get only the inbound tweets (The questions sent BY customers TO brands)
inbound_tweets = df[df['inbound'] == True]

print("Merging data...")
# 4. Merge them together! 
# We do an "inner join" which means we only keep rows where we have BOTH the question and the answer.
merged_df = pd.merge(
    apple_responses, # The left table
    inbound_tweets,  # The right table
    left_on='in_response_to_tweet_id', # Column in the left table to match on
    right_on='tweet_id',               # Column in the right table to match on
    suffixes=('_brand', '_customer')   # Since both tables have a 'text' column, this renames them
)

# 5. Clean it up to only keep the columns we actually care about for training our agent
final_df = merged_df[['text_customer', 'text_brand']]

print(f"Success! We extracted {len(final_df)} AppleSupport conversations.")

# 6. Save this smaller dataset so we never have to load the giant file again!
final_df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Saved processed data to {PROCESSED_DATA_PATH}")
