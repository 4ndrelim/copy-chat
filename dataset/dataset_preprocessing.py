"""
This file contains preprocess function that could be done on the 'text' column.
Mainly functions to deal with taggings and links (Could not decide if I should clean/generate the datasets without them so I just included them and provide the option to remove)
You will may still need to do preprocessing like case folding, removal of stopwords etc.
Feel free to add or update the functions.

For example
df = pd.read_csv("train_vanilla", encoding="latin1")
df['text'] = df['text'].apply(replace_tagging)

Do check for duplicates after removing/replacing taggings/links
df = df.drop_duplicates()

Do check for emppty values after removing/replacing taggings/links
df = df.dropna(subset=['text'])
"""

import pandas as pd
import re

def rm_startend_whitespace(text):
    return text.strip()


####### Dealing with taggings #######
# Abit tricky because '@' is also used to mean 'at' so I just decided that if there is a space '@ username' then length of 'username' needs to be at least 6 to be considered username
def replace_tagging(text):
    # Replaces all taggings with <username>, a standardized placeholder to represent all usernames
    pattern = r'@(?:\w+|(?:\s+\w{6,}))'
    text = re.sub(pattern, '<username>', text)
    return text.strip()

def remove_tagging(text):
    # Remove all taggings
    pattern = r'@(?:\w+|(?:\s+\w{6,}))(?:\s*:|)'
    text = re.sub(pattern, '', text)
    return text.strip()


####### Dealing with links #######
# Assume links appear in either one of these forms "http://", "https://", "www."
def replace_link(text):
    # Replaces all links with <link>, a standardized placeholder to represent all links
    pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    text = re.sub(pattern, '<link>', text)
    return text.strip()

def remove_link(text):
    # Remove all links
    pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    text = re.sub(pattern, '', text)
    return text.strip()