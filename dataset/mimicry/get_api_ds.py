"""
This script extracts tweets for a given username from tweets.db and saves them as a CSV file with optional end tokens. 

Run with python script_name.py <username> [--endtoken].
"""

import argparse
import sqlite3
import pandas as pd
import json

def main(username, endtoken):
    # Connect to the database
    conn = sqlite3.connect('tweets.db')
    cursor = conn.cursor()

    # Map username to author_id
    cursor.execute("SELECT DISTINCT author_id, username FROM pagination_state")
    user_data = cursor.fetchall()
    username_to_id = {uname: aid for aid, uname in user_data}

    if username not in username_to_id:
        print(f"Username '{username}' not found in pagination_state.")
        print("No csv file created")
        conn.close()
        return

    author_id = username_to_id[username]

    # Load tweets table
    df = pd.read_sql_query("SELECT * FROM tweets", conn)
    conn.close()

    if author_id not in df['author_id'].values:
        print(f"Author id '{author_id}' not found in tweets table.")
        print("No csv file created")
        return

    df = df[df['author_id'] == author_id].reset_index(drop=True)

    output_df = pd.DataFrame()
    output_df['id'] = df['id']
    output_df['text'] = df['text']
    output_df['date'] = df['created_at']

    if endtoken:
        output_df['text'] = output_df['text'].astype(str) + "|<end>|"

    # Static NA fields
    output_df['isRetweet'] = "na"
    output_df['isDeleted'] = "na"
    output_df['device'] = "na"

    # Extract from json_data
    parsed_json = df['json_data'].apply(json.loads)
    output_df['favorites'] = parsed_json.apply(lambda x: x['public_metrics']['like_count'])
    output_df['retweets'] = parsed_json.apply(lambda x: x['public_metrics']['retweet_count'])
    output_df['isFlagged'] = parsed_json.apply(lambda x: 't' if x.get('possibly_sensitive') else 'f')

    # Export to CSV
    fname = f"{username}_endtoken.csv" if endtoken else f"{username}.csv"
    output_df.to_csv(fname, index=False)
    print(f"{fname} file created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tweet CSV for a user")
    parser.add_argument("username", type=str, help="Username of X user")
    parser.add_argument("--endtoken", action="store_true", help="Add end token to text")
    args = parser.parse_args()

    main(args.username, args.endtoken)
