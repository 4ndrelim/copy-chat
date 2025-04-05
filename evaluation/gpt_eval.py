import pandas as pd
import openai
import random

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

def gpt_sentiment(text):
    try:
        system_prompt = (
            "You are a sentiment analysis expert. You will be given a short text. "
            "Your task is to determine whether the sentiment is 'positive', 'negative', or 'neutral'."
            "Respond strictly with one of these three words only: 'positive', 'neutral', or 'negative'."
        )
        user_prompt = f"Text: {text}\nSentiment:"

        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        sentiment = response.choices[0].message.content.strip().lower()

        if sentiment in VALID_SENTIMENTS:
            return sentiment
        else:
            print(f"Invalid sentiment received: '{sentiment}' — skipping this row.")
            return None

    except Exception as e:
        print(f"Error processing text: '{text}'. Error: {e}")
        return None

if __name__ == "__main__":
    
    filename = input("Enter CSV filename: ")
    output_filename = input("Enter name of output file: ")
    key = input("Enter OpenAI API key: ")
    percent = float(input("Enter percentage of rows to send to GPT-4 for analysis (eg. Enter 10 for 10%): "))
    
    client = openai.OpenAI(api_key=key)
    df = pd.read_csv(filename, encoding="utf-8")
    
    sample_size = int(len(df) * percent / 100)
    sampled_df = df.sample(n=sample_size, random_state=42).copy()

    sampled_df["gpt_predict"] = sampled_df["combined"].apply(gpt_sentiment)

    sampled_df = sampled_df[sampled_df["gpt_predict"].notna()]

    sampled_df.to_csv(f"{output_filename}.csv", index=False)
    print(f"\nSaved output to: {output_filename}.csv")

    matches = (sampled_df['sentiment'] == sampled_df['gpt_predict'])
    match_percent = 100 * matches.sum() / len(sampled_df)

    print(f"Percentage of matching sentiments: {match_percent:.2f}%")
