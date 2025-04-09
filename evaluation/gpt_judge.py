import pandas as pd
import openai

# Define the allowed sentiment values.
VALID_SENTIMENTS = {"positive", "neutral", "negative"}

def gpt_sentiment(text, client):
    """
    Sends the provided text to a GPT-based sentiment analysis model 
    and returns one of 'positive', 'neutral', or 'negative'.

    Parameters:
    - text (str): The text to analyze.
    - client: The initialized OpenAI client.

    Returns:
    - str or None: The sentiment result if valid, otherwise None.
    """
    try:
        system_prompt = (
            "You are a sentiment analysis expert. You will be given a short text. "
            "Your task is to determine whether the sentiment is 'positive', 'negative', or 'neutral'. "
            "Respond strictly with one of these three words only."
        )
        user_prompt = f"Text: {text}\nSentiment:"

        response = client.ChatCompletion.create(
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
            print(f"Invalid sentiment received: '{sentiment}' for text: {text}")
            return None
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None

def main():
    # User inputs for filenames and API key.
    filename = input("Enter CSV filename: ").strip()
    output_filename = input("Enter name of output file (without extension): ").strip()
    key = input("Enter OpenAI API key: ").strip()

    # Set the API key.
    openai.api_key = key

    # Load the CSV.
    try:
        df = pd.read_csv(filename, encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {filename}. Error: {e}")
        return

    # Process each row to determine the sentiment.
    df["judged_value"] = df["combined"].apply(lambda text: gpt_sentiment(text, openai))

    # Determine if the original sentiment matches the GPT-judged sentiment.
    # The comparison is made ignoring case and any extra spaces.
    df["match"] = df.apply(
        lambda row: "yes" if row["sentiment"].strip().lower() == row["judged_value"] else "no", 
        axis=1
    )

    # Save the updated DataFrame to a new CSV file.
    df.to_csv(f"{output_filename}.csv", index=False)
    print(f"\nSaved output to: {output_filename}.csv")

if __name__ == "__main__":
    main()
