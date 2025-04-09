import pandas as pd
import openai
from pathlib import Path

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

def gpt_sentiment(text, client):
    """
    Sends provided text to GPT model and returns one of 'positive', 'neutral', or 'negative'.

    Params:
    - text (str): The text to be analyzed.
    - client: initialized OpenAI client.

    Return:
    - str or None: One of the valid sentiments if response is valid, otherwise None.
    """
    try:
        system_prompt = (
            "You are a sentiment analysis expert. You will be given a short text. "
            "Your task is to determine whether the sentiment is 'positive', 'negative', or 'neutral'. "
            "Respond strictly with one of these three words only."
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
            print(f"Invalid sentiment received: '{sentiment}' for text: {text}")
            return None
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None

def main():
    # Get user inputs: path for CSV, output file name, and OpenAI API key.
    input_path = input("Enter full path to input CSV file (no need for quotation marks): ").strip()
    output_filename = input("Enter output file name (without extension): ").strip()
    key = input("Enter OpenAI API key: ").strip()
    
    # OpenAI API key.
    openai.api_key = key

    # Input file path.
    csv_path = Path(input_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    # Load CSV.
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {csv_path}. Error: {e}")
        return
    
    # Filter out rows where completion is empty or NaN
    df_filtered = df[df['completion'].notna() & (df['completion'] != '')]
    n_skipped = len(df) - len(df_filtered)
    if n_skipped > 0:
        print(f"Skipped {n_skipped} rows with empty completion")

    # Judge with GPT on combined column
    df_filtered["judged_value"] = df_filtered["combined"].apply(lambda text: gpt_sentiment(text, openai))

    # Check against original sentiment
    df_filtered["match"] = df_filtered.apply(
        lambda row: "yes" if row["sentiment"].strip().lower() == row["judged_value"] else "no", axis=1
    )

    # Save to new CSV.
    output_path = Path(f"{output_filename}.csv")
    try:
        df_filtered.to_csv(output_path, index=False)
        print(f"\nSaved output to: {output_path}")
    except Exception as e:
        print(f"Error saving output file: {output_path}. Error: {e}")

    # === Calculate statistics ===
    valid_results = df_filtered[df_filtered['judged_value'].notna()]
    total_processed = len(valid_results)
    matches = valid_results[valid_results['match'] == 'yes']
    match_count = len(matches)
    match_percent = 100 * match_count / total_processed if total_processed > 0 else 0
    
    # write to txt file
    with open("JudgeSummary.txt", "w", encoding="utf-8") as f:
        f.write("\nStatistics:\n")
        f.write(f"Total rows processed: {total_processed}\n")
        f.write(f"Matching sentiments: {match_count}/{total_processed} ({match_percent:.2f}%)\n")
        
        # Breakdown by sentiment type
        f.write("\nBreakdown by sentiment type:\n")
        for sentiment in VALID_SENTIMENTS:
            sentiment_rows = valid_results[valid_results['sentiment'] == sentiment]
            sentiment_count = len(sentiment_rows)
            if sentiment_count > 0:
                sentiment_matches = sentiment_rows[sentiment_rows['match'] == 'yes']
                sentiment_match_count = len(sentiment_matches)
                sentiment_match_percent = 100 * sentiment_match_count / sentiment_count
                f.write(f"  {sentiment.capitalize()}: {sentiment_match_count}/{sentiment_count} ({sentiment_match_percent:.2f}%)\n")
        
        # Confusion Matrix
        f.write("\nConfusion Matrix:\n")
        f.write("Intended vs. Judged sentiment:\n")
        
        for intended in VALID_SENTIMENTS:
            row_data = []
            for judged in VALID_SENTIMENTS:
                count = len(valid_results[(valid_results['sentiment'] == intended) & 
                                        (valid_results['judged_value'] == judged)])
                row_data.append(str(count))
            f.write(f"  {intended.ljust(8)}: {' | '.join(s.rjust(7) for s in row_data)}\n")
    print("Summary statistics have been saved to 'JudgeSummary.txt'.")


if __name__ == "__main__":
    main()