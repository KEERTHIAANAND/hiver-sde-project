import os
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv

def auto_label_dataset():
    print("Starting AI-Assisted Labeling...")
    
    # 1. Load the empty golden set you created earlier
    data_path = os.path.join("data", "golden_eval_set.csv")
    df = pd.read_csv(data_path)
    
    # Ensure the intent column exists
    if 'intent' not in df.columns:
        df['intent'] = ""
        
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    labeled_count = 0
    
    for index, row in df.iterrows():
        # Skip rows you already labeled manually
        if pd.notna(row['intent']) and str(row['intent']).strip() != "":
            continue
            
        tweet = row['text_customer']
        
        prompt = f"""
        Classify this AppleSupport customer tweet into EXACTLY ONE of these categories:
        [software_os_issue, hardware_power_issue, account_login_issue, general_inquiry, unknown]
        
        Tweet: "{tweet}"
        Output ONLY the category name. No other text.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            predicted_intent = response.text.strip()
            
            # Save the prediction to our dataframe
            df.at[index, 'intent'] = predicted_intent
            labeled_count += 1
            print(f"Row {index+1}: {predicted_intent}")
            
            # Sleep to respect free tier rate limits (15 API calls per minute)
                        # Sleep 5 seconds to guarantee we stay under 15 requests per minute
            time.sleep(5) 
            
        except Exception as e:
            print(f"Rate limit hit! Waiting 65 seconds to reset quota... (Row {index+1})")
            # If we hit the limit, we MUST wait a full minute for Google to forgive us
            time.sleep(65) 

            
        # Save every 10 rows just in case it crashes
        if labeled_count % 10 == 0:
            df.to_csv(data_path, index=False)
            
    # Final save
    df.to_csv(data_path, index=False)
    print("\nFinished pre-labeling! Open data/golden_eval_set.csv to verify the results.")

if __name__ == "__main__":
    auto_label_dataset()
