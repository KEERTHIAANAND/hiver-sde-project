import os
import json
from google import genai
from dotenv import load_dotenv

# Import our custom retriever
from src.retriever import SupportRetriever

class SupportAgent:
    def __init__(self):
        # Load API keys
        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        
        # Initialize our Retriever
        print("Initializing AI Agent and building search index...")
        self.retriever = SupportRetriever()
        
    def handle_tweet(self, new_tweet):
        # 1. Retrieve similar past conversations
        top_matches = self.retriever.search(new_tweet, top_k=2)
        
        # 2. Format historical context
        historical_context = ""
        for i, match in enumerate(top_matches):
            historical_context += f"Example {i+1}:\nCustomer: {match['past_customer_question']}\nAppleSupport: {match['past_apple_answer']}\n\n"
            
        # 3. Build the prompt
        prompt = f"""
        You are an expert AI customer support agent for AppleSupport.
        A customer just tweeted the following:
        "{new_tweet}"
        
        Here is how human AppleSupport agents have handled very similar tweets in the past:
        {historical_context}
        
        Your job is to output a JSON object with EXACTLY the following 4 keys:
        1. "intent": Classify the tweet into ONE of: [software_os_issue, hardware_power_issue, account_login_issue, general_inquiry, unknown].
        2. "action": Decide whether to "auto_handle" (if you can provide troubleshooting steps) or "escalate" (if it requires a human).
        3. "reason": A 1-sentence reason for your action decision.
        4. "draft_reply": Write a reply to the customer. Adopt the tone of the historical examples.
        
        Output purely valid JSON. No markdown formatting.
        """
        
        # 4. Call Gemini
        response = self.client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        
        # 5. Parse and return JSON
        try:
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_text)
        except Exception:
            return {"error": "Failed to parse JSON", "raw_response": response.text}
