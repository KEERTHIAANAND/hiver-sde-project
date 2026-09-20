import os
import json
from google import genai
from dotenv import load_dotenv

class SimpleBaselineAgent:
    """
    A simple baseline agent. It uses an LLM, but has NO historical RAG context.
    We will use this to prove that our Retriever actually improves performance.
    """
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        
    def handle_tweet(self, new_tweet):
        prompt = f"""
        You are an AI customer support agent for AppleSupport.
        A customer just tweeted the following: "{new_tweet}"
        
        Your job is to output a JSON object with EXACTLY the following 4 keys:
        1. "intent": Classify the tweet into ONE of: [software_os_issue, hardware_power_issue, account_login_issue, general_inquiry, unknown].
        2. "action": Decide whether to "auto_handle" or "escalate".
        3. "reason": A 1-sentence reason for your action decision.
        4. "draft_reply": Write a helpful reply to the customer.
        
        Output purely valid JSON. No markdown formatting.
        """
        
        response = self.client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        
        try:
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_text)
        except Exception:
            return {"error": "Failed to parse JSON", "raw_response": response.text}