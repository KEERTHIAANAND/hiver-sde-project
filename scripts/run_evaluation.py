import os
import sys
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import SupportAgent
from src.baseline_agent import SimpleBaselineAgent

def safe_grade_reply(client, customer_tweet, drafted_reply):
    if not drafted_reply: return 1
    prompt = f"""
    You are a strict QA Manager for AppleSupport.
    Customer tweeted: "{customer_tweet}"
    Agent drafted this reply: "{drafted_reply}"
    Score the drafted reply from 1 to 5 based on Tone, Relevance, and Safety.
    Output ONLY a single integer from 1 to 5.
    """
    while True:
        try:
            response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
            return int(response.text.strip())
        except Exception as e:
            print(f"Judge Error: {e}... pausing for 30s...")
            time.sleep(30)

def safe_handle(agent, tweet):
    while True:
        try:
            return agent.handle_tweet(tweet)
        except Exception as e:
            print(f"Agent Error: {e}... pausing for 30s...")
            time.sleep(30)

def run_evaluation():
    print("--- Starting Bulletproof Evaluation Showdown ---")
    load_dotenv()
    rag_agent = SupportAgent()
    baseline_agent = SimpleBaselineAgent()
    judge_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    df = pd.read_csv(os.path.join("data", "golden_eval_set.csv"))
    
    # Grab 20 random rows 
    test_set = df.sample(20, random_state=42) 
    
    results = {"baseline": {"correct": 0, "score": 0}, "rag": {"correct": 0, "score": 0}}
    total_tested = 0
    
    # Resetting the index so it counts nicely from 1 to 20!
    for index, row in enumerate(test_set.itertuples()):
        tweet = row.text_customer
        human_label = str(row.intent).strip()
        print(f"\nEvaluating Tweet {index+1}/20...")
        
        # 1. Run Baseline Agent
        base_res = safe_handle(baseline_agent, tweet)
        if base_res.get('intent') == human_label: results["baseline"]["correct"] += 1
        results["baseline"]["score"] += safe_grade_reply(judge_client, tweet, base_res.get('draft_reply', ''))
        time.sleep(15) 
        
        # 2. Run RAG Agent
        rag_res = safe_handle(rag_agent, tweet)
        if rag_res.get('intent') == human_label: results["rag"]["correct"] += 1
        results["rag"]["score"] += safe_grade_reply(judge_client, tweet, rag_res.get('draft_reply', ''))
        time.sleep(15) 
        
        total_tested += 1

    print("\n========================================")
    print("      FINAL EVALUATION REPORT           ")
    print("========================================")
    print(f"Total Evaluated: {total_tested}")
    print("\n[SIMPLE BASELINE AGENT (No RAG)]")
    print(f"Intent Accuracy: {(results['baseline']['correct']/total_tested)*100:.1f}%")
    print(f"Avg Reply Score: {(results['baseline']['score']/total_tested):.1f} / 5.0")
    
    print("\n[FINAL RAG AGENT (TF-IDF + LLM)]")
    print(f"Intent Accuracy: {(results['rag']['correct']/total_tested)*100:.1f}%")
    print(f"Avg Reply Score: {(results['rag']['score']/total_tested):.1f} / 5.0")
    print("========================================")

if __name__ == "__main__":
    run_evaluation()
