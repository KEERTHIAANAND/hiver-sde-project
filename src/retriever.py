import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SupportRetriever:
    """
    A simple Retriever using TF-IDF to find historical brand responses.
    This simulates a Vector Database in a RAG pipeline.
    """
    def __init__(self, data_path=None):
        # We use relative paths so it works no matter where the script is run from
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "processed", "apple_support_conversations.csv")
            
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Load data and build index
        self._load_and_index()

    def _load_and_index(self):
        """Loads the dataset and builds the TF-IDF index."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        self.historical_questions = df['text_customer'].fillna("").tolist()
        self.historical_answers = df['text_brand'].fillna("").tolist()
        
        # Build the mathematical search index
        self.tfidf_matrix = self.vectorizer.fit_transform(self.historical_questions)

    def search(self, new_tweet, top_k=2):
        """Finds the top_k most similar historical questions."""
        new_tweet_vector = self.vectorizer.transform([new_tweet])
        similarities = cosine_similarity(new_tweet_vector, self.tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "past_customer_question": self.historical_questions[idx],
                "past_apple_answer": self.historical_answers[idx]
            })
        return results
