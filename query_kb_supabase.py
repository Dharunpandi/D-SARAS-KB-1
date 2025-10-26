import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase_client import supabase

# Fetch baseline KB from Supabase
response = supabase.table("baseline_kb").select("*").execute()
baseline_kb = response.data

questions = [faq["question"] for faq in baseline_kb]
answers = [faq["answer"] for faq in baseline_kb]
categories = [faq["category"] for faq in baseline_kb]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(questions)

def query_faq(user_query, top_k=3):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        results.append({
            "category": categories[i],
            "question": questions[i],
            "answer": answers[i],
            "similarity": float(similarities[i])
        })
    return results

if __name__ == "__main__":
    print("🔍 College FAQ Retrieval (Supabase + TF-IDF)")
    while True:
        q = input("\nEnter your question (or 'exit'): ")
        if q.lower() == "exit":
            break
        results = query_faq(q)
        print("\nTop Matches:")
        for r in results:
            print(f"\n[{r['category']}]")
            print(f"Q: {r['question']}")
            print(f"A: {r['answer']}")
            print(f"Similarity: {r['similarity']:.2f}")
