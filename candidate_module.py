import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase_client import supabase

# 1️⃣ Load removed FAQs (candidates)
with open("deepseek_json_20251022_cdf97d.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Recreate baseline and candidate split (same as before)
import random
num_remove = int(len(data) * 0.25)
remove_indices = random.sample(range(len(data)), num_remove)
candidate_faqs = [faq for i, faq in enumerate(data) if i in remove_indices]

# 2️⃣ Fetch baseline KB from Supabase
response = supabase.table("baseline_kb").select("*").execute()
baseline_kb = response.data

baseline_questions = [faq["question"] for faq in baseline_kb]

# 3️⃣ TF-IDF similarity between candidates and baseline
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(baseline_questions)

candidate_scores = []
for faq in candidate_faqs:
    query_vec = vectorizer.transform([faq["question"]])
    baseline_vecs = vectorizer.transform(baseline_questions)
    sims = cosine_similarity(query_vec, baseline_vecs).flatten()
    max_sim = float(np.max(sims))  # best match similarity
    faq["similarity_score"] = max_sim
    faq["state"] = 0
    candidate_scores.append(faq)

# 4️⃣ Upload to Supabase
def upload_candidates():
    for entry in candidate_scores:
        supabase.table("candidate_kb").insert({
            "question": entry["question"],
            "answer": entry["answer"],
            "category": entry["category"],
            "state": entry["state"],
            "similarity_score": entry["similarity_score"]
        }).execute()
    print(f"✅ Uploaded {len(candidate_scores)} candidates to Supabase")

upload_candidates()
print("🎯 Candidate Knowledge Module complete!")
