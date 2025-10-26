import numpy as np
from supabase_client import supabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


baseline_response = supabase.table("baseline_kb").select("*").execute()
baseline = baseline_response.data

# Fetch candidate KB
candidate_response = supabase.table("candidate_kb").select("*").execute()
candidates = candidate_response.data

baseline_questions=[q["question"] for q in baseline]
vectorizer=TfidfVectorizer(stop_words="english")
tfidf_baseline=vectorizer.fit_transform(baseline_questions)

def compute_improvement(candidate_question):
    cand_vec=vectorizer.transform([candidate_question])
    sims=cosine_similarity(cand_vec,tfidf_baseline).flatten()
    max_sim=np.max(sims)
    return 1-max_sim

def run_bfa():
    best_improvement=-1
    best_candiate=None
    for cand in candidates:
        improvement=compute_improvement(cand["question"])
        if improvement>best_improvement:
            best_improvement=improvement
            best_candidate=cand
    if best_candidate:
        supabase.table("baseline_kb").insert({
            "category": best_candidate["category"],
            "question": best_candidate["question"],
            "answer": best_candidate["answer"],
            "state": 2
        }).execute()

        # Update candidate state
        supabase.table("candidate_kb").update({"state": 2}).eq("id", best_candidate["id"]).execute()

        # Log the update
        supabase.table("update_logs").insert({
            "candidate_id": best_candidate["id"],
            "added_to_kb": True,
            "method": "BFA",
            "improvement": float(best_improvement)
        }).execute()

        print(f"✅ BFA Added: {best_candidate['question'][:60]}... | Improvement: {best_improvement:.3f}")


def run_dfa():
    for cand in candidates:
        improvement = compute_improvement(cand["question"])
        if improvement > 0.3:  # threshold for adding
            supabase.table("baseline_kb").insert({
                "category": cand["category"],
                "question": cand["question"],
                "answer": cand["answer"],
                "state": 2
            }).execute()

            supabase.table("candidate_kb").update({"state": 2}).eq("id", cand["id"]).execute()

            supabase.table("update_logs").insert({
                "candidate_id": cand["id"],
                "added_to_kb": True,
                "method": "DFA",
                "improvement": float(improvement)
            }).execute()

            print(f"✅ DFA Added: {cand['question'][:60]}... | Improvement: {improvement:.3f}")

# Choose which algorithm to run
if __name__ == "__main__":
    print("Select update method: 1) BFA  2) DFA")
    choice = input("Enter choice: ")
    if choice == "1":
        run_bfa()
    else:
        run_dfa()