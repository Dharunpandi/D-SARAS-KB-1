from supabase_client import supabase
from query_kb_supabase import query_faq

# Load Golden KB from Supabase
response = supabase.table("golden_kb").select("*").execute()
golden_kb = response.data

correct = 0
total = 0

for faq in golden_kb:
    query = faq["question"]
    top_results = query_faq(query, top_k=1)
    if top_results and top_results[0]["question"].strip().lower() == query.strip().lower():
        correct += 1
    total += 1

accuracy = (correct / total) * 100
print(f"✅ Retrieval Accuracy: {accuracy:.2f}% ({correct}/{total})")
