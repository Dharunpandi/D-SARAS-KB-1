import json
import random
from supabase_client import supabase

# Load your existing data
with open("deepseek_json_20251022_cdf97d.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Add state = 2
for entry in data:
    entry["state"] = 2

# Create Golden KB
golden_kb = data
print(f"✅ Golden KB ready with {len(golden_kb)} entries")

# Create Baseline KB (remove 25%)
num_remove = int(len(data) * 0.25)
remove_indices = random.sample(range(len(data)), num_remove)
baseline_kb = [faq for i, faq in enumerate(data) if i not in remove_indices]
print(f"✅ Baseline KB ready with {len(baseline_kb)} entries")

# 🔼 Upload both to Supabase
def upload_to_supabase(table_name, dataset):
    for entry in dataset:
        supabase.table(table_name).insert({
            "category": entry["category"],
            "question": entry["question"],
            "answer": entry["answer"],
            "state": entry["state"]
        }).execute()
    print(f"✅ Uploaded {len(dataset)} records to '{table_name}'")

# Create tables in Supabase: golden_kb, baseline_kb
upload_to_supabase("golden_kb", golden_kb)
upload_to_supabase("baseline_kb", baseline_kb)
