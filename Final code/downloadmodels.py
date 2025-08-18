from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Local directories to store the models
mini_lm_dir = "./models/all-MiniLM-L6-v2"
vlm_dir = "./models/moondream2"

print("📥 Downloading all-MiniLM-L6-v2...")
mini_lm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
mini_lm_model.save(mini_lm_dir)
print(f"✅ Saved MiniLM model to {mini_lm_dir}")

print("📥 Downloading vikhyatk/moondream2...")
vlm_model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
vlm_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)

vlm_model.save_pretrained(vlm_dir)
vlm_tokenizer.save_pretrained(vlm_dir)
print(f"✅ Saved VLM model to {vlm_dir}")
