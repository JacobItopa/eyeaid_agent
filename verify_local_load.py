import os
import torch
from dotenv import load_dotenv
from models.medgemma_loader import MedGemmaLoader

load_dotenv()

def verify():
    print("Verifying local model loading...")
    
    # Check for token (Gemma models are gated)
    token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        print("HF Token found in environment.")
        # Ensure huggingface_hub uses it
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    else:
        print("WARNING: No HF Token found. If the model is gated, this might fail.")
    
    try:
        loader = MedGemmaLoader()
        model, processor = loader.load_model()
        print("Model loaded successfully!")
        
        # Simple test
        prompt = "test"
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        print("Input prepared. Running generation...")
        
        out = model.generate(**inputs, max_new_tokens=10)
        res = processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"Test generation result: {res}")
        print("Verification PASSED.")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")

if __name__ == "__main__":
    verify()
