import os
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

class MedGemmaLoader:
    """
    Loader for MedGemma model using local Hugging Face cache.
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
    
    def load_model(self):
        """
        Download (if needed) and load the model and processor with 4-bit quantization.
        
        Returns:
            model, processor
        """
        print(f"Loading MedGemma model: {self.model_id}...")
        
        # Quantization config for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            # MedGemma is typically based on PaliGemma or similar architecture.
            # Using AutoModelForCausalLM is standard for VLM/LLMs in transformers recently if supported,
            # but PaliGemma specifically might need PaliGemmaForConditionalGeneration.
            # 'google/medgemma-1.5-4b-it' is a PaliGemma fine-tune.
            # We will use AutoModelForCausalLM as it often auto-maps, but let's be safe with auto classes.
            # If it fails, we might need specific imports.
            
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("Model loaded successfully.")
            return model, processor

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
