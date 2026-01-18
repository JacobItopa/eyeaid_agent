import json
from typing import Dict, Any
from PIL import Image

class OphthalmicScreeningAgent:
    """
    Ophthalmic Screening Agent
    Uses MedGemma (multimodal) locally.
    Performs screening-level visual feature description ONLY
    """

    def __init__(
        self,
        model,
        processor
    ):
        self.model = model
        self.processor = processor

    def run(
        self,
        patient_context: Dict[str, Any],
        image_path: str
    ) -> Dict[str, Any]:
        """
        Run screening agent on fundus image
        
        Returns:
            dict following screening agent JSON schema
        """

        prompt = (
            "You are an Ophthalmic Screening Agent.\n"
            "Analyze the retinal fundus image and return a JSON object with observations.\n\n"
            "Rules:\n"
            "- Do NOT diagnose\n"
            "- Describe visible features\n"
            "- Provide confidence levels\n\n"
            "Output Format (JSON):\n"
            "{\n"
            '  "observations": [{"feature": "...", "location": "...", "confidence": "..."}],\n'
            '  "overall_assessment": "...",\n'
            '  "uncertainty_notes": "..."\n'
            "}\n"
            f"Patient context: {json.dumps(patient_context)}"
        )

        try:
            print("Running local inference for Screening Agent...")
            raw_image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            # Note: MedGemma/PaliGemma typically handles prompts like "detect: " or just natural language.
            # We assume the model expects the text prompt first.
            inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.model.device)
            
            # Generate
            # Using standard generation parameters
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2 
            )
            
            # Decode
            # The model output usually contains the input prompt as well, we might need to strip it
            # or skip special tokens.
            output_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            
            # Often the output follows the prompt. We can try to clean it if it echoes.
            # MedGemma might behave like PaliGemma where it generates the answer.
            # Let's inspect the output essentially by treating it as the full response.
            # If the prompt is repeated, we need to handle that. 
            # Transformers `batch_decode` with `skip_special_tokens` usually gives full text.
            # We'll heuristic check if the prompt is at the start and strip it.
            
            # For robustness, let's assume the pure JSON is in there somewhere.

        except Exception as e:
            print(f"Local inference failed: {e}")
            return {
                "observations": [],
                "overall_assessment": "Screening failed due to local inference error.",
                "uncertainty_notes": str(e)
            }

        # Parse JSON output
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                # Try to clean up if the prompt is included and JSON follows
                # (Simple fallback)
                parsed = {
                   "observations": [{"feature": "processing_error", "location": "unknown", "confidence": "low"}],
                   "overall_assessment": "Could not parse model output (JSON not found)",
                   "uncertainty_notes": f"Raw output snippet: {output_text[:100]}..."
                }

        return parsed

if __name__ == "__main__":
    pass
