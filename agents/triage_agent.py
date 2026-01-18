import json
from typing import Dict, Any

class RiskAndTriageAgent:
    """
    Risk & Triage Agent
    Uses MedGemma (text-only) locally.
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
        screening_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute triage reasoning
        """

        prompt = (
            "You are a Risk Stratification and Triage Agent.\n"
            "Based on the inputs, recommend a triage level (low, medium, high).\n\n"
            "Return STRICT JSON:\n"
            "{\n"
            '  "triage_level": "...",\n'
            '  "reasoning": "...",\n'
            '  "recommended_action": "..."\n'
            "}\n\n"
            f"Patient context: {json.dumps(patient_context)}\n"
            f"Screening observations: {json.dumps(screening_results)}"
        )
        
        try:
            print("Running local inference for Triage Agent...")
            # MedGemma expects text inputs to still go through the processor or tokens
            # If the model is a PaliGemma type, it usually expects 'input_ids' and 'attention_mask'
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2
            )
            
            output_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        except Exception as e:
            print(f"Local inference failed: {e}")
            return {
                "triage_level": "high",
                "reasoning": f"Local inference error: {e}",
                "recommended_action": "Refer to specialist"
            }
            
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            try:
                import re
                json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found")
            except (json.JSONDecodeError, ValueError, Exception):
                parsed = {
                    "triage_level": "high",
                    "reasoning": f"Parsing failure.",
                    "recommended_action": "Refer to specialist"
                }

        return parsed

if __name__ == "__main__":
    pass
