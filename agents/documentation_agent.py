import json
from typing import Dict, Any

class ClinicalDocumentationAgent:
    """
    Clinical Documentation Agent
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
        intake_results: Dict[str, Any],
        screening_results: Dict[str, Any],
        triage_results: Dict[str, Any]
    ) -> str:
        """
        Generate clinical documentation text
        """

        prompt = (
            "You are a Clinical Documentation Agent.\n"
            "Generate structured ophthalmic screening notes.\n\n"
            "Output format:\n"
            "Screening Summary:\n"
            "- Patient Summary:\n"
            "- Image Quality:\n"
            "- Screening Observations:\n"
            "- Triage Recommendation:\n\n"
            "Provide the content for these sections based on the inputs below:\n"
            f"Patient context: {json.dumps(patient_context)}\n"
            f"Intake & image quality: {json.dumps(intake_results)}\n"
            f"Screening findings: {json.dumps(screening_results)}\n"
            f"Triage decision: {json.dumps(triage_results)}"
        )
        
        try:
            print("Running local inference for Documentation Agent...")
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.3
            )
            documentation = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        except Exception as e:
            print(f"Local inference failed: {e}")
            return (
                "Screening Summary (System Generated Fallback):\n"
                "- Patient Summary: Context available in patient data.\n"
                "- Image Quality: " + str(intake_results.get('image_quality', 'Unknown')) + "\n"
                "- Screening Observations: Automated screening failed due to local inference error.\n"
                "- Triage Recommendation: HIGH RISK (Safety Fallback) - Please review manually."
            )
        
        # Clean up output
        if "Screening Summary:" in documentation:
             parts = documentation.split("Screening Summary:")
             if len(parts) > 1:
                 documentation = "Screening Summary:" + parts[-1]

        return documentation.strip()

if __name__ == "__main__":
    pass
