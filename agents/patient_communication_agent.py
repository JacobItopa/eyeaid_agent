import json
from typing import Dict, Any

class PatientCommunicationAgent:
    """
    Patient Communication Agent
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
        screening_results: Dict[str, Any],
        triage_results: Dict[str, Any]
    ) -> str:
        """
        Generate patient-facing explanation
        """

        prompt = (
            "You are a Patient Communication Agent.\n"
            "Explain the results to the patient in simple, reassuring language.\n\n"
            "Output format:\n"
            "Patient Explanation:\n"
            "... (2-3 paragraphs)\n\n"
            f"Patient context: {json.dumps(patient_context)}\n"
            f"Screening findings: {json.dumps(screening_results)}\n"
            f"Triage recommendation: {json.dumps(triage_results)}"
        )
        
        try:
             print("Running local inference for Patient Communication Agent...")
             inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
             generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3
             )
             explanation = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        except Exception as e:
            print(f"Local inference failed: {e}")
            return (
                "Patient Explanation:\n"
                "We are currently experiencing technical difficulties with our automated analysis system. "
                "However, your images have been safely captured. "
                "Please consult with your healthcare provider for a manual review of your screening results."
            )

        if "Patient Explanation:" in explanation:
             parts = explanation.split("Patient Explanation:")
             if len(parts) > 1:
                 explanation = "Patient Explanation:" + parts[-1]

        return explanation.strip()

if __name__ == "__main__":
    pass
