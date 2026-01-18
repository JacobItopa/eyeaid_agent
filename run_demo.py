import json
import os
from dotenv import load_dotenv

load_dotenv()
from pprint import pprint

from agents.intake_agent import IntakeAndImageQualityAgent
from agents.screening_agent import OphthalmicScreeningAgent
from agents.triage_agent import RiskAndTriageAgent
from agents.documentation_agent import ClinicalDocumentationAgent
from agents.patient_communication_agent import PatientCommunicationAgent
from models.medgemma_loader import MedGemmaLoader

def run_demo(
    patient_context: dict,
    image_path: str,
    model: object,
    processor: object
) -> dict:
    """
    Run full agentic ophthalmic screening workflow
    """

    print("\n=== STEP 1: Intake & Image Quality Check ===")
    intake_agent = IntakeAndImageQualityAgent()
    intake_results = intake_agent.run(
        patient_context=patient_context,
        image_path=image_path
    )
    pprint(intake_results)

    if not intake_results["input_valid"]:
        print("\n[STOP] Workflow stopped at intake stage.")
        return {
            "status": "stopped",
            "stage": "intake",
            "results": intake_results
        }

    print("\n=== STEP 2: Ophthalmic Screening (MedGemma Multimodal) ===")
    screening_agent = OphthalmicScreeningAgent(
        model=model,
        processor=processor
    )
    screening_results = screening_agent.run(
        patient_context=patient_context,
        image_path=image_path
    )
    pprint(screening_results)

    print("\n=== STEP 3: Risk Stratification & Triage ===")
    triage_agent = RiskAndTriageAgent(
         model=model,
        processor=processor
    )
    triage_results = triage_agent.run(
        patient_context=patient_context,
        screening_results=screening_results
    )
    pprint(triage_results)

    print("\n=== STEP 4: Clinical Documentation ===")
    documentation_agent = ClinicalDocumentationAgent(
         model=model,
        processor=processor
    )
    clinical_note = documentation_agent.run(
        patient_context=patient_context,
        intake_results=intake_results,
        screening_results=screening_results,
        triage_results=triage_results
    )
    print("\n--- Clinical Note ---")
    print(clinical_note)

    print("\n=== STEP 5: Patient Communication ===")
    patient_agent = PatientCommunicationAgent(
         model=model,
        processor=processor
    )
    patient_message = patient_agent.run(
        patient_context=patient_context,
        screening_results=screening_results,
        triage_results=triage_results
    )
    print("\n--- Patient Explanation ---")
    print(patient_message)

    return {
        "status": "completed",
        "intake": intake_results,
        "screening": screening_results,
        "triage": triage_results,
        "clinical_documentation": clinical_note,
        "patient_communication": patient_message
    }


if __name__ == "__main__":
    """
    Example demo run
    """
    print("Initializing Local MedGemma Model...")
    loader = MedGemmaLoader()
    model, processor = loader.load_model()
    print("Model loaded.")

    patient_info = {
        "age": 59,
        "known_conditions": ["diabetes"],
        "symptoms": ["blurred vision"],
        "language_preference": "English"
    }

    image_path = "data/sample_images/1ffa9331-8d87-11e8-9daf-6045cb817f5b..jpg"
    
    # Ensure sample image exists or warn user
    if not os.path.exists(image_path):
        print(f"Warning: Sample image not found at {image_path}. Please provide a valid image path.")
        # Create a dummy file for testing if it doesn't exist? 
        # Better to let it fail or assume the user has data.
        # But let's check if the directory exists atleast.
    
    if os.path.exists(image_path):
        output = run_demo(
            patient_context=patient_info,
            image_path=image_path,
            model=model,
            processor=processor
        )

        print("\n=== FINAL OUTPUT (JSON) ===")
        print(json.dumps(output, indent=2))
    else:
        print("Please ensure data/sample_images/... exists.")
