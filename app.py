import os
import json
from dotenv import load_dotenv

load_dotenv()
import tempfile
import streamlit as st

from agents.intake_agent import IntakeAndImageQualityAgent
from agents.screening_agent import OphthalmicScreeningAgent
from agents.triage_agent import RiskAndTriageAgent
from agents.documentation_agent import ClinicalDocumentationAgent
from agents.patient_communication_agent import PatientCommunicationAgent


st.set_page_config(
    page_title="EyeAid – Agentic Ophthalmic Screening",
    layout="wide"
)

st.title("🩺 EyeAid: Agentic Ophthalmic Screening with MedGemma")
st.caption(
    "An offline-capable, privacy-focused AI workflow for eye screening "
    "in resource-limited clinical settings."
)

# ---------------- Sidebar: Patient Intake ----------------
st.sidebar.header("🧾 Patient Intake")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)

conditions = st.sidebar.multiselect(
    "Known conditions",
    ["Diabetes", "Hypertension", "None"]
)

symptoms = st.sidebar.text_area(
    "Symptoms (optional)",
    placeholder="e.g. blurred vision, eye pain"
)

uploaded_image = st.sidebar.file_uploader(
    "Upload retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

run_button = st.sidebar.button("▶️ Run Screening Workflow")

# ---------------- Main Logic ----------------
if run_button:
    if uploaded_image is None:
        st.error("Please upload a retinal image to proceed.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_image.read())
        image_path = tmp.name

    patient_context = {
        "age": age,
        "known_conditions": conditions,
        "symptoms": symptoms.split(",") if symptoms else []
    }

    hf_api_token = os.getenv("HF_API_TOKEN")
    if not hf_api_token:
        st.error("HF_API_TOKEN environment variable not set.")
        st.stop()

    st.subheader("🔁 Agentic Workflow Execution")

    # -------- Agent 1: Intake --------
    with st.expander("1️⃣ Intake & Image Quality Agent", expanded=True):
        intake_agent = IntakeAndImageQualityAgent()
        intake_results = intake_agent.run(patient_context, image_path)
        st.json(intake_results)

        if not intake_results["input_valid"]:
            st.error("Workflow stopped due to intake issues.")
            st.stop()

    # -------- Agent 2: Screening --------
    with st.expander("2️⃣ Screening Agent (MedGemma – Multimodal)", expanded=True):
        screening_agent = OphthalmicScreeningAgent(hf_api_token)
        screening_results = screening_agent.run(patient_context, image_path)
        st.json(screening_results)

    # -------- Agent 3: Triage --------
    with st.expander("3️⃣ Risk & Triage Agent", expanded=True):
        triage_agent = RiskAndTriageAgent(hf_api_token)
        triage_results = triage_agent.run(patient_context, screening_results)
        st.json(triage_results)

    # -------- Agent 4 & 5: Outputs --------
    clinician_tab, patient_tab = st.tabs(
        ["🧑‍⚕️ Clinician View", "👤 Patient View"]
    )

    with clinician_tab:
        documentation_agent = ClinicalDocumentationAgent(hf_api_token)
        clinical_note = documentation_agent.run(
            patient_context,
            intake_results,
            screening_results,
            triage_results
        )
        st.subheader("📄 Clinical Screening Note")
        st.text_area("", clinical_note, height=300)

    with patient_tab:
        patient_agent = PatientCommunicationAgent(hf_api_token)
        patient_message = patient_agent.run(
            patient_context,
            screening_results,
            triage_results
        )
        st.subheader("💬 Patient Explanation")
        st.text_area("", patient_message, height=200)

    st.success("✅ Screening workflow completed successfully.")
