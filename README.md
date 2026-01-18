# EyeAid – Agentic Ophthalmic Screening

EyeAid is an offline-capable, privacy-focused AI workflow designed for ophthalmic screening in resource-limited clinical settings. It leverages a multi-agent system powered by **MedGemma** to assist clinicians with patient intake, screening, triage, documentation, and patient communication.

## Features

-   **Patient Intake Agent**: Validates patient data and image quality.
-   **Ophthalmic Screening Agent**: Uses a locally loaded MedGemma (multimodal) model to analyze retinal fundus images.
-   **Risk & Triage Agent**: Assesses risk levels and suggests triage priority.
-   **Clinical Documentation Agent**: Generates structured clinical notes.
-   **Patient Communication Agent**: Creates patient-friendly explanations of the findings.

## Prerequisites

-   Python 3.8 or higher
-   NVIDIA GPU (Recommended for local model inference)

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd eyeaid
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Environment Setup:
    -   Create a `.env` file (optional if not using API tokens, but good practice).
    -   Ensure you have the necessary permissions to download Hugging Face models if prompted.

## Usage

### Run the Demo (CLI)

The primary way to run the application with the local MedGemma model is via the command-line interface:

```bash
python run_demo.py
```

This script will:
1.  Load the MedGemma model locally.
2.  Process a sample image (defined in `run_demo.py`).
3.  Execute the full multi-agent workflow.
4.  Output the results for each stage to the console.

**Note:** You can modify the `image_path` variable in `run_demo.py` to test with different images.

### Web Interface (Streamlit)

The project includes a Streamlit app (`app.py`). However, please note that the current configuration is optimized for the local `run_demo.py` workflow. The `app.py` may require updates to support the local model loading mechanism fully.

To explore the UI (may require code adjustments):
```bash
streamlit run app.py
```

## Docker Deployment

You can containerize the application for easy deployment.

1.  **Build the Docker image:**
    ```bash
    docker build -t eyeaid .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 eyeaid
    ```

    The Streamlit app will be available at `http://localhost:8501`.

## Project Structure

-   `agents/`: Contains the logic for each specific agent (Intake, Screening, Triage, etc.).
-   `models/`: Handles model loading (MedGemma).
-   `utils/`: Utility scripts.
-   `data/`: Directory for sample images and data.
-   `prompts/`: System prompts for the agents.

## License

[License Name]
