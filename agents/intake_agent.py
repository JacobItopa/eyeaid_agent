import cv2
from typing import Dict, Any, List


class IntakeAndImageQualityAgent:
    """
    Intake & Image Quality Control Agent

    - Validates patient metadata
    - Performs basic, deterministic image quality checks
    - Gates downstream clinical reasoning
    """

    REQUIRED_PATIENT_FIELDS = ["age"]

    def __init__(
        self,
        min_resolution: int = 512,
        blur_threshold: float = 100.0
    ):
        """
        Args:
            min_resolution: minimum acceptable width/height in pixels
            blur_threshold: variance of Laplacian threshold for blur detection
        """
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold

    def _validate_patient_info(
        self, patient_context: Dict[str, Any]
    ) -> List[str]:
        """
        Validate required patient metadata
        """
        missing_fields = []

        for field in self.REQUIRED_PATIENT_FIELDS:
            if field not in patient_context:
                missing_fields.append(field)

        return missing_fields

    def _check_image_quality(
        self, image_path: str
    ) -> Dict[str, Any]:
        """
        Perform basic image quality checks:
        - Resolution
        - Blur detection
        """
        image = cv2.imread(image_path)

        if image is None:
            return {
                "image_quality": "poor",
                "issues": ["image could not be loaded"]
            }

        height, width = image.shape[:2]
        issues = []

        if height < self.min_resolution or width < self.min_resolution:
            issues.append("low resolution")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < self.blur_threshold:
            issues.append("image appears blurry")

        if len(issues) == 0:
            quality = "adequate"
        elif len(issues) == 1:
            quality = "marginal"
        else:
            quality = "poor"

        return {
            "image_quality": quality,
            "issues": issues
        }

    def run(
        self,
        patient_context: Dict[str, Any],
        image_path: str
    ) -> Dict[str, Any]:
        """
        Execute intake validation and image QC
        """

        missing_fields = self._validate_patient_info(patient_context)
        image_quality_result = self._check_image_quality(image_path)

        limitations = []
        recommendation = "proceed"
        input_valid = True

        if missing_fields:
            input_valid = False
            limitations.append(
                f"missing patient fields: {', '.join(missing_fields)}"
            )
            recommendation = "collect more info"

        if image_quality_result["image_quality"] == "poor":
            input_valid = False
            limitations.extend(image_quality_result["issues"])
            recommendation = "retake image"

        elif image_quality_result["image_quality"] == "marginal":
            limitations.extend(image_quality_result["issues"])

        return {
            "input_valid": input_valid,
            "image_quality": image_quality_result["image_quality"],
            "limitations": limitations,
            "recommendation": recommendation
        }


if __name__ == "__main__":
    # Example local test
    agent = IntakeAndImageQualityAgent()

    patient_info = {
        "age": 60,
        "known_conditions": ["diabetes"]
    }

    result = agent.run(
        patient_context=patient_info,
        image_path="data/sample_images/fundus_sample.jpg"
    )

    print(result)
