#!/usr/bin/env python3
"""
Test Data Fixtures and Mocks

Provides comprehensive test data fixtures, mock objects, and synthetic data generation
for integration testing of the Medical AI platform.
"""

import io
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydicom import Dataset, dcmwrite
from pydicom.uid import generate_uid


class TestDataFixtures:
    """Comprehensive test data fixtures for integration testing."""

    def __init__(self, output_dir: str = "tests/integration/fixtures"):
        """Initialize test data fixtures."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test data configuration
        self.config = {
            "image_sizes": [(224, 224), (512, 512), (1024, 1024)],
            "dicom_modalities": ["SM", "CR", "MG", "US"],
            "cancer_types": ["breast", "lung", "prostate", "colon", "melanoma"],
            "stain_types": ["H&E", "IHC", "Trichrome", "PAS"],
            "magnifications": ["4x", "10x", "20x", "40x"],
        }

        print(f"🧪 Test Data Fixtures initialized")
        print(f"📁 Output directory: {self.output_dir}")

    def generate_synthetic_pathology_image(
        self,
        width: int = 224,
        height: int = 224,
        cancer_type: str = "breast",
        stain_type: str = "H&E",
        has_cancer: bool = True,
    ) -> np.ndarray:
        """Generate synthetic pathology image with realistic features."""

        # Base tissue color based on stain type
        stain_colors = {
            "H&E": {
                "nuclei": [100, 50, 150],
                "cytoplasm": [200, 150, 200],
                "background": [240, 220, 240],
            },
            "IHC": {
                "nuclei": [139, 69, 19],
                "cytoplasm": [100, 150, 200],
                "background": [230, 230, 230],
            },
            "Trichrome": {
                "nuclei": [0, 100, 200],
                "cytoplasm": [200, 0, 0],
                "background": [220, 220, 220],
            },
            "PAS": {
                "nuclei": [200, 0, 200],
                "cytoplasm": [150, 150, 150],
                "background": [240, 240, 240],
            },
        }

        colors = stain_colors.get(stain_type, stain_colors["H&E"])

        # Initialize image with background
        image = np.full((height, width, 3), colors["background"], dtype=np.uint8)

        # Add noise
        noise = np.random.normal(0, 10, (height, width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Generate tissue structures
        num_cells = random.randint(50, 200)

        for _ in range(num_cells):
            # Random cell position
            center_x = random.randint(10, width - 10)
            center_y = random.randint(10, height - 10)

            # Cell size varies based on cancer presence
            if has_cancer:
                # Cancer cells are often larger and more irregular
                cell_radius = random.randint(3, 8)
                irregularity = random.uniform(0.3, 0.8)
            else:
                # Normal cells are smaller and more regular
                cell_radius = random.randint(2, 5)
                irregularity = random.uniform(0.1, 0.3)

            # Draw cell nucleus
            for y in range(max(0, center_y - cell_radius), min(height, center_y + cell_radius)):
                for x in range(max(0, center_x - cell_radius), min(width, center_x + cell_radius)):
                    # Distance from center with irregularity
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    noise_factor = 1 + irregularity * random.uniform(-0.5, 0.5)

                    if dist < cell_radius * noise_factor:
                        # Nucleus color
                        image[y, x] = colors["nuclei"]

            # Draw cytoplasm around nucleus
            cytoplasm_radius = cell_radius + random.randint(1, 3)
            for y in range(
                max(0, center_y - cytoplasm_radius), min(height, center_y + cytoplasm_radius)
            ):
                for x in range(
                    max(0, center_x - cytoplasm_radius), min(width, center_x + cytoplasm_radius)
                ):
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                    if cell_radius < dist < cytoplasm_radius:
                        # Cytoplasm color (lighter)
                        cytoplasm_color = [
                            min(255, c + 30) for c in colors.get("cytoplasm", colors["background"])
                        ]
                        image[y, x] = cytoplasm_color

        # Add cancer-specific features
        if has_cancer:
            # Add mitotic figures (dark spots)
            num_mitoses = random.randint(2, 8)
            for _ in range(num_mitoses):
                x = random.randint(2, width - 2)
                y = random.randint(2, height - 2)
                image[y - 1 : y + 2, x - 1 : x + 2] = [50, 20, 80]  # Dark purple

            # Add necrosis areas (pale regions)
            if random.random() < 0.3:  # 30% chance of necrosis
                necrosis_x = random.randint(20, width - 40)
                necrosis_y = random.randint(20, height - 40)
                necrosis_size = random.randint(15, 30)

                for y in range(necrosis_y, min(height, necrosis_y + necrosis_size)):
                    for x in range(necrosis_x, min(width, necrosis_x + necrosis_size)):
                        image[y, x] = [250, 240, 250]  # Very pale

        return image

    def create_test_image_file(
        self, filename: str, width: int = 224, height: int = 224, format: str = "PNG", **kwargs
    ) -> Path:
        """Create and save a test image file."""

        image_array = self.generate_synthetic_pathology_image(width, height, **kwargs)
        image = Image.fromarray(image_array)

        filepath = self.output_dir / filename
        image.save(filepath, format=format)

        return filepath

    def create_test_dicom_file(
        self, filename: str, width: int = 512, height: int = 512, modality: str = "SM", **kwargs
    ) -> Path:
        """Create and save a test DICOM file."""

        # Generate image data
        image_array = self.generate_synthetic_pathology_image(width, height, **kwargs)

        # Create DICOM dataset
        ds = Dataset()

        # Patient information
        ds.PatientName = f"TEST^PATIENT^{random.randint(1000, 9999)}"
        ds.PatientID = f"TEST{random.randint(100000, 999999)}"
        ds.PatientBirthDate = (
            datetime.now() - timedelta(days=random.randint(18 * 365, 80 * 365))
        ).strftime("%Y%m%d")
        ds.PatientSex = random.choice(["M", "F"])

        # Study information
        ds.StudyInstanceUID = generate_uid()
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S")
        ds.StudyDescription = f"Pathology Study - {kwargs.get('cancer_type', 'breast').title()}"
        ds.AccessionNumber = f"ACC{random.randint(100000, 999999)}"

        # Series information
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 1
        ds.SeriesDescription = f"{kwargs.get('stain_type', 'H&E')} Stain"
        ds.Modality = modality

        # Instance information
        ds.SOPInstanceUID = generate_uid()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"  # VL Whole Slide Microscopy
        ds.InstanceNumber = 1

        # Image information
        ds.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.Rows = height
        ds.Columns = width
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0

        # Microscopy-specific tags
        if modality == "SM":
            ds.ImagedVolumeWidth = width * 0.25  # 0.25 microns per pixel
            ds.ImagedVolumeHeight = height * 0.25
            ds.ImagedVolumeDepth = 5.0  # 5 micron section thickness
            ds.SpecimenLabelInImage = "YES"
            ds.FocusMethod = "AUTO"
            ds.ExtendedDepthOfField = "NO"

        # Convert image to bytes
        image_bytes = image_array.tobytes()
        ds.PixelData = image_bytes

        # Save DICOM file
        filepath = self.output_dir / filename
        dcmwrite(filepath, ds, write_like_original=False)

        return filepath

    def create_test_case_data(self, num_cases: int = 10) -> List[Dict]:
        """Create test case data for API testing."""

        cases = []

        for i in range(num_cases):
            case_id = str(uuid.uuid4())
            patient_id = f"PATIENT_{i+1:04d}"

            case_data = {
                "case_id": case_id,
                "patient_id": patient_id,
                "patient_name": f"Test Patient {i+1}",
                "study_id": f"STUDY_{i+1:04d}",
                "accession_number": f"ACC{random.randint(100000, 999999)}",
                "case_type": random.choice(self.config["cancer_types"]),
                "priority": random.choice(["low", "normal", "high", "urgent"]),
                "status": random.choice(["pending", "in_progress", "completed", "reviewed"]),
                "created_date": (
                    datetime.now() - timedelta(days=random.randint(0, 30))
                ).isoformat(),
                "due_date": (datetime.now() + timedelta(days=random.randint(1, 7))).isoformat(),
                "assigned_pathologist": f"Dr. Pathologist {random.randint(1, 5)}",
                "stain_type": random.choice(self.config["stain_types"]),
                "magnification": random.choice(self.config["magnifications"]),
                "specimen_site": random.choice(["breast", "lung", "prostate", "colon", "skin"]),
                "clinical_history": f"Clinical history for patient {i+1}",
                "diagnosis": (
                    random.choice(["benign", "malignant", "atypical", "insufficient"])
                    if random.random() > 0.3
                    else None
                ),
                "confidence_score": random.uniform(0.7, 0.99) if random.random() > 0.3 else None,
                "ai_prediction": (
                    random.choice(["positive", "negative", "uncertain"])
                    if random.random() > 0.2
                    else None
                ),
                "processing_time_ms": random.randint(15000, 45000),
                "image_count": random.randint(1, 5),
                "annotations": [],
            }

            # Add some annotations
            num_annotations = random.randint(0, 3)
            for j in range(num_annotations):
                annotation = {
                    "annotation_id": str(uuid.uuid4()),
                    "type": random.choice(["polygon", "rectangle", "point"]),
                    "label": random.choice(["tumor", "normal", "necrosis", "mitosis"]),
                    "coordinates": self._generate_annotation_coordinates(),
                    "confidence": random.uniform(0.6, 0.95),
                    "created_by": "ai_model",
                    "created_date": datetime.now().isoformat(),
                }
                case_data["annotations"].append(annotation)

            cases.append(case_data)

        return cases

    def _generate_annotation_coordinates(self) -> List[List[int]]:
        """Generate random annotation coordinates."""
        annotation_type = random.choice(["polygon", "rectangle"])

        if annotation_type == "rectangle":
            x1 = random.randint(10, 200)
            y1 = random.randint(10, 200)
            x2 = x1 + random.randint(20, 50)
            y2 = y1 + random.randint(20, 50)
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        else:  # polygon
            center_x = random.randint(50, 150)
            center_y = random.randint(50, 150)
            radius = random.randint(15, 30)
            num_points = random.randint(6, 12)

            points = []
            for i in range(num_points):
                angle = (2 * np.pi * i) / num_points
                x = center_x + radius * np.cos(angle) + random.randint(-5, 5)
                y = center_y + radius * np.sin(angle) + random.randint(-5, 5)
                points.append([int(x), int(y)])

            return points

    def create_test_user_data(self, num_users: int = 5) -> List[Dict]:
        """Create test user data for authentication testing."""

        roles = ["pathologist", "resident", "admin", "technician"]
        departments = ["Pathology", "Oncology", "Surgery", "Radiology"]

        users = []

        for i in range(num_users):
            user_data = {
                "user_id": str(uuid.uuid4()),
                "username": f"testuser{i+1}",
                "email": f"testuser{i+1}@example.com",
                "password": f"testpass{i+1}",
                "first_name": f"Test",
                "last_name": f"User{i+1}",
                "role": random.choice(roles),
                "department": random.choice(departments),
                "license_number": f"LIC{random.randint(100000, 999999)}",
                "phone": f"+1-555-{random.randint(1000, 9999)}",
                "created_date": (
                    datetime.now() - timedelta(days=random.randint(30, 365))
                ).isoformat(),
                "last_login": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                "is_active": random.choice([True, True, True, False]),  # 75% active
                "preferences": {
                    "theme": random.choice(["light", "dark"]),
                    "notifications": random.choice([True, False]),
                    "default_magnification": random.choice(self.config["magnifications"]),
                    "auto_save": True,
                },
            }

            users.append(user_data)

        return users

    def create_performance_test_data(self) -> Dict:
        """Create performance test configuration and data."""

        return {
            "load_test_config": {
                "concurrent_users": [1, 5, 10, 20, 50],
                "test_duration_seconds": 60,
                "ramp_up_time_seconds": 10,
                "image_sizes_kb": [50, 100, 500, 1000, 2000],
            },
            "performance_thresholds": {
                "api_response_time_ms": 2000,
                "inference_time_ms": 30000,
                "database_query_time_ms": 1000,
                "memory_usage_mb": 2048,
                "cpu_usage_percent": 80.0,
                "throughput_requests_per_second": 5.0,
            },
            "test_scenarios": [
                {
                    "name": "light_load",
                    "concurrent_users": 5,
                    "requests_per_user": 10,
                    "image_size_kb": 100,
                },
                {
                    "name": "medium_load",
                    "concurrent_users": 15,
                    "requests_per_user": 20,
                    "image_size_kb": 500,
                },
                {
                    "name": "heavy_load",
                    "concurrent_users": 30,
                    "requests_per_user": 50,
                    "image_size_kb": 1000,
                },
            ],
        }

    def save_fixtures_to_files(self) -> Dict[str, Path]:
        """Save all test fixtures to JSON files."""

        fixtures = {
            "cases": self.create_test_case_data(50),
            "users": self.create_test_user_data(20),
            "performance": self.create_performance_test_data(),
        }

        saved_files = {}

        for fixture_name, fixture_data in fixtures.items():
            filepath = self.output_dir / f"{fixture_name}_fixtures.json"

            with open(filepath, "w") as f:
                json.dump(fixture_data, f, indent=2, default=str)

            saved_files[fixture_name] = filepath
            print(f"💾 Saved {fixture_name} fixtures to: {filepath}")

        return saved_files

    def create_sample_images_and_dicoms(self, num_samples: int = 10) -> Dict[str, List[Path]]:
        """Create sample images and DICOM files for testing."""

        created_files = {"images": [], "dicoms": []}

        # Create sample images
        for i in range(num_samples):
            cancer_type = random.choice(self.config["cancer_types"])
            stain_type = random.choice(self.config["stain_types"])
            has_cancer = random.choice([True, False])
            size = random.choice(self.config["image_sizes"])

            filename = f"sample_{i+1:02d}_{cancer_type}_{stain_type}_{'positive' if has_cancer else 'negative'}.png"

            filepath = self.create_test_image_file(
                filename=filename,
                width=size[0],
                height=size[1],
                cancer_type=cancer_type,
                stain_type=stain_type,
                has_cancer=has_cancer,
            )

            created_files["images"].append(filepath)
            print(f"🖼️ Created image: {filepath.name}")

        # Create sample DICOM files
        for i in range(num_samples // 2):  # Fewer DICOM files
            cancer_type = random.choice(self.config["cancer_types"])
            stain_type = random.choice(self.config["stain_types"])
            has_cancer = random.choice([True, False])
            modality = random.choice(self.config["dicom_modalities"])

            filename = f"sample_{i+1:02d}_{cancer_type}_{stain_type}_{'positive' if has_cancer else 'negative'}.dcm"

            filepath = self.create_test_dicom_file(
                filename=filename,
                width=512,
                height=512,
                modality=modality,
                cancer_type=cancer_type,
                stain_type=stain_type,
                has_cancer=has_cancer,
            )

            created_files["dicoms"].append(filepath)
            print(f"🏥 Created DICOM: {filepath.name}")

        return created_files

    def cleanup_fixtures(self):
        """Clean up all generated test fixtures."""

        if self.output_dir.exists():
            import shutil

            shutil.rmtree(self.output_dir)
            print(f"🧹 Cleaned up fixtures directory: {self.output_dir}")


def main():
    """Generate all test fixtures."""

    print("🧪 Generating Test Data Fixtures")
    print("=" * 50)

    fixtures = TestDataFixtures()

    # Generate JSON fixtures
    saved_files = fixtures.save_fixtures_to_files()

    # Generate sample images and DICOM files
    created_files = fixtures.create_sample_images_and_dicoms(20)

    print("\n✅ Test Data Generation Complete")
    print(f"📁 Output directory: {fixtures.output_dir}")
    print(f"📄 JSON fixtures: {len(saved_files)} files")
    print(f"🖼️ Sample images: {len(created_files['images'])} files")
    print(f"🏥 Sample DICOMs: {len(created_files['dicoms'])} files")


if __name__ == "__main__":
    main()
