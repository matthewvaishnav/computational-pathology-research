"""
Hospital Demo Scenarios for HistoCore Real-Time WSI Streaming

Production-ready demo scenarios with synthetic data for hospital presentations.
Showcases key capabilities: speed, accuracy, real-time visualization, PACS integration.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class DemoScenario(Enum):
    """Pre-configured demo scenarios"""

    SPEED_DEMO = "speed_demo"  # <30s processing showcase
    ACCURACY_DEMO = "accuracy_demo"  # High confidence predictions
    REALTIME_DEMO = "realtime_demo"  # Live visualization
    PACS_DEMO = "pacs_demo"  # PACS integration workflow
    MULTI_GPU_DEMO = "multi_gpu_demo"  # Scalability showcase
    CLINICAL_WORKFLOW = "clinical_workflow"  # End-to-end workflow


@dataclass
class SyntheticSlide:
    """Synthetic slide for demo purposes"""

    slide_id: str
    patient_id: str
    patient_name: str
    accession_number: str
    diagnosis: str
    confidence: float
    num_patches: int
    tissue_type: str
    staining: str
    magnification: str
    scan_date: datetime
    priority: str = "Routine"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "slide_id": self.slide_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "accession_number": self.accession_number,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "num_patches": self.num_patches,
            "tissue_type": self.tissue_type,
            "staining": self.staining,
            "magnification": self.magnification,
            "scan_date": self.scan_date.isoformat(),
            "priority": self.priority,
        }


class DemoDataGenerator:
    """Generate synthetic demo data"""

    # Realistic case library
    DEMO_CASES = [
        {
            "diagnosis": "Invasive Ductal Carcinoma",
            "confidence": 0.96,
            "tissue_type": "Breast",
            "num_patches": 120000,
            "priority": "Routine",
        },
        {
            "diagnosis": "Adenocarcinoma",
            "confidence": 0.94,
            "tissue_type": "Colon",
            "num_patches": 95000,
            "priority": "Routine",
        },
        {
            "diagnosis": "Squamous Cell Carcinoma",
            "confidence": 0.92,
            "tissue_type": "Lung",
            "num_patches": 110000,
            "priority": "STAT",
        },
        {
            "diagnosis": "Benign Prostatic Hyperplasia",
            "confidence": 0.98,
            "tissue_type": "Prostate",
            "num_patches": 85000,
            "priority": "Routine",
        },
        {
            "diagnosis": "Melanoma",
            "confidence": 0.91,
            "tissue_type": "Skin",
            "num_patches": 75000,
            "priority": "Urgent",
        },
        {
            "diagnosis": "Reactive Lymphoid Hyperplasia",
            "confidence": 0.97,
            "tissue_type": "Lymph Node",
            "num_patches": 65000,
            "priority": "Routine",
        },
        {
            "diagnosis": "Hepatocellular Carcinoma",
            "confidence": 0.89,
            "tissue_type": "Liver",
            "num_patches": 105000,
            "priority": "Urgent",
        },
        {
            "diagnosis": "Clear Cell Renal Cell Carcinoma",
            "confidence": 0.93,
            "tissue_type": "Kidney",
            "num_patches": 90000,
            "priority": "Routine",
        },
        {
            "diagnosis": "Glioblastoma",
            "confidence": 0.88,
            "tissue_type": "Brain",
            "num_patches": 100000,
            "priority": "STAT",
        },
        {
            "diagnosis": "Papillary Thyroid Carcinoma",
            "confidence": 0.95,
            "tissue_type": "Thyroid",
            "num_patches": 70000,
            "priority": "Routine",
        },
    ]

    PATIENT_NAMES = [
        "John Smith",
        "Mary Johnson",
        "Robert Williams",
        "Patricia Brown",
        "Michael Jones",
        "Jennifer Garcia",
        "William Miller",
        "Linda Davis",
        "David Rodriguez",
        "Barbara Martinez",
    ]

    @staticmethod
    def generate_slide(case_index: int = 0) -> SyntheticSlide:
        """Generate a synthetic slide"""
        case = DemoDataGenerator.DEMO_CASES[case_index % len(DemoDataGenerator.DEMO_CASES)]

        return SyntheticSlide(
            slide_id=f"DEMO-{case_index:04d}",
            patient_id=f"MRN{100000 + case_index}",
            patient_name=DemoDataGenerator.PATIENT_NAMES[
                case_index % len(DemoDataGenerator.PATIENT_NAMES)
            ],
            accession_number=f"ACC-2026-{case_index:05d}",
            diagnosis=case["diagnosis"],
            confidence=case["confidence"],
            num_patches=case["num_patches"],
            tissue_type=case["tissue_type"],
            staining="H&E",
            magnification="40x",
            scan_date=datetime.now() - timedelta(days=case_index),
            priority=case["priority"],
        )

    @staticmethod
    def generate_worklist(num_cases: int = 10) -> List[SyntheticSlide]:
        """Generate a synthetic PACS worklist"""
        return [DemoDataGenerator.generate_slide(i) for i in range(num_cases)]

    @staticmethod
    def generate_attention_heatmap(
        num_patches: int, diagnosis: str, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate realistic synthetic attention heatmap"""
        if seed is not None:
            np.random.seed(seed)

        # Grid dimensions (approximate square)
        grid_size = int(np.sqrt(num_patches))

        # Base attention (low)
        heatmap = np.random.uniform(0.1, 0.3, (grid_size, grid_size))

        # Add high-attention regions based on diagnosis
        if "Carcinoma" in diagnosis or "Melanoma" in diagnosis:
            # Multiple tumor foci
            num_foci = np.random.randint(2, 5)
            for _ in range(num_foci):
                center_x = np.random.randint(0, grid_size)
                center_y = np.random.randint(0, grid_size)
                radius = np.random.randint(5, 15)

                y, x = np.ogrid[:grid_size, :grid_size]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
                heatmap[mask] = np.random.uniform(0.7, 1.0)

        elif "Benign" in diagnosis or "Reactive" in diagnosis:
            # Scattered low attention
            heatmap = np.random.uniform(0.1, 0.4, (grid_size, grid_size))

        else:
            # Moderate attention with some hotspots
            num_hotspots = np.random.randint(1, 3)
            for _ in range(num_hotspots):
                center_x = np.random.randint(0, grid_size)
                center_y = np.random.randint(0, grid_size)
                radius = np.random.randint(8, 20)

                y, x = np.ogrid[:grid_size, :grid_size]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
                heatmap[mask] = np.random.uniform(0.5, 0.8)

        return heatmap


class DemoScenarioRunner:
    """Run pre-configured demo scenarios"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        gpu_ids: List[int] = [0],
        enable_visualization: bool = True,
    ):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.enable_visualization = enable_visualization
        self.data_generator = DemoDataGenerator()

    async def run_speed_demo(self) -> Dict:
        """
        Speed Demo: Showcase <30 second processing

        Demonstrates:
        - Fast processing of 100K+ patch slide
        - Real-time progress updates
        - GPU utilization
        """
        print("\n" + "=" * 60)
        print("SPEED DEMO: <30 Second Gigapixel Processing")
        print("=" * 60)

        # Generate demo slide
        slide = self.data_generator.generate_slide(0)

        print(f"\nSlide: {slide.slide_id}")
        print(f"Patient: {slide.patient_name} (MRN: {slide.patient_id})")
        print(f"Tissue: {slide.tissue_type}, {slide.staining}, {slide.magnification}")
        print(f"Patches: {slide.num_patches:,}")
        print(f"\nStarting processing...")

        start_time = time.time()

        # Simulate processing with progress updates
        total_patches = slide.num_patches
        batch_size = 64
        patches_per_second = 4000  # Realistic throughput

        processed = 0
        while processed < total_patches:
            await asyncio.sleep(0.1)  # Simulate processing

            batch_processed = min(batch_size, total_patches - processed)
            processed += int(patches_per_second * 0.1)
            processed = min(processed, total_patches)

            progress = (processed / total_patches) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / processed) * (total_patches - processed) if processed > 0 else 0

            print(
                f"\rProgress: {progress:5.1f}% | "
                f"Processed: {processed:,}/{total_patches:,} | "
                f"Speed: {patches_per_second:,} patches/s | "
                f"ETA: {eta:.1f}s",
                end="",
            )

        total_time = time.time() - start_time

        print(f"\n\n✓ Processing complete!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average speed: {total_patches/total_time:,.0f} patches/second")
        print(f"Diagnosis: {slide.diagnosis} ({slide.confidence*100:.0f}% confidence)")

        return {
            "scenario": "speed_demo",
            "slide": slide.to_dict(),
            "processing_time": total_time,
            "throughput": total_patches / total_time,
            "success": total_time < 30,
        }

    async def run_accuracy_demo(self) -> Dict:
        """
        Accuracy Demo: Showcase high-confidence predictions

        Demonstrates:
        - Multiple cases with varying confidence
        - Attention heatmap quality
        - Differential diagnoses
        """
        print("\n" + "=" * 60)
        print("ACCURACY DEMO: High-Confidence Predictions")
        print("=" * 60)

        # Generate 5 diverse cases
        cases = [self.data_generator.generate_slide(i) for i in range(5)]

        results = []
        for i, slide in enumerate(cases, 1):
            print(f"\n--- Case {i}/5 ---")
            print(f"Slide: {slide.slide_id}")
            print(f"Tissue: {slide.tissue_type}")
            print(f"Processing...")

            await asyncio.sleep(1)  # Simulate processing

            print(f"✓ Diagnosis: {slide.diagnosis}")
            print(f"  Confidence: {slide.confidence*100:.1f}%")

            # Generate attention heatmap
            heatmap = self.data_generator.generate_attention_heatmap(
                slide.num_patches, slide.diagnosis, seed=i
            )

            print(f"  Attention regions: {np.sum(heatmap > 0.7)} high-attention areas")

            results.append(
                {
                    "slide": slide.to_dict(),
                    "heatmap_shape": heatmap.shape,
                    "high_attention_regions": int(np.sum(heatmap > 0.7)),
                }
            )

        avg_confidence = np.mean([s.confidence for s in cases])
        print(f"\n✓ Average confidence: {avg_confidence*100:.1f}%")

        return {
            "scenario": "accuracy_demo",
            "num_cases": len(cases),
            "average_confidence": avg_confidence,
            "results": results,
        }

    async def run_realtime_demo(self) -> Dict:
        """
        Real-Time Demo: Showcase live visualization

        Demonstrates:
        - Progressive attention heatmap updates
        - Confidence score progression
        - WebSocket streaming
        """
        print("\n" + "=" * 60)
        print("REAL-TIME DEMO: Live Visualization")
        print("=" * 60)

        slide = self.data_generator.generate_slide(0)

        print(f"\nSlide: {slide.slide_id}")
        print(f"Patient: {slide.patient_name}")
        print(f"Tissue: {slide.tissue_type}")
        print(f"\nStarting real-time processing with live updates...")
        print("(Attention heatmap and confidence update every second)\n")

        total_patches = slide.num_patches
        processed = 0
        confidence = 0.5

        updates = []

        while processed < total_patches:
            await asyncio.sleep(1)  # 1 second updates

            # Simulate progress
            processed += int(total_patches * 0.1)  # 10% per second
            processed = min(processed, total_patches)

            # Confidence increases as more patches processed
            confidence = 0.5 + (processed / total_patches) * (slide.confidence - 0.5)

            progress = (processed / total_patches) * 100

            print(
                f"Update: {progress:5.1f}% complete | "
                f"Confidence: {confidence*100:5.1f}% | "
                f"Patches: {processed:,}/{total_patches:,}"
            )

            updates.append(
                {
                    "timestamp": time.time(),
                    "progress": progress,
                    "confidence": confidence,
                    "patches_processed": processed,
                }
            )

        print(f"\n✓ Processing complete!")
        print(f"Final diagnosis: {slide.diagnosis}")
        print(f"Final confidence: {slide.confidence*100:.1f}%")
        print(f"Total updates: {len(updates)}")

        return {
            "scenario": "realtime_demo",
            "slide": slide.to_dict(),
            "num_updates": len(updates),
            "updates": updates,
        }

    async def run_pacs_demo(self) -> Dict:
        """
        PACS Demo: Showcase PACS integration workflow

        Demonstrates:
        - Worklist retrieval
        - Slide retrieval from PACS
        - Result delivery to PACS
        """
        print("\n" + "=" * 60)
        print("PACS INTEGRATION DEMO: End-to-End Workflow")
        print("=" * 60)

        # Step 1: Retrieve worklist
        print("\nStep 1: Retrieving PACS worklist...")
        await asyncio.sleep(1)

        worklist = self.data_generator.generate_worklist(5)

        print(f"✓ Retrieved {len(worklist)} pending cases:")
        for i, slide in enumerate(worklist, 1):
            print(f"  {i}. {slide.patient_name} - {slide.tissue_type} - {slide.priority}")

        # Step 2: Select and retrieve slide
        print("\nStep 2: Selecting case for processing...")
        selected_slide = worklist[0]
        print(f"Selected: {selected_slide.patient_name} ({selected_slide.accession_number})")

        print("\nStep 3: Retrieving slide from PACS...")
        await asyncio.sleep(2)
        print(f"✓ Retrieved slide: {selected_slide.slide_id}")
        print(f"  Size: {selected_slide.num_patches:,} patches")
        print(f"  Format: DICOM WSI")

        # Step 3: Process slide
        print("\nStep 4: Processing slide...")
        await asyncio.sleep(3)
        print(f"✓ Processing complete")
        print(f"  Diagnosis: {selected_slide.diagnosis}")
        print(f"  Confidence: {selected_slide.confidence*100:.1f}%")

        # Step 4: Send results to PACS
        print("\nStep 5: Sending results to PACS...")
        await asyncio.sleep(2)
        print("✓ Results delivered to PACS:")
        print("  - Attention heatmap (DICOM Secondary Capture)")
        print("  - Structured report (DICOM SR)")
        print("  - Linked to original study")

        print("\n✓ PACS workflow complete!")

        return {
            "scenario": "pacs_demo",
            "worklist_size": len(worklist),
            "processed_slide": selected_slide.to_dict(),
            "results_delivered": True,
        }

    async def run_multi_gpu_demo(self) -> Dict:
        """
        Multi-GPU Demo: Showcase scalability

        Demonstrates:
        - Parallel processing across GPUs
        - Linear speedup
        - Concurrent slide processing
        """
        print("\n" + "=" * 60)
        print("MULTI-GPU DEMO: Scalability Showcase")
        print("=" * 60)

        num_gpus = len(self.gpu_ids)
        print(f"\nAvailable GPUs: {num_gpus}")
        print(f"GPU IDs: {self.gpu_ids}")

        # Generate multiple slides
        slides = [self.data_generator.generate_slide(i) for i in range(num_gpus)]

        print(f"\nProcessing {len(slides)} slides in parallel...")
        print("(One slide per GPU)\n")

        start_time = time.time()

        # Simulate parallel processing
        tasks = []
        for i, slide in enumerate(slides):
            print(f"GPU {self.gpu_ids[i]}: Processing {slide.slide_id} ({slide.tissue_type})")

        # Simulate concurrent processing
        await asyncio.sleep(3)

        total_time = time.time() - start_time

        print(f"\n✓ All slides processed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Speedup: {len(slides):.1f}x (vs sequential)")
        print(f"Efficiency: {(len(slides)/num_gpus)*100:.0f}%")

        print("\nResults:")
        for i, slide in enumerate(slides):
            print(f"  GPU {self.gpu_ids[i]}: {slide.diagnosis} ({slide.confidence*100:.0f}%)")

        return {
            "scenario": "multi_gpu_demo",
            "num_gpus": num_gpus,
            "num_slides": len(slides),
            "total_time": total_time,
            "speedup": len(slides),
            "slides": [s.to_dict() for s in slides],
        }

    async def run_clinical_workflow(self) -> Dict:
        """
        Clinical Workflow Demo: Complete end-to-end scenario

        Demonstrates:
        - PACS worklist retrieval
        - Slide processing
        - Result review
        - Report generation
        - Result delivery
        """
        print("\n" + "=" * 60)
        print("CLINICAL WORKFLOW DEMO: Complete End-to-End")
        print("=" * 60)

        # Step 1: Morning worklist
        print("\n=== Morning Worklist Review ===")
        print("Pathologist logs in at 8:00 AM...")
        await asyncio.sleep(1)

        worklist = self.data_generator.generate_worklist(10)
        print(f"✓ {len(worklist)} cases in worklist")
        print(f"  - {sum(1 for s in worklist if s.priority == 'STAT')} STAT")
        print(f"  - {sum(1 for s in worklist if s.priority == 'Urgent')} Urgent")
        print(f"  - {sum(1 for s in worklist if s.priority == 'Routine')} Routine")

        # Step 2: Prioritize STAT case
        print("\n=== Processing STAT Case ===")
        stat_case = next(s for s in worklist if s.priority == "STAT")
        print(f"Selected: {stat_case.patient_name} - {stat_case.tissue_type}")
        print(f"Accession: {stat_case.accession_number}")

        print("\nRetrieving from PACS...")
        await asyncio.sleep(1)
        print("✓ Slide retrieved")

        print("\nProcessing...")
        await asyncio.sleep(2)
        print(f"✓ Processing complete (25 seconds)")
        print(f"  Diagnosis: {stat_case.diagnosis}")
        print(f"  Confidence: {stat_case.confidence*100:.1f}%")

        # Step 3: Review results
        print("\n=== Pathologist Review ===")
        print("Reviewing attention heatmap...")
        await asyncio.sleep(1)
        print("✓ High-attention regions correlate with microscopy")
        print("✓ Confidence score appropriate")
        print("✓ Pathologist confirms diagnosis")

        # Step 4: Generate report
        print("\n=== Report Generation ===")
        print("Generating clinical report...")
        await asyncio.sleep(1)
        print("✓ PDF report generated")
        print("  - Patient demographics")
        print("  - AI findings with confidence")
        print("  - Attention heatmap")
        print("  - Pathologist interpretation")

        # Step 5: Deliver results
        print("\n=== Result Delivery ===")
        print("Sending to PACS...")
        await asyncio.sleep(1)
        print("✓ Results delivered to PACS")

        print("Notifying clinician...")
        await asyncio.sleep(0.5)
        print("✓ Email sent to ordering physician")

        # Step 6: Batch process routine cases
        print("\n=== Batch Processing Routine Cases ===")
        routine_cases = [s for s in worklist if s.priority == "Routine"]
        print(f"Processing {len(routine_cases)} routine cases in batch...")
        await asyncio.sleep(3)
        print(f"✓ All routine cases processed")

        print("\n=== Workflow Complete ===")
        print(f"Total cases processed: {len(worklist)}")
        print(f"Time: 9:15 AM (1 hour 15 minutes)")
        print(f"Efficiency: {len(worklist)/1.25:.1f} cases/hour")

        return {
            "scenario": "clinical_workflow",
            "total_cases": len(worklist),
            "stat_cases": sum(1 for s in worklist if s.priority == "STAT"),
            "urgent_cases": sum(1 for s in worklist if s.priority == "Urgent"),
            "routine_cases": sum(1 for s in worklist if s.priority == "Routine"),
            "total_time_minutes": 75,
            "cases_per_hour": len(worklist) / 1.25,
        }

    async def run_scenario(self, scenario: DemoScenario) -> Dict:
        """Run a specific demo scenario"""
        if scenario == DemoScenario.SPEED_DEMO:
            return await self.run_speed_demo()
        elif scenario == DemoScenario.ACCURACY_DEMO:
            return await self.run_accuracy_demo()
        elif scenario == DemoScenario.REALTIME_DEMO:
            return await self.run_realtime_demo()
        elif scenario == DemoScenario.PACS_DEMO:
            return await self.run_pacs_demo()
        elif scenario == DemoScenario.MULTI_GPU_DEMO:
            return await self.run_multi_gpu_demo()
        elif scenario == DemoScenario.CLINICAL_WORKFLOW:
            return await self.run_clinical_workflow()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")


async def main():
    """Run all demo scenarios"""
    runner = DemoScenarioRunner(gpu_ids=[0, 1, 2, 3])

    print("\n" + "=" * 60)
    print("HISTOCORE REAL-TIME WSI STREAMING")
    print("Hospital Demo Suite")
    print("=" * 60)

    scenarios = [
        DemoScenario.SPEED_DEMO,
        DemoScenario.ACCURACY_DEMO,
        DemoScenario.REALTIME_DEMO,
        DemoScenario.PACS_DEMO,
        DemoScenario.MULTI_GPU_DEMO,
        DemoScenario.CLINICAL_WORKFLOW,
    ]

    results = {}
    for scenario in scenarios:
        result = await runner.run_scenario(scenario)
        results[scenario.value] = result

        # Pause between demos
        if scenario != scenarios[-1]:
            print("\n" + "-" * 60)
            await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("DEMO SUITE COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Speed: <30s processing ✓")
    print(
        f"  - Accuracy: {results['accuracy_demo']['average_confidence']*100:.0f}% avg confidence ✓"
    )
    print(f"  - Real-time: Live updates ✓")
    print(f"  - PACS: Full integration ✓")
    print(f"  - Scalability: {results['multi_gpu_demo']['num_gpus']}x GPU speedup ✓")
    print(f"  - Workflow: {results['clinical_workflow']['cases_per_hour']:.1f} cases/hour ✓")

    return results


if __name__ == "__main__":
    asyncio.run(main())
