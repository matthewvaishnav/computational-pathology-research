"""
Case-Based Reasoning System
Comprehensive case database with FAISS indexing, efficient retrieval, and metadata management
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CaseMetadata:
    """Comprehensive metadata for training cases"""

    case_id: str
    slide_id: str
    patient_id: Optional[str]  # Anonymized
    institution: str
    scanner_type: str
    magnification: float
    stain_type: str
    tissue_type: str
    diagnosis: str
    grade: Optional[str]
    stage: Optional[str]
    molecular_markers: Dict[str, Any]
    pathologist_id: str
    confidence_score: float
    annotation_time: datetime
    image_quality_score: float
    artifact_flags: List[str]
    demographics: Dict[str, Any]  # Age group, sex (anonymized)
    treatment_response: Optional[str]
    follow_up_months: Optional[int]
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result["annotation_time"] = self.annotation_time.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseMetadata":
        """Create from dictionary"""
        data["annotation_time"] = datetime.fromisoformat(data["annotation_time"])
        return cls(**data)


@dataclass
class SimilarCase:
    """Similar case with features and metadata"""

    case_id: str
    similarity_score: float
    diagnosis: str
    confidence: float
    features: torch.Tensor
    metadata: CaseMetadata
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensor)"""
        return {
            "case_id": self.case_id,
            "similarity_score": self.similarity_score,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict(),
            "explanation": self.explanation,
        }


@dataclass
class RetrievalQuery:
    """Query for case retrieval"""

    features: torch.Tensor
    disease_filter: Optional[str] = None
    institution_filter: Optional[str] = None
    confidence_threshold: float = 0.0
    quality_threshold: float = 0.0
    exclude_case_ids: Optional[List[str]] = None
    require_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    k: int = 5
    diversity_weight: float = 0.1


@dataclass
class CaseStatistics:
    """Statistics about the case database"""

    total_cases: int
    cases_by_disease: Dict[str, int]
    cases_by_institution: Dict[str, int]
    cases_by_scanner: Dict[str, int]
    average_quality_score: float
    date_range: Tuple[datetime, datetime]
    feature_dimensionality: int
    index_size_mb: float


class CaseDatabase:
    """Comprehensive case database with FAISS indexing and metadata management"""

    def __init__(
        self,
        database_path: str,
        feature_dim: int = 2048,
        index_type: str = "IVF",
        nlist: int = 100,
        use_gpu: bool = True,
    ):
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)

        self.feature_dim = feature_dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Initialize components
        self.index = None
        self.cases = []
        self.case_id_to_idx = {}
        self.metadata_db_path = self.database_path / "metadata.db"
        self.index_path = self.database_path / "faiss_index.bin"
        self.features_path = self.database_path / "features.npy"

        # Thread safety
        self.lock = threading.RLock()

        self.logger = logging.getLogger(__name__)

        # Initialize database
        self._init_metadata_db()
        self._load_existing_data()

    def _init_metadata_db(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                slide_id TEXT,
                patient_id TEXT,
                institution TEXT,
                scanner_type TEXT,
                magnification REAL,
                stain_type TEXT,
                tissue_type TEXT,
                diagnosis TEXT,
                grade TEXT,
                stage TEXT,
                molecular_markers TEXT,
                pathologist_id TEXT,
                confidence_score REAL,
                annotation_time TEXT,
                image_quality_score REAL,
                artifact_flags TEXT,
                demographics TEXT,
                treatment_response TEXT,
                follow_up_months INTEGER,
                tags TEXT,
                feature_idx INTEGER
            )
        """)

        # Create indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis ON cases(diagnosis)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_institution ON cases(institution)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON cases(image_quality_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON cases(confidence_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotation_time ON cases(annotation_time)")

        conn.commit()
        conn.close()

    def _load_existing_data(self):
        """Load existing cases and index from disk"""
        try:
            # Load metadata
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases ORDER BY feature_idx")
            rows = cursor.fetchall()
            conn.close()

            if rows:
                # Reconstruct cases list
                self.cases = []
                self.case_id_to_idx = {}

                for row in rows:
                    metadata = CaseMetadata(
                        case_id=row[0],
                        slide_id=row[1],
                        patient_id=row[2],
                        institution=row[3],
                        scanner_type=row[4],
                        magnification=row[5],
                        stain_type=row[6],
                        tissue_type=row[7],
                        diagnosis=row[8],
                        grade=row[9],
                        stage=row[10],
                        molecular_markers=json.loads(row[11]) if row[11] else {},
                        pathologist_id=row[12],
                        confidence_score=row[13],
                        annotation_time=datetime.fromisoformat(row[14]),
                        image_quality_score=row[15],
                        artifact_flags=json.loads(row[16]) if row[16] else [],
                        demographics=json.loads(row[17]) if row[17] else {},
                        treatment_response=row[18],
                        follow_up_months=row[19],
                        tags=json.loads(row[20]) if row[20] else [],
                    )

                    self.cases.append(metadata)
                    self.case_id_to_idx[metadata.case_id] = len(self.cases) - 1

                # Load FAISS index
                if self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                    if self.use_gpu:
                        gpu_res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

                    self.logger.info(f"Loaded {len(self.cases)} cases from existing database")
                else:
                    self.logger.warning("Metadata found but no FAISS index - will rebuild")
                    self._rebuild_index()

        except Exception as e:
            self.logger.warning(f"Could not load existing data: {e}")
            self.cases = []
            self.case_id_to_idx = {}

    def add_case(
        self,
        case_id: str,
        features: torch.Tensor,
        metadata: CaseMetadata,
        update_index: bool = True,
    ) -> bool:
        """Add a new case to the database"""

        with self.lock:
            # Check if case already exists
            if case_id in self.case_id_to_idx:
                self.logger.warning(f"Case {case_id} already exists - skipping")
                return False

            # Validate features
            if features.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.feature_dim}, got {features.shape[-1]}"
                )

            # Add to cases list
            feature_idx = len(self.cases)
            self.cases.append(metadata)
            self.case_id_to_idx[case_id] = feature_idx

            # Save to metadata database
            self._save_case_metadata(metadata, feature_idx)

            # Save features
            self._save_case_features(features, feature_idx)

            # Update FAISS index
            if update_index:
                self._add_to_index(features)

            self.logger.debug(f"Added case {case_id} to database")
            return True

    def add_cases_batch(self, cases: List[Tuple[str, torch.Tensor, CaseMetadata]]) -> int:
        """Add multiple cases in batch for efficiency"""

        with self.lock:
            added_count = 0
            features_to_add = []

            for case_id, features, metadata in cases:
                if case_id not in self.case_id_to_idx:
                    # Validate features
                    if features.shape[-1] != self.feature_dim:
                        self.logger.warning(f"Skipping case {case_id}: feature dimension mismatch")
                        continue

                    # Add to cases list
                    feature_idx = len(self.cases)
                    self.cases.append(metadata)
                    self.case_id_to_idx[case_id] = feature_idx

                    # Save metadata
                    self._save_case_metadata(metadata, feature_idx)

                    # Save features
                    self._save_case_features(features, feature_idx)

                    # Collect for batch index update
                    features_to_add.append(features.cpu().numpy().flatten())
                    added_count += 1

            # Batch update FAISS index
            if features_to_add:
                self._add_to_index_batch(np.array(features_to_add))

            self.logger.info(f"Added {added_count} cases to database in batch")
            return added_count

    def _save_case_metadata(self, metadata: CaseMetadata, feature_idx: int):
        """Save case metadata to SQLite database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metadata.case_id,
                metadata.slide_id,
                metadata.patient_id,
                metadata.institution,
                metadata.scanner_type,
                metadata.magnification,
                metadata.stain_type,
                metadata.tissue_type,
                metadata.diagnosis,
                metadata.grade,
                metadata.stage,
                json.dumps(metadata.molecular_markers),
                metadata.pathologist_id,
                metadata.confidence_score,
                metadata.annotation_time.isoformat(),
                metadata.image_quality_score,
                json.dumps(metadata.artifact_flags),
                json.dumps(metadata.demographics),
                metadata.treatment_response,
                metadata.follow_up_months,
                json.dumps(metadata.tags),
                feature_idx,
            ),
        )

        conn.commit()
        conn.close()

    def _save_case_features(self, features: torch.Tensor, feature_idx: int):
        """Save case features to disk"""
        features_np = features.cpu().numpy().flatten()

        # Load existing features or create new array
        if self.features_path.exists():
            existing_features = np.load(self.features_path)
            # Extend array
            new_features = np.zeros((len(self.cases), self.feature_dim), dtype=np.float32)
            new_features[: existing_features.shape[0]] = existing_features
            new_features[feature_idx] = features_np
        else:
            new_features = np.zeros((len(self.cases), self.feature_dim), dtype=np.float32)
            new_features[feature_idx] = features_np

        np.save(self.features_path, new_features)

    def _add_to_index(self, features: torch.Tensor):
        """Add single feature vector to FAISS index"""
        features_np = features.cpu().numpy().flatten().astype("float32").reshape(1, -1)
        faiss.normalize_L2(features_np)

        if self.index is None:
            self._build_index()

        self.index.add(features_np)
        self._save_index()

    def _add_to_index_batch(self, features_batch: np.ndarray):
        """Add batch of feature vectors to FAISS index"""
        features_batch = features_batch.astype("float32")
        faiss.normalize_L2(features_batch)

        if self.index is None:
            self._build_index()

        self.index.add(features_batch)
        self._save_index()

    def _build_index(self):
        """Build FAISS index from scratch"""
        if len(self.cases) == 0:
            return

        # Load all features
        if not self.features_path.exists():
            self.logger.error("No features file found - cannot build index")
            return

        features = np.load(self.features_path).astype("float32")
        faiss.normalize_L2(features)

        # Create index based on type
        if self.index_type == "Flat":
            index = faiss.IndexFlatIP(self.feature_dim)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            index = faiss.IndexIVFFlat(quantizer, self.feature_dim, self.nlist)
            index.train(features)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.feature_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add features to index
        index.add(features)

        # Move to GPU if requested
        if self.use_gpu:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        self.index = index
        self._save_index()

        self.logger.info(f"Built FAISS index with {len(self.cases)} cases")

    def _rebuild_index(self):
        """Rebuild index from existing features"""
        self._build_index()

    def _save_index(self):
        """Save FAISS index to disk"""
        if self.index is None:
            return

        # Move to CPU for saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, str(self.index_path))

    def retrieve_similar(self, query: RetrievalQuery) -> List[SimilarCase]:
        """Retrieve similar cases based on query"""

        with self.lock:
            if self.index is None or len(self.cases) == 0:
                return []

            # Prepare query features
            query_features = query.features.cpu().numpy().flatten().astype("float32").reshape(1, -1)
            faiss.normalize_L2(query_features)

            # Search with larger k for filtering
            search_k = min(query.k * 5, len(self.cases))
            similarities, indices = self.index.search(query_features, search_k)

            # Filter and rank results
            results = []
            seen_diagnoses = set()

            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= len(self.cases) or idx < 0:
                    continue

                case_metadata = self.cases[idx]

                # Apply filters
                if not self._passes_filters(case_metadata, query):
                    continue

                # Apply diversity constraint
                if query.diversity_weight > 0:
                    if case_metadata.diagnosis in seen_diagnoses:
                        # Reduce similarity score for repeated diagnoses
                        sim *= 1.0 - query.diversity_weight
                    seen_diagnoses.add(case_metadata.diagnosis)

                # Load features for this case
                case_features = self._load_case_features(idx)

                similar_case = SimilarCase(
                    case_id=case_metadata.case_id,
                    similarity_score=float(sim),
                    diagnosis=case_metadata.diagnosis,
                    confidence=case_metadata.confidence_score,
                    features=case_features,
                    metadata=case_metadata,
                    explanation=self._generate_case_explanation(case_metadata, float(sim)),
                )

                results.append(similar_case)

                if len(results) >= query.k:
                    break

            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)

            return results[: query.k]

    def _passes_filters(self, metadata: CaseMetadata, query: RetrievalQuery) -> bool:
        """Check if case passes all query filters"""

        # Disease filter
        if query.disease_filter and metadata.diagnosis != query.disease_filter:
            return False

        # Institution filter
        if query.institution_filter and metadata.institution != query.institution_filter:
            return False

        # Confidence threshold
        if metadata.confidence_score < query.confidence_threshold:
            return False

        # Quality threshold
        if metadata.image_quality_score < query.quality_threshold:
            return False

        # Exclude case IDs
        if query.exclude_case_ids and metadata.case_id in query.exclude_case_ids:
            return False

        # Required tags
        if query.require_tags:
            if not all(tag in metadata.tags for tag in query.require_tags):
                return False

        # Excluded tags
        if query.exclude_tags:
            if any(tag in metadata.tags for tag in query.exclude_tags):
                return False

        # Date range
        if query.date_range:
            start_date, end_date = query.date_range
            if not (start_date <= metadata.annotation_time <= end_date):
                return False

        return True

    def _load_case_features(self, idx: int) -> torch.Tensor:
        """Load features for a specific case"""
        features = np.load(self.features_path)
        return torch.from_numpy(features[idx])

    def _generate_case_explanation(self, metadata: CaseMetadata, similarity: float) -> str:
        """Generate explanation for why this case is similar"""
        explanation = f"Similar case from {metadata.institution} "
        explanation += f"(similarity: {similarity:.3f}). "

        if metadata.grade:
            explanation += f"Grade: {metadata.grade}. "

        if metadata.stage:
            explanation += f"Stage: {metadata.stage}. "

        if metadata.molecular_markers:
            markers = ", ".join([f"{k}: {v}" for k, v in metadata.molecular_markers.items()])
            explanation += f"Markers: {markers}. "

        explanation += f"Pathologist confidence: {metadata.confidence_score:.2%}."

        return explanation

    def get_statistics(self) -> CaseStatistics:
        """Get database statistics"""

        with self.lock:
            if not self.cases:
                return CaseStatistics(
                    total_cases=0,
                    cases_by_disease={},
                    cases_by_institution={},
                    cases_by_scanner={},
                    average_quality_score=0.0,
                    date_range=(datetime.now(), datetime.now()),
                    feature_dimensionality=self.feature_dim,
                    index_size_mb=0.0,
                )

            # Count by disease
            cases_by_disease = {}
            for case in self.cases:
                cases_by_disease[case.diagnosis] = cases_by_disease.get(case.diagnosis, 0) + 1

            # Count by institution
            cases_by_institution = {}
            for case in self.cases:
                cases_by_institution[case.institution] = (
                    cases_by_institution.get(case.institution, 0) + 1
                )

            # Count by scanner
            cases_by_scanner = {}
            for case in self.cases:
                cases_by_scanner[case.scanner_type] = cases_by_scanner.get(case.scanner_type, 0) + 1

            # Average quality score
            avg_quality = sum(case.image_quality_score for case in self.cases) / len(self.cases)

            # Date range
            dates = [case.annotation_time for case in self.cases]
            date_range = (min(dates), max(dates))

            # Index size
            index_size_mb = 0.0
            if self.index_path.exists():
                index_size_mb = self.index_path.stat().st_size / (1024 * 1024)

            return CaseStatistics(
                total_cases=len(self.cases),
                cases_by_disease=cases_by_disease,
                cases_by_institution=cases_by_institution,
                cases_by_scanner=cases_by_scanner,
                average_quality_score=avg_quality,
                date_range=date_range,
                feature_dimensionality=self.feature_dim,
                index_size_mb=index_size_mb,
            )

    def update_case_metadata(self, case_id: str, updated_metadata: CaseMetadata) -> bool:
        """Update metadata for an existing case"""

        with self.lock:
            if case_id not in self.case_id_to_idx:
                return False

            idx = self.case_id_to_idx[case_id]
            self.cases[idx] = updated_metadata

            # Update in database
            self._save_case_metadata(updated_metadata, idx)

            return True

    def remove_case(self, case_id: str) -> bool:
        """Remove a case from the database"""

        with self.lock:
            if case_id not in self.case_id_to_idx:
                return False

            # Remove from metadata database
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
            conn.commit()
            conn.close()

            # Remove from memory structures
            idx = self.case_id_to_idx[case_id]
            del self.cases[idx]
            del self.case_id_to_idx[case_id]

            # Update indices for remaining cases
            for cid, cidx in self.case_id_to_idx.items():
                if cidx > idx:
                    self.case_id_to_idx[cid] = cidx - 1

            # Rebuild index (expensive but necessary)
            self._rebuild_index()

            return True

    def export_cases(self, output_path: str, format: str = "json") -> bool:
        """Export cases to file"""

        try:
            output_path = Path(output_path)

            if format == "json":
                cases_data = [case.to_dict() for case in self.cases]
                with open(output_path, "w") as f:
                    json.dump(cases_data, f, indent=2)
            elif format == "csv":
                import pandas as pd

                cases_data = [case.to_dict() for case in self.cases]
                df = pd.json_normalize(cases_data)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unknown export format: {format}")

            self.logger.info(f"Exported {len(self.cases)} cases to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export cases: {e}")
            return False


# Example usage
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Create case database
    db = CaseDatabase(database_path="./case_database", feature_dim=2048, index_type="IVF")

    # Create sample metadata
    metadata = CaseMetadata(
        case_id="case_001",
        slide_id="slide_001",
        patient_id="patient_001_anon",
        institution="Hospital A",
        scanner_type="Leica Aperio",
        magnification=40.0,
        stain_type="H&E",
        tissue_type="breast",
        diagnosis="invasive ductal carcinoma",
        grade="Grade 2",
        stage="T2N1M0",
        molecular_markers={"ER": "positive", "PR": "positive", "HER2": "negative"},
        pathologist_id="pathologist_001",
        confidence_score=0.92,
        annotation_time=datetime.now(),
        image_quality_score=0.88,
        artifact_flags=["minor_blur"],
        demographics={"age_group": "50-60", "sex": "F"},
        treatment_response="complete_response",
        follow_up_months=24,
        tags=["training", "high_quality"],
    )

    # Add case
    features = torch.randn(2048)
    success = db.add_case("case_001", features, metadata)
    print(f"Added case: {success}")

    # Create query
    query = RetrievalQuery(
        features=torch.randn(2048), disease_filter="invasive ductal carcinoma", k=5
    )

    # Retrieve similar cases
    similar_cases = db.retrieve_similar(query)
    print(f"Found {len(similar_cases)} similar cases")

    # Get statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats.total_cases} total cases")
    print(f"Cases by disease: {stats.cases_by_disease}")
