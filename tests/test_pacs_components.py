from datetime import datetime
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import uuid

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from src.clinical.pacs.data_models import (
    DicomPriority,
    OperationResult,
    PACSEndpoint,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
    StudyInfo,
)
from src.clinical.pacs.retrieval_engine import RetrievalEngine
from src.clinical.pacs.security_manager import SecurityManager
from src.clinical.pacs.workflow_orchestrator import WorkflowOrchestrator


def _make_workspace_temp_dir() -> Path:
    path = Path("test_results") / f"pacs_tests_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _make_endpoint() -> PACSEndpoint:
    return PACSEndpoint(
        endpoint_id="test",
        ae_title="TEST_AE",
        host="localhost",
        port=11112,
        vendor=PACSVendor.GENERIC,
        security_config=SecurityConfig(
            tls_enabled=False,
            verify_certificates=False,
            mutual_authentication=False,
        ),
        performance_config=PerformanceConfig(),
        is_primary=True,
    )


def _make_study(uid: str) -> StudyInfo:
    return StudyInfo(
        study_instance_uid=uid,
        patient_id=f"patient-{uid}",
        patient_name="Test Patient",
        study_date=datetime(2026, 4, 23),
        study_description="Test Study",
        modality="SM",
        series_count=1,
        priority=DicomPriority.MEDIUM,
    )


def test_execute_c_move_uses_storage_scp_ae_title(monkeypatch):
    engine = RetrievalEngine(ae_title="HISTOCORE_R")
    endpoint = _make_endpoint()

    class DummyAssoc:
        def __init__(self):
            self.is_established = True
            self.destination_ae = None
            self.released = False

        def send_c_move(self, move_ds, destination_ae, model):
            self.destination_ae = destination_ae
            yield SimpleNamespace(
                Status=0x0000,
                NumberOfCompletedSuboperations=0,
            ), None

        def release(self):
            self.released = True

    assoc = DummyAssoc()
    monkeypatch.setattr(engine.ae, "associate", lambda **kwargs: assoc)
    monkeypatch.setattr(engine, "_collect_stored_files", lambda destination_path: [])

    temp_dir = _make_workspace_temp_dir()
    try:
        engine._execute_c_move_study(endpoint, "1.2.3", temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert assoc.destination_ae == engine.storage_scp_ae_title
    assert assoc.released is True


def test_process_new_studies_returns_all_results_for_queued_work(monkeypatch):
    orchestrator = WorkflowOrchestrator(
        pacs_adapter=SimpleNamespace(),
        clinical_workflow=SimpleNamespace(),
        max_concurrent_studies=1,
    )

    def fake_process(study: StudyInfo) -> OperationResult:
        return OperationResult.success_result(
            operation_id=f"process_{study.study_instance_uid}",
            message=f"processed {study.study_instance_uid}",
        )

    monkeypatch.setattr(orchestrator, "_process_single_study", fake_process)

    studies = [_make_study("1.2.3"), _make_study("1.2.4")]
    results = orchestrator.process_new_studies(studies)

    assert len(results) == 2
    assert {result.operation_id for result in results} == {
        "process_1.2.3",
        "process_1.2.4",
    }
    assert orchestrator._processing_queue == []


def test_stop_and_restart_recreates_processing_executor(monkeypatch):
    orchestrator = WorkflowOrchestrator(
        pacs_adapter=SimpleNamespace(),
        clinical_workflow=SimpleNamespace(),
        max_concurrent_studies=1,
    )

    monkeypatch.setattr(orchestrator, "_polling_loop", lambda: time.sleep(0.01))

    orchestrator.start_automated_polling()
    orchestrator._polling_thread.join(timeout=1)
    orchestrator.stop_automated_polling()

    assert orchestrator._processing_executor is None

    orchestrator.start_automated_polling()
    assert orchestrator._processing_executor is not None
    orchestrator._polling_thread.join(timeout=1)
    orchestrator.stop_automated_polling()


def test_validate_certificate_accepts_leaf_signed_by_ca():
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Test CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime(2026, 1, 1))
        .not_valid_after(datetime(2027, 1, 1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "pacs.local"),
    ])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_subject)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime(2026, 1, 1))
        .not_valid_after(datetime(2027, 1, 1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("pacs.local")]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    temp_dir = _make_workspace_temp_dir()
    try:
        ca_bundle_path = temp_dir / "ca_bundle.pem"
        ca_bundle_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

        validation = SecurityManager().validate_certificate(leaf_cert, ca_bundle_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert validation.is_valid is True
    assert validation.errors == []
