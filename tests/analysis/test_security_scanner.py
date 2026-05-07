"""
Unit tests for Security Scanner.

Tests injection detection, secrets detection, and HIPAA compliance checklist generation.
Requirements: 7.1, 7.2, 7.4
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.analysis.security import SecurityScanner
from src.analysis.models import SecurityAnalysis


class TestSecurityScanner:
    """Test suite for SecurityScanner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.scanner = SecurityScanner(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test scanner initialization."""
        scanner = SecurityScanner("/path/to/project")
        assert scanner.project_path == Path("/path/to/project").resolve()
    
    def test_analyze_returns_security_analysis(self):
        """Test that analyze() returns a SecurityAnalysis object."""
        with patch.object(self.scanner, '_run_bandit_scan', return_value=[]):
            with patch.object(self.scanner, '_detect_injection_risks', return_value=[]):
                with patch.object(self.scanner, '_validate_tls_ssl_config', return_value=[]):
                    with patch.object(self.scanner, '_detect_hardcoded_secrets', return_value=[]):
                        with patch.object(self.scanner, '_assess_hipaa_compliance', return_value=85.0):
                            result = self.scanner.analyze()
        
        assert isinstance(result, SecurityAnalysis)
        assert result.vulnerabilities == []
        assert result.hipaa_compliance_score == 85.0
        assert result.hardcoded_secrets == []
        assert result.injection_risks == []
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100


class TestBanditIntegration:
    """Test Bandit security scanner integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.scanner = SecurityScanner(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_run_bandit_success(self, mock_run):
        """Test successful Bandit execution."""
        # Mock Bandit JSON output
        bandit_output = {
            "results": [
                {
                    "filename": "src/app.py",
                    "issue_confidence": "HIGH",
                    "issue_severity": "HIGH",
                    "issue_text": "Use of insecure MD5 hash function.",
                    "line_number": 15,
                    "test_id": "B303",
                    "test_name": "blacklist"
                },
                {
                    "filename": "src/utils.py",
                    "issue_confidence": "MEDIUM",
                    "issue_severity": "MEDIUM",
                    "issue_text": "Possible hardcoded password: 'secret123'",
                    "line_number": 8,
                    "test_id": "B106",
                    "test_name": "hardcoded_password_string"
                }
            ]
        }
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(bandit_output)
        
        vulnerabilities = self.scanner._run_bandit_scan()
        
        assert len(vulnerabilities) == 2
        
        # Check first vulnerability
        vuln1 = vulnerabilities[0]
        assert vuln1['file'] == 'src/app.py'
        assert vuln1['severity'] == 'HIGH'
        assert vuln1['confidence'] == 'HIGH'
        assert vuln1['line'] == 15
        assert 'MD5 hash' in vuln1['description']
        
        # Check second vulnerability
        vuln2 = vulnerabilities[1]
        assert vuln2['file'] == 'src/utils.py'
        assert vuln2['severity'] == 'MEDIUM'
        assert vuln2['line'] == 8
        assert 'hardcoded password' in vuln2['description']
    
    @patch('subprocess.run')
    def test_run_bandit_no_issues(self, mock_run):
        """Test Bandit execution with no security issues."""
        bandit_output = {"results": []}
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(bandit_output)
        
        vulnerabilities = self.scanner._run_bandit_scan()
        
        assert vulnerabilities == []
    
    @patch('subprocess.run')
    def test_run_bandit_command_failure(self, mock_run):
        """Test Bandit execution failure."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        
        vulnerabilities = self.scanner._run_bandit_scan()
        
        assert vulnerabilities == []
    
    @patch('subprocess.run')
    def test_run_bandit_invalid_json(self, mock_run):
        """Test Bandit execution with invalid JSON output."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "invalid json {"
        
        vulnerabilities = self.scanner._run_bandit_scan()
        
        assert vulnerabilities == []


class TestInjectionVulnerabilityDetection:
    """Test injection vulnerability detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.scanner = SecurityScanner(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_python_file(self, path: str, content: str):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_detect_sql_injection_vulnerabilities(self):
        """Test SQL injection vulnerability detection."""
        vulnerable_code = '''
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    # Vulnerable SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()

def get_user_safe(user_id):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    # Safe parameterized query
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
'''
        
        self.create_python_file('src/database.py', vulnerable_code)
        
        injection_risks = self.scanner._detect_injection_risks()
        
        # Should detect SQL injection in vulnerable function
        assert len(injection_risks) >= 1
        
        # Check that it found the vulnerable pattern
        sql_injection_found = any(
            'sql' in risk['type'].lower() and 'database.py' in risk['file']
            for risk in injection_risks
        )
        assert sql_injection_found
    
    def test_detect_command_injection_vulnerabilities(self):
        """Test command injection vulnerability detection."""
        vulnerable_code = '''
import subprocess
import os

def process_file(filename):
    # Vulnerable command injection
    os.system(f"cat {filename}")
    
def process_file_unsafe(filename):
    # Another vulnerable pattern
    subprocess.call(f"ls -la {filename}", shell=True)

def process_file_safe(filename):
    # Safe approach
    subprocess.run(["cat", filename], check=True)
'''
        
        self.create_python_file('src/processor.py', vulnerable_code)
        
        injection_risks = self.scanner._detect_injection_risks()
        
        # Should detect command injection vulnerabilities
        command_injection_found = any(
            'command' in risk['type'].lower() and 'processor.py' in risk['file']
            for risk in injection_risks
        )
        assert command_injection_found
    
    def test_detect_eval_exec_vulnerabilities(self):
        """Test detection of unsafe eval() and exec() usage."""
        vulnerable_code = '''
def dynamic_execution(user_input):
    # Dangerous eval usage
    result = eval(user_input)
    return result

def execute_code(code_string):
    # Dangerous exec usage
    exec(code_string)

def safe_calculation():
    # Safe usage
    result = eval("2 + 2")  # Static string is safer
    return result
'''
        
        self.create_python_file('src/dynamic.py', vulnerable_code)
        
        injection_risks = self.scanner._detect_injection_risks()
        
        # Should detect eval/exec vulnerabilities
        eval_exec_found = any(
            ('eval' in risk['type'].lower() or 'exec' in risk['type'].lower()) 
            and 'dynamic.py' in risk['file']
            for risk in injection_risks
        )
        assert eval_exec_found
    
    def test_detect_injection_vulnerabilities_no_issues(self):
        """Test injection detection with safe code."""
        safe_code = '''
import sqlite3
import subprocess

def get_user_safe(user_id):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()

def process_file_safe(filename):
    subprocess.run(["cat", filename], check=True)
'''
        
        self.create_python_file('src/safe_code.py', safe_code)
        
        injection_risks = self.scanner._detect_injection_risks()
        
        # Should not detect any injection risks in safe code
        assert len(injection_risks) == 0


class TestHardcodedSecretsDetection:
    """Test hardcoded secrets detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.scanner = SecurityScanner(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_python_file(self, path: str, content: str):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_detect_api_keys(self):
        """Test detection of hardcoded API keys."""
        code_with_secrets = '''
# Configuration with hardcoded secrets
API_KEY = "sk-1234567890abcdef1234567890abcdef"
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_1234567890abcdef1234567890abcdef123456"

def connect_to_service():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-API-Key": "secret-key-12345"
    }
    return headers
'''
        
        self.create_python_file('src/config.py', code_with_secrets)
        
        secrets = self.scanner._detect_hardcoded_secrets()
        
        # Should detect multiple types of secrets
        assert len(secrets) >= 3
        
        # Check for different secret types
        secret_types = [secret['type'] for secret in secrets]
        assert 'api_key' in [t.lower() for t in secret_types]
        assert any('aws' in t.lower() for t in secret_types)
    
    def test_detect_passwords(self):
        """Test detection of hardcoded passwords."""
        code_with_passwords = '''
# Database configuration
DB_PASSWORD = "super_secret_password123"
ADMIN_PASS = "admin123"

class DatabaseConnection:
    def __init__(self):
        self.password = "hardcoded_password"
        
def authenticate():
    if password == "default_password":
        return True
'''
        
        self.create_python_file('src/auth.py', code_with_passwords)
        
        secrets = self.scanner._detect_hardcoded_secrets()
        
        # Should detect password patterns
        password_secrets = [s for s in secrets if 'password' in s['type'].lower()]
        assert len(password_secrets) >= 2
    
    def test_detect_private_keys(self):
        """Test detection of private keys."""
        code_with_keys = '''
# RSA private key (truncated for test)
PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890abcdef...
-----END RSA PRIVATE KEY-----"""

# SSH key
SSH_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ..."
'''
        
        self.create_python_file('src/keys.py', code_with_keys)
        
        secrets = self.scanner._detect_hardcoded_secrets()
        
        # Should detect private key patterns
        key_secrets = [s for s in secrets if 'key' in s['type'].lower()]
        assert len(key_secrets) >= 1
    
    def test_detect_hardcoded_secrets_no_secrets(self):
        """Test secrets detection with clean code."""
        clean_code = '''
import os

# Good practices - using environment variables
API_KEY = os.getenv('API_KEY')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

def get_config():
    return {
        'api_key': os.getenv('API_KEY', 'default'),
        'password': os.environ['DB_PASSWORD']
    }
'''
        
        self.create_python_file('src/clean_config.py', clean_code)
        
        secrets = self.scanner._detect_hardcoded_secrets()
        
        # Should not detect any secrets in clean code
        assert len(secrets) == 0


class TestHIPAAComplianceAssessment:
    """Test HIPAA compliance assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.scanner = SecurityScanner(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_assess_hipaa_compliance_placeholder(self):
        """Test HIPAA compliance assessment (currently placeholder)."""
        score = self.scanner._assess_hipaa_compliance()
        
        # Currently returns 75.0 as placeholder
        assert score == 75.0
        assert isinstance(score, float)
    
    def test_hipaa_compliance_checklist_items(self):
        """Test HIPAA compliance checklist generation."""
        # This test validates the structure of HIPAA compliance checking
        # When implemented, it should check for:
        # - Audit logging with 7-year retention
        # - Input sanitization
        # - Access controls
        # - Encryption requirements
        
        checklist = self.scanner._generate_hipaa_checklist()
        
        # Should return a list of compliance items
        assert isinstance(checklist, list)
        
        # When implemented, should have specific checklist items
        expected_categories = [
            'audit_logging',
            'input_sanitization', 
            'access_controls',
            'encryption',
            'data_retention'
        ]
        
        # For now, just verify the method exists and returns a list
        # TODO: Update when HIPAA compliance is fully implemented


class TestSecurityScoreCalculation:
    """Test security score calculation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = SecurityScanner("/test/project")
    
    def test_calculate_security_score_perfect(self):
        """Test security score calculation with perfect security."""
        score = self.scanner._calculate_security_score(
            vulnerabilities=[],
            hipaa_score=100.0,
            secrets=[],
            injection_risks=[]
        )

        # Perfect security should result in high score
        assert score >= 95.0

    def test_calculate_security_score_without_tls_issues(self):
        """Test security score calculation keeps backward compatibility."""
        score = self.scanner._calculate_security_score(
            vulnerabilities=[],
            hipaa_score=80.0,
            secrets=[],
            injection_risks=[]
        )

        assert isinstance(score, float)

    def test_calculate_security_score_poor(self):
        """Test security score calculation with poor security."""
        vulnerabilities = [
            {'severity': 'HIGH', 'confidence': 'HIGH'},
            {'severity': 'CRITICAL', 'confidence': 'HIGH'},
            {'severity': 'MEDIUM', 'confidence': 'MEDIUM'}
        ]
        
        secrets = [
            {'type': 'api_key', 'severity': 'HIGH'},
            {'type': 'password', 'severity': 'MEDIUM'}
        ]
        
        injection_risks = [
            {'type': 'sql_injection', 'severity': 'HIGH'},
            {'type': 'command_injection', 'severity': 'MEDIUM'}
        ]
        
        score = self.scanner._calculate_security_score(
            vulnerabilities=vulnerabilities,
            hipaa_score=30.0,
            secrets=secrets,
            injection_risks=injection_risks
        )
        
        # Poor security should result in low score
        assert score < 40.0
    
    def test_calculate_security_score_mixed(self):
        """Test security score calculation with mixed security."""
        vulnerabilities = [
            {'severity': 'MEDIUM', 'confidence': 'HIGH'}
        ]
        
        secrets = []
        injection_risks = []
        
        score = self.scanner._calculate_security_score(
            vulnerabilities=vulnerabilities,
            hipaa_score=80.0,
            secrets=secrets,
            injection_risks=injection_risks
        )
        
        # Mixed security should result in moderate score
        assert 50.0 <= score <= 85.0

    def test_calculate_security_score_penalizes_tls_issues(self):
        """Test TLS findings reduce the score when present."""
        score = self.scanner._calculate_security_score(
            vulnerabilities=[],
            hipaa_score=100.0,
            secrets=[],
            injection_risks=[],
            tls_issues=[{'type': 'verify_false'}]
        )

        assert score < 100.0


class TestIntegrationWithMockData:
    """Integration tests with mock security data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.scanner = SecurityScanner(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_analysis_with_mock_data(self):
        """Test complete analysis workflow with mocked data."""
        mock_vulnerabilities = [
            {'file': 'src/app.py', 'severity': 'HIGH', 'line': 15, 'description': 'SQL injection'}
        ]
        
        mock_secrets = [
            {'file': 'src/config.py', 'type': 'api_key', 'line': 5, 'value': 'sk-***'}
        ]
        
        mock_injection_risks = [
            {'file': 'src/db.py', 'type': 'sql_injection', 'line': 20, 'pattern': 'f-string in query'}
        ]
        
        # Mock all the detection methods
        with patch.object(self.scanner, '_run_bandit_scan', return_value=mock_vulnerabilities):
            with patch.object(self.scanner, '_detect_injection_risks', return_value=mock_injection_risks):
                with patch.object(self.scanner, '_validate_tls_ssl_config', return_value=[]):
                    with patch.object(self.scanner, '_detect_hardcoded_secrets', return_value=mock_secrets):
                        with patch.object(self.scanner, '_assess_hipaa_compliance', return_value=70.0):
                            
                            result = self.scanner.analyze()
        
        # Verify all fields are populated
        assert len(result.vulnerabilities) == 1
        assert result.hipaa_compliance_score == 70.0
        assert len(result.hardcoded_secrets) == 1
        assert result.hardcoded_secrets[0]['type'] == 'api_key'
        assert result.hardcoded_secrets[0]['file'] == 'src/config.py'
        assert len(result.injection_risks) == 1
        
        # Verify score calculation
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100
    
    def test_analysis_with_high_security(self):
        """Test analysis with high security (few issues)."""
        with patch.object(self.scanner, '_run_bandit_scan', return_value=[]):
            with patch.object(self.scanner, '_detect_injection_risks', return_value=[]):
                with patch.object(self.scanner, '_validate_tls_ssl_config', return_value=[]):
                    with patch.object(self.scanner, '_detect_hardcoded_secrets', return_value=[]):
                        with patch.object(self.scanner, '_assess_hipaa_compliance', return_value=95.0):
                            
                            result = self.scanner.analyze()
        
        # High security should result in high score
        assert result.score > 90.0
    
    def test_analysis_with_poor_security(self):
        """Test analysis with poor security (many issues)."""
        many_vulnerabilities = [
            {'severity': 'CRITICAL', 'confidence': 'HIGH'},
            {'severity': 'HIGH', 'confidence': 'HIGH'},
            {'severity': 'HIGH', 'confidence': 'MEDIUM'}
        ]
        
        many_secrets = [
            {'type': 'api_key'}, {'type': 'password'}, {'type': 'private_key'}
        ]
        
        many_injection_risks = [
            {'type': 'sql_injection'}, {'type': 'command_injection'}
        ]
        
        with patch.object(self.scanner, '_run_bandit_scan', return_value=many_vulnerabilities):
            with patch.object(self.scanner, '_detect_injection_risks', return_value=many_injection_risks):
                with patch.object(self.scanner, '_validate_tls_ssl_config', return_value=['TLS issue']):
                    with patch.object(self.scanner, '_detect_hardcoded_secrets', return_value=many_secrets):
                        with patch.object(self.scanner, '_assess_hipaa_compliance', return_value=40.0):
                            
                            result = self.scanner.analyze()
        
        # Poor security should result in low score
        assert result.score < 50.0


if __name__ == '__main__':
    pytest.main([__file__])
