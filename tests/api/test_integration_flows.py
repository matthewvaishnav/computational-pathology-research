#!/usr/bin/env python3
"""
Integration Tests for End-to-End API Flows

Tests complete workflows across the refactored API routers to ensure
the modular architecture maintains proper end-to-end functionality.

**Validates: Requirements 10.1-10.5, 12.1-12.4**
"""

import io
import json
import time
from typing import Dict, Optional

import pytest
from fastapi.testclient import TestClient


class TestIntegrationFlows:
    """Integration tests for end-to-end API flows."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.test_user_data = {
            "username": "integration_test_user",
            "email": "integration@test.com",
            "password": "TestPassword123!"
        }
        self.admin_user_data = {
            "username": "integration_admin_user", 
            "email": "admin@test.com",
            "password": "AdminPassword123!"
        }
        self.access_token = None
        self.admin_token = None

    def _register_and_login_user(self, client: TestClient, user_data: Dict) -> str:
        """Helper method to register and login a user, returning access token."""
        # Register user (may already exist)
        register_response = client.post("/api/v1/auth/register", json=user_data)
        assert register_response.status_code in [201, 409]  # Created or already exists
        
        # Login user
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        login_response = client.post("/api/v1/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        token_data = login_response.json()
        assert "access_token" in token_data
        return token_data["access_token"]

    def _create_auth_headers(self, token: str) -> Dict[str, str]:
        """Helper method to create authorization headers."""
        return {"Authorization": f"Bearer {token}"}

    def test_user_registration_and_login_flow(self, test_client: TestClient):
        """
        Test complete user registration and login flow.
        
        **Validates: Requirements 4.1-4.7, 10.1-10.5, 12.1-12.4**
        """
        # Step 1: Register new user
        register_response = test_client.post("/api/v1/auth/register", json=self.test_user_data)
        assert register_response.status_code in [201, 409]  # Created or conflict if exists
        
        if register_response.status_code == 201:
            register_data = register_response.json()
            assert "user_id" in register_data
            assert register_data["username"] == self.test_user_data["username"]
        
        # Step 2: Login with credentials
        login_data = {
            "username": self.test_user_data["username"],
            "password": self.test_user_data["password"]
        }
        login_response = test_client.post("/api/v1/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        token_data = login_response.json()
        assert "access_token" in token_data
        assert "token_type" in token_data
        assert token_data["token_type"] == "bearer"
        
        # Step 3: Get current user info with JWT token
        headers = self._create_auth_headers(token_data["access_token"])
        me_response = test_client.get("/api/v1/auth/me", headers=headers)
        assert me_response.status_code == 200
        
        user_info = me_response.json()
        assert user_info["username"] == self.test_user_data["username"]
        assert user_info["email"] == self.test_user_data["email"]
        
        # Step 4: Verify JWT token works for protected endpoints
        cases_response = test_client.get("/api/v1/cases", headers=headers)
        assert cases_response.status_code == 200

    def test_image_analysis_flow(self, test_client: TestClient):
        """
        Test complete image analysis workflow.
        
        **Validates: Requirements 5.1-5.9, 10.1-10.5, 12.1-12.4**
        """
        # Setup: Login user
        self.access_token = self._register_and_login_user(test_client, self.test_user_data)
        headers = self._create_auth_headers(self.access_token)
        
        # Step 1: Upload image for analysis
        test_image_data = b"fake_image_data_for_testing" * 100  # Create test image data
        files = {
            "file": ("test_image.png", io.BytesIO(test_image_data), "image/png")
        }
        
        upload_response = test_client.post(
            "/api/v1/analyze/upload",
            files=files,
            headers=headers
        )
        assert upload_response.status_code == 200
        
        upload_data = upload_response.json()
        assert "analysis_id" in upload_data
        analysis_id = upload_data["analysis_id"]
        
        # Step 2: Poll for analysis result
        max_attempts = 10
        for attempt in range(max_attempts):
            result_response = test_client.get(
                f"/api/v1/analyze/{analysis_id}",
                headers=headers
            )
            assert result_response.status_code == 200
            
            result_data = result_response.json()
            assert "status" in result_data
            
            if result_data["status"] in ["completed", "failed"]:
                break
            
            time.sleep(1)  # Wait before next poll
        
        # Step 3: Verify result format
        assert result_data["status"] in ["completed", "failed"]
        assert "analysis_id" in result_data
        
        if result_data["status"] == "completed":
            assert "result" in result_data
            assert "confidence" in result_data["result"]

    def test_case_management_flow(self, test_client: TestClient):
        """
        Test complete case management workflow.
        
        **Validates: Requirements 5.1-5.9, 10.1-10.5, 12.1-12.4**
        """
        # Setup: Login user
        self.access_token = self._register_and_login_user(test_client, self.test_user_data)
        headers = self._create_auth_headers(self.access_token)
        
        # Step 1: Create new case
        case_data = {
            "patient_id": "TEST_PATIENT_001",
            "study_id": "TEST_STUDY_001", 
            "priority": "normal",
            "case_type": "breast_cancer_screening"
        }
        
        create_response = test_client.post("/api/v1/cases", json=case_data, headers=headers)
        assert create_response.status_code == 201
        
        create_result = create_response.json()
        assert "case_id" in create_result
        case_id = create_result["case_id"]
        
        # Step 2: List cases (verify only user's cases returned)
        list_response = test_client.get("/api/v1/cases", headers=headers)
        assert list_response.status_code == 200
        
        cases_data = list_response.json()
        assert "cases" in cases_data
        
        # Find our created case
        created_case = None
        for case in cases_data["cases"]:
            if case["case_id"] == case_id:
                created_case = case
                break
        
        assert created_case is not None
        assert created_case["patient_id"] == case_data["patient_id"]
        assert created_case["study_id"] == case_data["study_id"]
        
        # Step 3: Get case details
        detail_response = test_client.get(f"/api/v1/cases/{case_id}", headers=headers)
        assert detail_response.status_code == 200
        
        detail_data = detail_response.json()
        assert detail_data["case_id"] == case_id
        assert detail_data["patient_id"] == case_data["patient_id"]
        assert detail_data["status"] == "pending"  # Default status
        
        # Step 4: Update case status
        status_update = {
            "status": "in_progress",
            "notes": "Starting analysis"
        }
        
        update_response = test_client.put(
            f"/api/v1/cases/{case_id}/status",
            json=status_update,
            headers=headers
        )
        assert update_response.status_code == 200
        
        # Step 5: Verify status update
        updated_detail_response = test_client.get(f"/api/v1/cases/{case_id}", headers=headers)
        assert updated_detail_response.status_code == 200
        
        updated_data = updated_detail_response.json()
        assert updated_data["status"] == "in_progress"

    def test_admin_operations_flow(self, test_client: TestClient):
        """
        Test complete admin operations workflow.
        
        **Validates: Requirements 6.1-6.8, 10.1-10.5, 12.1-12.4**
        """
        # Setup: Login admin user
        self.admin_token = self._register_and_login_user(test_client, self.admin_user_data)
        admin_headers = self._create_auth_headers(self.admin_token)
        
        # Step 1: List all users
        users_response = test_client.get("/api/v1/admin/users", headers=admin_headers)
        assert users_response.status_code == 200
        
        users_data = users_response.json()
        assert "users" in users_data
        assert len(users_data["users"]) >= 1  # At least the admin user
        
        # Step 2: Get system config
        config_response = test_client.get("/api/v1/admin/config", headers=admin_headers)
        assert config_response.status_code == 200
        
        config_data = config_response.json()
        assert isinstance(config_data, dict)
        assert len(config_data) > 0  # Should have some config settings
        
        # Step 3: Generate report
        report_request = {
            "report_type": "weekly_summary",
            "parameters": {
                "start_date": "2026-04-20",
                "end_date": "2026-04-27"
            }
        }
        
        report_response = test_client.post(
            "/api/v1/admin/reports/generate",
            json=report_request,
            headers=admin_headers
        )
        assert report_response.status_code == 202  # Accepted for processing
        
        report_data = report_response.json()
        assert "report_id" in report_data
        report_id = report_data["report_id"]
        
        # Step 4: Check report status
        status_response = test_client.get(
            f"/api/v1/admin/reports/{report_id}/status",
            headers=admin_headers
        )
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert "status" in status_data
        assert status_data["status"] in ["pending", "processing", "completed", "failed"]

    def test_mobile_device_flow(self, test_client: TestClient):
        """
        Test complete mobile device workflow.
        
        **Validates: Requirements 7.1-7.6, 10.1-10.5, 12.1-12.4**
        """
        # Setup: Login user
        self.access_token = self._register_and_login_user(test_client, self.test_user_data)
        headers = self._create_auth_headers(self.access_token)
        
        # Step 1: Register mobile device
        device_data = {
            "device_id": "test_device_123",
            "device_type": "mobile",
            "os_version": "iOS 17.0",
            "app_version": "1.0.0"
        }
        
        register_response = test_client.post(
            "/api/v1/mobile/register-device",
            json=device_data,
            headers=headers
        )
        assert register_response.status_code == 200
        
        register_result = register_response.json()
        assert "device_id" in register_result
        assert register_result["device_id"] == device_data["device_id"]
        
        # Step 2: Sync data
        sync_response = test_client.get("/api/v1/mobile/sync", headers=headers)
        assert sync_response.status_code == 200
        
        sync_data = sync_response.json()
        assert "pending_cases" in sync_data
        assert "last_sync" in sync_data
        
        # Step 3: Get offline cases
        offline_response = test_client.get("/api/v1/mobile/cases/offline", headers=headers)
        assert offline_response.status_code == 200
        
        offline_data = offline_response.json()
        assert "cases" in offline_data
        assert isinstance(offline_data["cases"], list)
        
        # Step 4: Download mobile model
        model_response = test_client.get("/api/v1/mobile/model/download", headers=headers)
        assert model_response.status_code == 200
        
        # Should return binary model data or redirect
        assert len(model_response.content) > 0 or model_response.status_code == 302

    def test_cross_router_workflow(self, test_client: TestClient):
        """
        Test workflow that spans multiple routers to ensure proper integration.
        
        **Validates: Requirements 10.1-10.5, 12.1-12.4**
        """
        # Setup: Login user and admin
        self.access_token = self._register_and_login_user(test_client, self.test_user_data)
        self.admin_token = self._register_and_login_user(test_client, self.admin_user_data)
        
        user_headers = self._create_auth_headers(self.access_token)
        admin_headers = self._create_auth_headers(self.admin_token)
        
        # Step 1: User creates case (Analysis router)
        case_data = {
            "patient_id": "CROSS_TEST_001",
            "study_id": "CROSS_STUDY_001",
            "priority": "high",
            "case_type": "urgent_screening"
        }
        
        create_response = test_client.post("/api/v1/cases", json=case_data, headers=user_headers)
        assert create_response.status_code == 201
        case_id = create_response.json()["case_id"]
        
        # Step 2: User uploads image for analysis (Analysis router)
        test_image_data = b"cross_router_test_image" * 50
        files = {
            "file": ("cross_test.png", io.BytesIO(test_image_data), "image/png")
        }
        
        upload_response = test_client.post(
            "/api/v1/analyze/upload",
            files=files,
            data={"case_id": case_id},
            headers=user_headers
        )
        assert upload_response.status_code == 200
        analysis_id = upload_response.json()["analysis_id"]
        
        # Step 3: Admin monitors system (Monitoring router)
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        
        metrics_response = test_client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # Step 4: Admin views audit logs (Admin router)
        audit_response = test_client.get("/api/v1/admin/audit-logs", headers=admin_headers)
        assert audit_response.status_code == 200
        
        # Step 5: User registers mobile device (Mobile router)
        device_data = {
            "device_id": "cross_test_device",
            "device_type": "mobile",
            "os_version": "Android 14",
            "app_version": "1.0.0"
        }
        
        device_response = test_client.post(
            "/api/v1/mobile/register-device",
            json=device_data,
            headers=user_headers
        )
        assert device_response.status_code == 200
        
        # Step 6: Verify all operations completed successfully
        # Check case exists
        case_response = test_client.get(f"/api/v1/cases/{case_id}", headers=user_headers)
        assert case_response.status_code == 200
        
        # Check analysis exists
        analysis_response = test_client.get(f"/api/v1/analyze/{analysis_id}", headers=user_headers)
        assert analysis_response.status_code == 200

    def test_error_handling_across_routers(self, test_client: TestClient):
        """
        Test error handling consistency across all routers.
        
        **Validates: Requirements 3.1-3.5, 10.1-10.5, 12.1-12.4**
        """
        # Setup: Login user
        self.access_token = self._register_and_login_user(test_client, self.test_user_data)
        headers = self._create_auth_headers(self.access_token)
        
        # Test 404 errors across routers
        error_tests = [
            ("/api/v1/auth/nonexistent", "GET"),
            ("/api/v1/cases/nonexistent-case-id", "GET"),
            ("/api/v1/analyze/nonexistent-analysis-id", "GET"),
            ("/api/v1/mobile/nonexistent-endpoint", "GET"),
        ]
        
        for endpoint, method in error_tests:
            if method == "GET":
                response = test_client.get(endpoint, headers=headers)
            elif method == "POST":
                response = test_client.post(endpoint, json={}, headers=headers)
            
            assert response.status_code == 404
            error_data = response.json()
            assert "detail" in error_data

    def test_authentication_flow_across_routers(self, test_client: TestClient):
        """
        Test that authentication works consistently across all routers.
        
        **Validates: Requirements 15.1-15.7, 10.1-10.5, 12.1-12.4**
        """
        # Test unauthenticated access to protected endpoints
        protected_endpoints = [
            "/api/v1/auth/me",
            "/api/v1/cases",
            "/api/v1/analyze/upload",
            "/api/v1/mobile/sync",
            "/api/v1/admin/users",
        ]
        
        for endpoint in protected_endpoints:
            response = test_client.get(endpoint)
            assert response.status_code == 401
            error_data = response.json()
            assert "detail" in error_data
        
        # Test authenticated access works
        self.access_token = self._register_and_login_user(test_client, self.test_user_data)
        headers = self._create_auth_headers(self.access_token)
        
        # These should work with valid token (except admin endpoint)
        accessible_endpoints = [
            "/api/v1/auth/me",
            "/api/v1/cases",
            "/api/v1/mobile/sync",
        ]
        
        for endpoint in accessible_endpoints:
            response = test_client.get(endpoint, headers=headers)
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])