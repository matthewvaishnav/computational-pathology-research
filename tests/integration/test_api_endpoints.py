#!/usr/bin/env python3
"""
API Endpoint Integration Tests

Tests all API endpoints for the Medical AI platform including mobile app integration,
authentication, and comprehensive endpoint validation.
"""

import json
import time
from typing import Dict, List, Optional

import pytest
import requests


class APIEndpointTests:
    """Comprehensive API endpoint testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API endpoint tests."""
        self.base_url = base_url
        self.session = requests.Session()
        self.auth_token = None

    def test_authentication_endpoints(self) -> bool:
        """Test authentication and authorization endpoints."""
        print("\n🔐 Testing Authentication Endpoints...")

        try:
            # Test user registration
            register_data = {
                "username": "test_user",
                "email": "test@example.com",
                "password": "test_password_123",
                "role": "pathologist",
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/auth/register", json=register_data, timeout=10
            )

            if response.status_code == 201:
                print("✅ User registration successful")
            elif response.status_code == 409:
                print("✅ User already exists (expected)")
            else:
                print(f"❌ Registration failed: {response.status_code}")
                return False

            # Test user login
            login_data = {"username": "test_user", "password": "test_password_123"}

            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login", json=login_data, timeout=10
            )

            if response.status_code == 200:
                auth_result = response.json()
                self.auth_token = auth_result.get("access_token")

                if self.auth_token:
                    print("✅ User login successful")
                    # Set authorization header for subsequent requests
                    self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
                else:
                    print("❌ No access token returned")
                    return False
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False

            # Test token validation
            response = self.session.get(f"{self.base_url}/api/v1/auth/me", timeout=10)

            if response.status_code == 200:
                user_info = response.json()
                print(f"✅ Token validation successful: {user_info.get('username')}")
                return True
            else:
                print(f"❌ Token validation failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Authentication test error: {e}")
            return False

    def test_mobile_app_endpoints(self) -> bool:
        """Test mobile app specific endpoints."""
        print("\n📱 Testing Mobile App Endpoints...")

        try:
            # Test mobile device registration
            device_data = {
                "device_id": "test_device_123",
                "platform": "iOS",
                "version": "1.0.0",
                "push_token": "test_push_token",
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/mobile/register-device", json=device_data, timeout=10
            )

            if response.status_code == 200:
                print("✅ Mobile device registration successful")
            else:
                print(f"❌ Device registration failed: {response.status_code}")
                return False

            # Test mobile sync endpoint
            response = self.session.get(f"{self.base_url}/api/v1/mobile/sync", timeout=10)

            if response.status_code == 200:
                sync_data = response.json()
                print(
                    f"✅ Mobile sync successful: {len(sync_data.get('pending_cases', []))} pending cases"
                )
            else:
                print(f"❌ Mobile sync failed: {response.status_code}")
                return False

            # Test offline case download
            response = self.session.get(f"{self.base_url}/api/v1/mobile/cases/offline", timeout=10)

            if response.status_code == 200:
                offline_cases = response.json()
                print(f"✅ Offline cases retrieved: {len(offline_cases.get('cases', []))} cases")
            else:
                print(f"❌ Offline cases failed: {response.status_code}")
                return False

            # Test mobile model download
            response = self.session.get(f"{self.base_url}/api/v1/mobile/model/download", timeout=30)

            if response.status_code == 200:
                print("✅ Mobile model download successful")
            else:
                print(f"❌ Mobile model download failed: {response.status_code}")
                return False

            return True

        except Exception as e:
            print(f"❌ Mobile app test error: {e}")
            return False

    def test_case_management_endpoints(self) -> bool:
        """Test case management and workflow endpoints."""
        print("\n📋 Testing Case Management Endpoints...")

        try:
            # Test case list retrieval
            response = self.session.get(
                f"{self.base_url}/api/v1/cases",
                params={"limit": 10, "status": "pending"},
                timeout=10,
            )

            if response.status_code == 200:
                cases = response.json()
                print(f"✅ Case list retrieved: {len(cases.get('cases', []))} cases")
            else:
                print(f"❌ Case list failed: {response.status_code}")
                return False

            # Test case creation
            case_data = {
                "patient_id": "TEST_PATIENT_001",
                "study_id": "TEST_STUDY_001",
                "priority": "normal",
                "case_type": "breast_cancer_screening",
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/cases", json=case_data, timeout=10
            )

            if response.status_code == 201:
                case_result = response.json()
                case_id = case_result.get("case_id")
                print(f"✅ Case created successfully: {case_id}")
            else:
                print(f"❌ Case creation failed: {response.status_code}")
                return False

            # Test case details retrieval
            if case_id:
                response = self.session.get(f"{self.base_url}/api/v1/cases/{case_id}", timeout=10)

                if response.status_code == 200:
                    case_details = response.json()
                    print(f"✅ Case details retrieved: {case_details.get('status')}")
                else:
                    print(f"❌ Case details failed: {response.status_code}")
                    return False

            # Test case status update
            if case_id:
                status_data = {"status": "in_progress", "notes": "Starting analysis"}

                response = self.session.patch(
                    f"{self.base_url}/api/v1/cases/{case_id}/status", json=status_data, timeout=10
                )

                if response.status_code == 200:
                    print("✅ Case status updated successfully")
                else:
                    print(f"❌ Case status update failed: {response.status_code}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Case management test error: {e}")
            return False

    def test_reporting_endpoints(self) -> bool:
        """Test reporting and analytics endpoints."""
        print("\n📊 Testing Reporting Endpoints...")

        try:
            # Test analytics dashboard data
            response = self.session.get(f"{self.base_url}/api/v1/analytics/dashboard", timeout=10)

            if response.status_code == 200:
                dashboard_data = response.json()
                print(f"✅ Dashboard data retrieved: {len(dashboard_data.keys())} metrics")
            else:
                print(f"❌ Dashboard data failed: {response.status_code}")
                return False

            # Test performance metrics
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/performance", params={"period": "7d"}, timeout=10
            )

            if response.status_code == 200:
                performance_data = response.json()
                print(
                    f"✅ Performance metrics retrieved: {performance_data.get('total_cases', 0)} cases"
                )
            else:
                print(f"❌ Performance metrics failed: {response.status_code}")
                return False

            # Test report generation
            report_data = {
                "report_type": "weekly_summary",
                "date_range": {"start": "2026-04-20", "end": "2026-04-27"},
                "format": "pdf",
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/reports/generate", json=report_data, timeout=30
            )

            if response.status_code == 202:
                report_result = response.json()
                report_id = report_result.get("report_id")
                print(f"✅ Report generation started: {report_id}")
            else:
                print(f"❌ Report generation failed: {response.status_code}")
                return False

            # Test report status check
            if report_id:
                time.sleep(2)  # Wait for report processing
                response = self.session.get(
                    f"{self.base_url}/api/v1/reports/{report_id}/status", timeout=10
                )

                if response.status_code == 200:
                    report_status = response.json()
                    print(f"✅ Report status retrieved: {report_status.get('status')}")
                else:
                    print(f"❌ Report status failed: {response.status_code}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Reporting test error: {e}")
            return False

    def test_admin_endpoints(self) -> bool:
        """Test administrative endpoints."""
        print("\n👑 Testing Admin Endpoints...")

        try:
            # Test user management
            response = self.session.get(
                f"{self.base_url}/api/v1/admin/users", params={"limit": 10}, timeout=10
            )

            if response.status_code == 200:
                users = response.json()
                print(f"✅ User list retrieved: {len(users.get('users', []))} users")
            else:
                print(f"❌ User list failed: {response.status_code}")
                return False

            # Test system configuration
            response = self.session.get(f"{self.base_url}/api/v1/admin/config", timeout=10)

            if response.status_code == 200:
                config = response.json()
                print(f"✅ System config retrieved: {len(config.keys())} settings")
            else:
                print(f"❌ System config failed: {response.status_code}")
                return False

            # Test audit logs
            response = self.session.get(
                f"{self.base_url}/api/v1/admin/audit-logs", params={"limit": 10}, timeout=10
            )

            if response.status_code == 200:
                audit_logs = response.json()
                print(f"✅ Audit logs retrieved: {len(audit_logs.get('logs', []))} entries")
            else:
                print(f"❌ Audit logs failed: {response.status_code}")
                return False

            return True

        except Exception as e:
            print(f"❌ Admin endpoints test error: {e}")
            return False

    def test_websocket_endpoints(self) -> bool:
        """Test WebSocket endpoints for real-time updates."""
        print("\n🔌 Testing WebSocket Endpoints...")

        try:
            # Test WebSocket connection info
            response = self.session.get(f"{self.base_url}/api/v1/ws/info", timeout=10)

            if response.status_code == 200:
                ws_info = response.json()
                print(f"✅ WebSocket info retrieved: {ws_info.get('endpoint')}")
            else:
                print(f"❌ WebSocket info failed: {response.status_code}")
                return False

            # Test notification subscription
            subscription_data = {
                "topics": ["case_updates", "system_alerts"],
                "device_id": "test_device_123",
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/notifications/subscribe",
                json=subscription_data,
                timeout=10,
            )

            if response.status_code == 200:
                print("✅ Notification subscription successful")
            else:
                print(f"❌ Notification subscription failed: {response.status_code}")
                return False

            return True

        except Exception as e:
            print(f"❌ WebSocket test error: {e}")
            return False

    def run_all_endpoint_tests(self) -> Dict[str, bool]:
        """Run all API endpoint tests."""
        print("🚀 Starting API Endpoint Tests")
        print("=" * 50)

        tests = [
            ("Authentication", self.test_authentication_endpoints),
            ("Mobile App", self.test_mobile_app_endpoints),
            ("Case Management", self.test_case_management_endpoints),
            ("Reporting", self.test_reporting_endpoints),
            ("Admin", self.test_admin_endpoints),
            ("WebSocket", self.test_websocket_endpoints),
        ]

        results = {}
        passed = 0
        total = len(tests)

        for test_name, test_method in tests:
            try:
                result = test_method()
                results[test_name] = result

                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")

            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False

        print(f"\nAPI Endpoint Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return results


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    test_suite = APIEndpointTests(base_url=base_url)
    results = test_suite.run_all_endpoint_tests()

    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
