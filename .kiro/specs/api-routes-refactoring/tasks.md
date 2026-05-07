# Implementation Plan: API Routes Refactoring

## Overview

The API has already been successfully refactored from a monolithic structure into modular routers. This implementation plan focuses on:
1. **Verification tasks**: Document and validate the current refactored state
2. **Optional improvements**: Validators module, error handlers module, enhanced OpenAPI documentation
3. **Testing tasks**: Ensure comprehensive test coverage for the refactored architecture

**Current Status**:
- ✅ Main module: 122 lines (target: <150 lines)
- ✅ 5 routers extracted: auth, analysis, admin, mobile, monitoring
- ✅ Dependencies module: Exists with shared dependency functions
- ✅ Clean separation of concerns achieved

## Tasks

### Phase 1: Verification and Documentation

- [x] 1. Verify current architecture meets requirements
  - Review main.py structure (should be <150 lines)
  - Verify all 5 routers are properly extracted and included
  - Verify dependencies module contains shared functions
  - Document any deviations from requirements
  - _Requirements: 1.1-1.5, 2.1-2.5, 4.1-4.7, 5.1-5.9, 6.1-6.8, 7.1-7.6, 8.1-8.7, 9.1-9.8_

- [x] 2. Measure and document code metrics
  - Measure line counts for main.py and each router
  - Measure cyclomatic complexity for key functions
  - Measure code duplication percentage
  - Document baseline performance metrics (response times, memory usage)
  - Create metrics report in `.kiro/specs/api-routes-refactoring/metrics.md`
  - _Requirements: 14.1-14.7, 11.1-11.4_

- [x] 3. Checkpoint - Review metrics with user
  - Ensure all metrics meet requirements, ask the user if questions arise.

### Phase 2: Optional Improvements

- [ ] 4. Extract validators module (Optional)
  - [ ]* 4.1 Create `src/api/validators.py` module
    - Implement `validate_email(email: str) -> bool` function
    - Implement `validate_password(password: str) -> bool` function
    - Implement `validate_file_upload(file_content: bytes, filename: str) -> tuple[str, str]` function
    - Add comprehensive docstrings
    - _Requirements: 2.1-2.5_
  
  - [ ]* 4.2 Refactor routers to use validators module
    - Update auth router to use `validate_email()` and `validate_password()`
    - Update analysis router to use `validate_file_upload()`
    - Remove duplicate validation logic from routers
    - _Requirements: 2.1-2.5, 14.6_
  
  - [ ]* 4.3 Write unit tests for validators module
    - Test email validation with valid/invalid emails
    - Test password validation with weak/strong passwords
    - Test file upload validation with various file types
    - _Requirements: 2.4, 12.1-12.4_

- [ ] 5. Extract error handlers module (Optional)
  - [ ]* 5.1 Create `src/api/errors.py` module
    - Implement `not_found_handler(request, exc)` function
    - Implement `internal_error_handler(request, exc)` function
    - Implement `validation_error_handler(request, exc)` function
    - Add comprehensive docstrings
    - _Requirements: 3.1-3.5_
  
  - [ ]* 5.2 Update main.py to use error handlers module
    - Import error handlers from `src/api/errors`
    - Register error handlers with app
    - Remove error handler definitions from main.py
    - _Requirements: 3.1-3.5, 9.1-9.8_
  
  - [ ]* 5.3 Write unit tests for error handlers module
    - Test 404 error handler returns correct JSON response
    - Test 500 error handler returns correct JSON response
    - Test validation error handler returns correct JSON response
    - _Requirements: 3.4, 12.1-12.4_

- [ ] 6. Enhance OpenAPI documentation (Optional)
  - [ ]* 6.1 Add detailed descriptions to router endpoints
    - Add `summary` and `description` to all auth endpoints
    - Add `summary` and `description` to all analysis endpoints
    - Add `summary` and `description` to all admin endpoints
    - Add `summary` and `description` to all mobile endpoints
    - Add `summary` and `description` to all monitoring endpoints
    - _Requirements: 13.1-13.5_
  
  - [ ]* 6.2 Add request/response examples to Pydantic models
    - Add `schema_extra` examples to UserRegistration model
    - Add `schema_extra` examples to UserLogin model
    - Add `schema_extra` examples to AnalysisRequest model
    - Add `schema_extra` examples to CaseData model
    - Add `schema_extra` examples to DeviceRegistration model
    - _Requirements: 13.3_
  
  - [ ]* 6.3 Add response schemas to router endpoints
    - Add `responses` parameter to auth endpoints
    - Add `responses` parameter to analysis endpoints
    - Add `responses` parameter to admin endpoints
    - Add `responses` parameter to mobile endpoints
    - Add `responses` parameter to monitoring endpoints
    - _Requirements: 13.3_

- [ ] 7. Checkpoint - Review optional improvements
  - Ensure all tests pass, ask the user if questions arise.

### Phase 3: Testing and Validation

- [x] 8. Write unit tests for routers
  - [x] 8.1 Write unit tests for auth router
    - Test user registration endpoint
    - Test user login endpoint with valid/invalid credentials
    - Test get current user endpoint
    - Test OAuth login and callback endpoints
    - Mock dependencies (database, security functions)
    - _Requirements: 4.1-4.7, 12.1-12.4_
  
  - [x] 8.2 Write unit tests for analysis router
    - Test image upload endpoint
    - Test get analysis result endpoint
    - Test DICOM upload endpoint
    - Test case management endpoints (list, create, get, update)
    - Mock dependencies (database, inference engine)
    - _Requirements: 5.1-5.9, 12.1-12.4_
  
  - [x] 8.3 Write unit tests for admin router
    - Test list users endpoint
    - Test get config endpoint
    - Test audit logs endpoint
    - Test report generation endpoints
    - Mock dependencies (database, admin checks)
    - _Requirements: 6.1-6.8, 12.1-12.4_
  
  - [x] 8.4 Write unit tests for mobile router
    - Test device registration endpoint
    - Test sync endpoint
    - Test offline cases endpoint
    - Test model download endpoint
    - Mock dependencies (database)
    - _Requirements: 7.1-7.6, 12.1-12.4_
  
  - [x] 8.5 Write unit tests for monitoring router
    - Test health check endpoint
    - Test readiness probe endpoint
    - Test metrics endpoint
    - Test security monitoring endpoints
    - Mock dependencies (database, inference engine)
    - _Requirements: 8.1-8.7, 12.1-12.4_

- [ ] 9. Write integration tests for end-to-end flows
  - [ ] 9.1 Write integration test for user registration and login flow
    - Register new user
    - Login with credentials
    - Get current user info
    - Verify JWT token works
    - _Requirements: 4.1-4.7, 10.1-10.5, 12.1-12.4_
  
  - [ ] 9.2 Write integration test for image analysis flow
    - Login as user
    - Upload image for analysis
    - Poll for analysis result
    - Verify result format
    - _Requirements: 5.1-5.9, 10.1-10.5, 12.1-12.4_
  
  - [ ] 9.3 Write integration test for case management flow
    - Login as user
    - Create new case
    - List cases (verify only user's cases returned)
    - Get case details
    - Update case status
    - _Requirements: 5.1-5.9, 10.1-10.5, 12.1-12.4_
  
  - [ ] 9.4 Write integration test for admin operations flow
    - Login as admin user
    - List all users
    - Get system config
    - Generate report
    - Check report status
    - _Requirements: 6.1-6.8, 10.1-10.5, 12.1-12.4_
  
  - [ ] 9.5 Write integration test for mobile device flow
    - Login as user
    - Register mobile device
    - Sync data
    - Get offline cases
    - Download mobile model
    - _Requirements: 7.1-7.6, 10.1-10.5, 12.1-12.4_

- [ ] 10. Write security tests
  - [ ] 10.1 Write test for authentication requirements
    - Test protected endpoints reject unauthenticated requests
    - Test protected endpoints accept valid JWT tokens
    - Test protected endpoints reject expired tokens
    - Test protected endpoints reject invalid tokens
    - _Requirements: 15.1-15.7, 12.1-12.4_
  
  - [ ] 10.2 Write test for authorization requirements
    - Test admin endpoints reject non-admin users
    - Test admin endpoints accept admin users
    - Test users can only access their own resources (IDOR protection)
    - _Requirements: 15.1-15.7, 12.1-12.4_
  
  - [ ] 10.3 Write test for rate limiting
    - Test login endpoint rate limiting (5 requests/minute)
    - Test case creation rate limiting
    - Verify 429 status code returned when limit exceeded
    - _Requirements: 15.1-15.7, 12.1-12.4_
  
  - [ ] 10.4 Write test for input validation
    - Test file upload validation (magic bytes, size limits)
    - Test email validation
    - Test password validation
    - Verify appropriate error messages returned
    - _Requirements: 15.1-15.7, 12.1-12.4_

- [ ] 11. Write performance tests
  - [ ] 11.1 Write test for endpoint response times
    - Measure response times for key endpoints
    - Verify response times are within ±5% of baseline
    - Test with realistic payloads
    - _Requirements: 11.1, 12.1-12.4_
  
  - [ ] 11.2 Write test for memory usage
    - Measure memory usage during typical operations
    - Verify memory usage is within ±5% of baseline
    - Test with multiple concurrent requests
    - _Requirements: 11.2, 12.1-12.4_
  
  - [ ] 11.3 Write test for startup time
    - Measure application startup time
    - Verify startup time is within ±5% of baseline
    - Test with cold start (no cached models)
    - _Requirements: 11.3, 12.1-12.4_

- [x] 12. Measure test coverage
  - Run pytest with coverage reporting
  - Verify overall coverage is above 80%
  - Identify any uncovered code paths
  - Add tests for uncovered paths if needed
  - Generate coverage report in `.kiro/specs/api-routes-refactoring/coverage.md`
  - _Requirements: 12.1-12.4_

- [x] 13. Final checkpoint - Verify all requirements met
  - Ensure all tests pass, ask the user if questions arise.

### Phase 4: Documentation

- [x] 14. Update project documentation
  - [x] 14.1 Update README with router structure
    - Document the 5 routers and their responsibilities
    - Document how to add new endpoints to routers
    - Document the dependencies module
    - _Requirements: 13.4_
  
  - [x] 14.2 Create developer guide for adding endpoints
    - Document the router pattern to follow
    - Document dependency injection pattern
    - Document error handling pattern
    - Document testing pattern
    - Create guide at `.kiro/specs/api-routes-refactoring/developer-guide.md`
    - _Requirements: 13.5_
  
  - [x] 14.3 Document deployment considerations
    - Document required environment variables
    - Document production checklist
    - Document scaling considerations
    - Update deployment documentation
    - _Requirements: 13.1-13.5_

- [x] 15. Final review and completion
  - Review all requirements are met
  - Review all tests pass
  - Review all documentation is updated
  - Create final summary report
  - Mark spec as complete

## Notes

- **Tasks marked with `*` are optional** and can be skipped for faster completion
- **Phase 1 (Verification)** is critical to understand the current state
- **Phase 2 (Optional Improvements)** can be skipped if the current architecture is sufficient
- **Phase 3 (Testing)** is essential to ensure the refactored architecture is robust
- **Phase 4 (Documentation)** ensures the architecture is well-documented for future developers
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and user feedback

## Success Criteria

The refactoring is complete when:

1. ✅ Current architecture is verified and documented
2. ✅ Code metrics meet requirements (main.py <150 lines, etc.)
3. ✅ All unit tests pass with >80% coverage
4. ✅ All integration tests pass
5. ✅ All security tests pass
6. ✅ Performance tests show ±5% of baseline
7. ✅ Documentation is updated (README, developer guide, deployment guide)
8. ✅ Optional improvements are implemented (if user chooses)

---

**Status**: Ready for execution  
**Created**: 2026-05-03  
**Estimated Effort**: 2-3 days (1 day for verification/testing, 1-2 days for optional improvements)
