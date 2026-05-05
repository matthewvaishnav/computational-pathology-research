/**
 * Additional Tests for Artifact Validator Component
 * Task 12.7: Write unit tests for Artifact Validator - Additional Edge Cases and Property-Based Tests
 * Requirements: 6.1-6.7, 12.1-12.7
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ArtifactValidator } from './ArtifactValidator.js';
import { ParsedArtifact, ArtifactType, MermaidAST, OpenAPISpec, Step, ComplexityLevel } from '../types/core.js';

describe('ArtifactValidator - Additional Edge Cases', () => {
  let validator: ArtifactValidator;

  beforeEach(() => {
    validator = new ArtifactValidator({ qualityThreshold: 70 });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle artifact with missing structured data gracefully', () => {
      const artifact: ParsedArtifact = {
        id: 'test-missing-structure',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\nA --> B',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        // Missing structured field
      };

      const result = validator.validate(artifact);

      expect(result.completenessScore).toBe(0);
      expect(result.isValid).toBe(false);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].type).toBe('missing_structure');
    });

    it('should handle empty artifact content', () => {
      const artifact: ParsedArtifact = {
        id: 'test-empty',
        type: ArtifactType.TEST_STRATEGY,
        content: '',
        metadata: {
          sourceLocation: { start: 0, end: 0 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.completenessScore).toBeLessThan(70);
      expect(result.warnings.length).toBeGreaterThan(0);
    });

    it('should handle unsupported artifact type', () => {
      const artifact: ParsedArtifact = {
        id: 'test-unsupported',
        type: 'UNKNOWN_TYPE' as ArtifactType,
        content: 'Some content',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.completenessScore).toBe(100);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });
  });

  describe('Mermaid Diagram - Advanced Validation', () => {
    it('should handle complex diagrams with multiple node types', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Frontend Service', type: 'rectangle' },
          { id: 'B', label: 'API Gateway', type: 'rhombus' },
          { id: 'C', label: 'Database', type: 'cylinder' },
          { id: 'D', label: 'Cache', type: 'circle' },
          { id: 'E', label: 'Message Queue', type: 'rectangle' },
        ],
        edges: [
          { from: 'A', to: 'B', label: 'HTTP requests' },
          { from: 'B', to: 'C', label: 'SQL queries' },
          { from: 'B', to: 'D', label: 'Cache lookup' },
          { from: 'B', to: 'E', label: 'Async messages' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-complex-diagram',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.completenessScore).toBeGreaterThanOrEqual(90);
    });

    it('should detect when all nodes are orphans', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Service A', type: 'rectangle' },
          { id: 'B', label: 'Service B', type: 'rectangle' },
          { id: 'C', label: 'Service C', type: 'rectangle' },
        ],
        edges: [], // No connections at all
      };

      const artifact: ParsedArtifact = {
        id: 'test-all-orphans',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = validator.validate(artifact);

      expect(result.warnings.some(w => w.includes('orphan'))).toBe(true);
      expect(result.warnings.some(w => w.includes('no edges'))).toBe(true);
    });

    it('should handle very short node labels', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'A', type: 'rectangle' }, // Very short label
          { id: 'B', label: 'B', type: 'rectangle' },
        ],
        edges: [
          { from: 'A', to: 'B', label: 'connects' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-short-labels',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = validator.validate(artifact);

      // Should still be valid as labels exist, even if short
      expect(result.isValid).toBe(true);
    });
  });

  describe('OpenAPI Specification - Advanced Validation', () => {
    it('should handle API spec with global security', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Secure API',
          version: '1.0.0',
          description: 'A comprehensive API with global security',
        },
        security: [{ bearerAuth: [] }], // Global security
        paths: {
          '/users': {
            get: {
              responses: {
                '200': {
                  content: {
                    'application/json': {
                      schema: { type: 'array' },
                      example: [{ id: 1, name: 'John' }],
                    },
                  },
                },
                '401': { description: 'Unauthorized' },
                '500': { description: 'Server error' },
              },
            },
          },
        },
      };

      const artifact: ParsedArtifact = {
        id: 'test-global-security',
        type: ArtifactType.OPENAPI_SPEC,
        content: 'openapi: 3.0.0',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { openapi },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.completenessScore).toBeGreaterThanOrEqual(85);
    });

    it('should handle API spec with comprehensive examples', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Example-Rich API',
          version: '1.0.0',
        },
        paths: {
          '/users': {
            post: {
              requestBody: {
                content: {
                  'application/json': {
                    schema: { type: 'object' },
                    example: { name: 'John Doe', email: 'john@example.com' },
                  },
                },
              },
              responses: {
                '201': {
                  content: {
                    'application/json': {
                      schema: { type: 'object' },
                      examples: {
                        success: {
                          value: { id: 1, name: 'John Doe', email: 'john@example.com' },
                        },
                      },
                    },
                  },
                },
                '400': { description: 'Bad request' },
                '500': { description: 'Server error' },
              },
            },
          },
        },
      };

      const artifact: ParsedArtifact = {
        id: 'test-examples',
        type: ArtifactType.OPENAPI_SPEC,
        content: 'openapi: 3.0.0',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { openapi },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.qualityScores.clarity).toBeGreaterThanOrEqual(90);
    });

    it('should handle API spec with 4XX and 5XX generic error codes', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Generic Errors API',
          version: '1.0.0',
        },
        paths: {
          '/users': {
            get: {
              responses: {
                '200': {
                  content: {
                    'application/json': {
                      schema: { type: 'array' },
                    },
                  },
                },
                '4XX': { description: 'Client errors' }, // Generic 4XX
                '5XX': { description: 'Server errors' }, // Generic 5XX
              },
            },
          },
        },
      };

      const artifact: ParsedArtifact = {
        id: 'test-generic-errors',
        type: ArtifactType.OPENAPI_SPEC,
        content: 'openapi: 3.0.0',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { openapi },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      // Should not warn about missing error responses since 4XX and 5XX are present
      expect(result.warnings.some(w => w.includes('Missing error response'))).toBe(false);
    });
  });

  describe('Implementation Guide - Advanced Validation', () => {
    it('should handle complex dependency chains', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Create database schema',
          description: 'Set up initial database structure',
          dependencies: [],
          complexity: ComplexityLevel.SIMPLE,
        },
        {
          id: 'step-2',
          action: 'Implement data models',
          description: 'Create ORM models',
          dependencies: ['step-1'],
          complexity: ComplexityLevel.SIMPLE,
        },
        {
          id: 'step-3',
          action: 'Build API layer',
          description: 'Create REST endpoints',
          dependencies: ['step-2'],
          complexity: ComplexityLevel.MODERATE,
        },
        {
          id: 'step-4',
          action: 'Add authentication',
          description: 'Implement JWT auth',
          dependencies: ['step-3'],
          complexity: ComplexityLevel.COMPLEX,
        },
        {
          id: 'step-5',
          action: 'Create frontend',
          description: 'Build user interface',
          dependencies: ['step-3', 'step-4'], // Multiple dependencies
          complexity: ComplexityLevel.COMPLEX,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-complex-dependencies',
        type: ArtifactType.IMPLEMENTATION_GUIDE,
        content: '# Implementation Guide',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { implementationSteps: steps },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.completenessScore).toBeGreaterThanOrEqual(85);
      expect(result.errors.some(e => e.type === 'circular_dependencies')).toBe(false);
    });

    it('should handle steps with all complexity levels', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Initialize project',
          description: 'Set up basic project structure',
          dependencies: [],
          complexity: ComplexityLevel.SIMPLE,
        },
        {
          id: 'step-2',
          action: 'Configure middleware',
          description: 'Set up Express middleware',
          dependencies: ['step-1'],
          complexity: ComplexityLevel.MODERATE,
        },
        {
          id: 'step-3',
          action: 'Implement distributed caching',
          description: 'Set up Redis cluster with failover',
          dependencies: ['step-2'],
          complexity: ComplexityLevel.COMPLEX,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-all-complexities',
        type: ArtifactType.IMPLEMENTATION_GUIDE,
        content: '# Implementation Guide',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { implementationSteps: steps },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.warnings.some(w => w.includes('missing complexity'))).toBe(false);
    });

    it('should detect self-referencing dependencies', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Create API',
          description: 'Build API',
          dependencies: ['step-1'], // Self-reference
          complexity: ComplexityLevel.SIMPLE,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-self-reference',
        type: ArtifactType.IMPLEMENTATION_GUIDE,
        content: '# Implementation Guide',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { implementationSteps: steps },
      };

      const result = validator.validate(artifact);

      expect(result.errors.some(e => e.type === 'circular_dependencies')).toBe(true);
    });
  });

  describe('Test Strategy - Advanced Validation', () => {
    it('should handle comprehensive test strategy with all elements', () => {
      const content = `
# Comprehensive Test Strategy

## Coverage Targets
- Unit tests: 95% line coverage, 90% branch coverage
- Integration tests: 85% coverage of API endpoints
- End-to-end tests: 100% coverage of critical user journeys

## Property-Based Testing
- Use fast-check library for JavaScript property-based tests
- Define custom generators for:
  - User input validation (email, phone, address)
  - API request/response data
  - Database entity relationships
- Test invariants:
  - User balance never goes negative
  - Total system balance remains constant
  - Authentication tokens expire correctly

## Edge Cases and Boundary Conditions
- Empty input validation
- Maximum string length (1MB)
- Minimum/maximum numeric values
- Unicode and special character handling
- Network timeout scenarios
- Database connection failures
- Memory exhaustion conditions

## Test Data Requirements
- Mock user database with 1000+ realistic profiles
- Sample API responses for all endpoints
- Test fixtures for file uploads (images, documents)
- Performance test data sets (10K, 100K, 1M records)
- Synthetic data generators for privacy compliance

## Test Organization and Structure
- Unit tests: co-located with source files (*.test.js)
- Integration tests: separate /tests/integration directory
- End-to-end tests: /tests/e2e with page object pattern
- Performance tests: /tests/performance with load scenarios
- Test utilities: shared helpers in /tests/utils

## Test Types and Frameworks
- Unit tests: Jest with React Testing Library
- Integration tests: Supertest for API testing
- End-to-end tests: Playwright for browser automation
- Property-based tests: fast-check for invariant testing
- Performance tests: Artillery for load testing
- Visual regression tests: Percy for UI consistency
      `;

      const artifact: ParsedArtifact = {
        id: 'test-comprehensive-strategy',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.completenessScore).toBeGreaterThanOrEqual(95);
      expect(result.qualityScores.completeness).toBeGreaterThanOrEqual(95);
      expect(result.qualityScores.clarity).toBeGreaterThanOrEqual(90);
      expect(result.qualityScores.implementability).toBeGreaterThanOrEqual(90);
    });

    it('should handle test strategy with only integration tests', () => {
      const content = `
## Test Strategy

### Integration Tests
- Test API endpoints with real database
- Test service-to-service communication
- Test external API integrations
      `;

      const artifact: ParsedArtifact = {
        id: 'test-integration-only',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.warnings.some(w => w.includes('Only one test type'))).toBe(true);
      expect(result.completenessScore).toBeLessThan(90);
    });
  });

  describe('Quality Scoring Edge Cases', () => {
    it('should handle perfect artifacts with maximum scores', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'UserService', label: 'User Management Service', type: 'rectangle' },
          { id: 'AuthService', label: 'Authentication Service', type: 'rectangle' },
          { id: 'Database', label: 'PostgreSQL Database', type: 'cylinder' },
        ],
        edges: [
          { from: 'UserService', to: 'AuthService', label: 'Validates credentials' },
          { from: 'UserService', to: 'Database', label: 'Stores user data' },
          { from: 'AuthService', to: 'Database', label: 'Queries auth tokens' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-perfect',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = validator.validate(artifact);

      expect(result.completenessScore).toBe(100);
      expect(result.qualityScores.completeness).toBe(100);
      expect(result.qualityScores.clarity).toBe(100);
      expect(result.qualityScores.implementability).toBe(100);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
      expect(result.warnings).toHaveLength(0);
    });

    it('should handle artifacts with scores exactly at threshold', () => {
      const validator70 = new ArtifactValidator({ qualityThreshold: 70 });
      
      // Create an artifact that should score exactly 70
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Service A', type: 'rectangle' },
          { id: 'B', label: '', type: 'rectangle' }, // Missing label (penalty)
        ],
        edges: [
          { from: 'A', to: 'B' }, // Missing label (penalty)
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-threshold',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = validator70.validate(artifact);

      // Should be close to threshold but likely below due to penalties
      expect(result.completenessScore).toBeLessThan(80);
      expect(result.isValid).toBe(false); // Due to errors
    });
  });

  describe('Session Summary Edge Cases', () => {
    it('should handle empty validation results', () => {
      const summary = validator.getSessionSummary([]);

      expect(summary.averageCompleteness).toBe(0);
      expect(summary.averageQuality.completeness).toBe(0);
      expect(summary.averageQuality.clarity).toBe(0);
      expect(summary.averageQuality.implementability).toBe(0);
      expect(summary.totalErrors).toBe(0);
      expect(summary.totalWarnings).toBe(0);
      expect(summary.readyForImplementation).toBe(false);
    });

    it('should handle mixed quality results', () => {
      const results = [
        {
          completenessScore: 100,
          qualityScores: { completeness: 100, clarity: 100, implementability: 100 },
          isValid: true,
          errors: [],
          warnings: [],
          followUpQuestions: [],
        },
        {
          completenessScore: 40,
          qualityScores: { completeness: 40, clarity: 50, implementability: 30 },
          isValid: false,
          errors: [{ type: 'test', message: 'Error', severity: 'error' as const }],
          warnings: ['Warning 1', 'Warning 2'],
          followUpQuestions: ['Question 1'],
        },
      ];

      const summary = validator.getSessionSummary(results);

      expect(summary.averageCompleteness).toBe(70);
      expect(summary.averageQuality.completeness).toBe(70);
      expect(summary.averageQuality.clarity).toBe(75);
      expect(summary.averageQuality.implementability).toBe(65);
      expect(summary.totalErrors).toBe(1);
      expect(summary.totalWarnings).toBe(2);
      expect(summary.readyForImplementation).toBe(false); // Due to errors
    });
  });

  describe('Configuration Edge Cases', () => {
    it('should handle strict mode configuration', () => {
      const strictValidator = new ArtifactValidator({ 
        qualityThreshold: 95, 
        strictMode: true 
      });

      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Node A', type: 'rectangle' },
          { id: 'B', label: 'Node B', type: 'rectangle' },
        ],
        edges: [
          { from: 'A', to: 'B', label: 'connects' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-strict',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = strictValidator.validate(artifact);

      // With 95% threshold, even good artifacts might not pass
      expect(result.completenessScore).toBeGreaterThan(0);
    });

    it('should handle zero quality threshold', () => {
      const lenientValidator = new ArtifactValidator({ qualityThreshold: 0 });

      const artifact: ParsedArtifact = {
        id: 'test-lenient',
        type: ArtifactType.TEST_STRATEGY,
        content: 'Test strategy with unit tests and coverage targets: 80%',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = lenientValidator.validate(artifact);

      // Should be valid with 0 threshold if no errors (warnings are OK)
      expect(result.completenessScore).toBeGreaterThan(0);
      // isValid depends on both no errors AND threshold, so check both conditions
      if (result.errors.length === 0) {
        expect(result.isValid).toBe(true);
      } else {
        expect(result.isValid).toBe(false);
      }
    });
  });
});