/**
 * Tests for Artifact Validator Component
 * Task 12.7: Write unit tests for Artifact Validator
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ArtifactValidator } from './ArtifactValidator.js';
import { ParsedArtifact, ArtifactType, MermaidAST, OpenAPISpec, Step, ComplexityLevel } from '../types/core.js';

describe('ArtifactValidator', () => {
  let validator: ArtifactValidator;

  beforeEach(() => {
    validator = new ArtifactValidator({ qualityThreshold: 70 });
  });

  describe('Mermaid Diagram Validation', () => {
    it('should validate complete diagram with all labels', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Frontend', type: 'rectangle' },
          { id: 'B', label: 'Backend API', type: 'rectangle' },
          { id: 'C', label: 'Database', type: 'rectangle' },
        ],
        edges: [
          { from: 'A', to: 'B', label: 'HTTP requests' },
          { from: 'B', to: 'C', label: 'SQL queries' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\nA[Frontend] --> B[Backend API]',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: { mermaid },
      };

      const result = validator.validate(artifact);

      expect(result.isValid).toBe(true);
      expect(result.completenessScore).toBeGreaterThanOrEqual(70);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect nodes without labels', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Frontend', type: 'rectangle' },
          { id: 'B', label: '', type: 'rectangle' }, // Missing label
          { id: 'C', label: 'Database', type: 'rectangle' },
        ],
        edges: [
          { from: 'A', to: 'B', label: 'connects' },
          { from: 'B', to: 'C', label: 'queries' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-2',
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

      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors[0].type).toBe('missing_node_labels');
      expect(result.followUpQuestions.length).toBeGreaterThan(0);
    });

    it('should detect edges without labels', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Frontend', type: 'rectangle' },
          { id: 'B', label: 'Backend', type: 'rectangle' },
        ],
        edges: [
          { from: 'A', to: 'B' }, // Missing label
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-3',
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

      expect(result.warnings.length).toBeGreaterThan(0);
      expect(result.warnings[0]).toContain('edge(s) missing labels');
    });

    it('should detect orphan nodes', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Frontend', type: 'rectangle' },
          { id: 'B', label: 'Backend', type: 'rectangle' },
          { id: 'C', label: 'Orphan', type: 'rectangle' }, // Orphan node
        ],
        edges: [
          { from: 'A', to: 'B', label: 'connects' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-4',
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

      expect(result.warnings.some((w) => w.includes('orphan'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('not connected'))).toBe(true);
    });

    it('should detect naming inconsistencies', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'UserService', label: 'User Service', type: 'rectangle' },
          { id: 'user_service', label: 'User Service Alt', type: 'rectangle' },
          { id: 'Database', label: 'DB', type: 'rectangle' },
        ],
        edges: [
          { from: 'UserService', to: 'Database', label: 'queries' },
          { from: 'user_service', to: 'Database', label: 'queries' },
        ],
      };

      const artifact: ParsedArtifact = {
        id: 'test-5',
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

      expect(result.warnings.some((w) => w.includes('Inconsistent naming'))).toBe(true);
    });

    it('should warn about minimal diagrams', () => {
      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [{ id: 'A', label: 'Single Node', type: 'rectangle' }],
        edges: [],
      };

      const artifact: ParsedArtifact = {
        id: 'test-6',
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

      expect(result.warnings.some((w) => w.includes('fewer than 2 nodes'))).toBe(true);
    });
  });

  describe('OpenAPI Specification Validation', () => {
    it('should validate complete OpenAPI spec', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Test API',
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
                '400': { description: 'Bad request' },
                '500': { description: 'Server error' },
              },
            },
            post: {
              requestBody: {
                content: {
                  'application/json': {
                    schema: { type: 'object' },
                  },
                },
              },
              responses: {
                '201': {
                  content: {
                    'application/json': {
                      schema: { type: 'object' },
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
        id: 'test-7',
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
      expect(result.completenessScore).toBeGreaterThanOrEqual(70);
    });

    it('should detect missing endpoints', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Test API',
          version: '1.0.0',
        },
        paths: {},
      };

      const artifact: ParsedArtifact = {
        id: 'test-8',
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

      expect(result.errors.some((e) => e.type === 'missing_endpoints')).toBe(true);
      expect(result.followUpQuestions.length).toBeGreaterThan(0);
    });

    it('should detect missing request schemas', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Test API',
          version: '1.0.0',
        },
        paths: {
          '/users': {
            post: {
              // Missing requestBody
              responses: {
                '201': {
                  content: {
                    'application/json': {
                      schema: { type: 'object' },
                    },
                  },
                },
              },
            },
          },
        },
      };

      const artifact: ParsedArtifact = {
        id: 'test-9',
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

      expect(result.warnings.some((w) => w.includes('Missing request body schema'))).toBe(true);
    });

    it('should detect missing error responses', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Test API',
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
                // Missing 400 and 500
              },
            },
          },
        },
      };

      const artifact: ParsedArtifact = {
        id: 'test-10',
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

      expect(result.warnings.some((w) => w.includes('Missing error response definitions'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('error responses'))).toBe(true);
    });

    it('should warn about missing authentication', () => {
      const openapi: OpenAPISpec = {
        openapi: '3.0.0',
        info: {
          title: 'Test API',
          version: '1.0.0',
        },
        paths: {
          '/users': {
            get: {
              // No security defined
              responses: {
                '200': {
                  content: {
                    'application/json': {
                      schema: { type: 'array' },
                    },
                  },
                },
              },
            },
          },
        },
      };

      const artifact: ParsedArtifact = {
        id: 'test-11',
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

      expect(result.warnings.some((w) => w.includes('missing authentication'))).toBe(true);
    });
  });

  describe('Implementation Guide Validation', () => {
    it('should validate complete implementation guide', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Create database schema',
          description: 'Define tables and relationships',
          dependencies: [],
          complexity: ComplexityLevel.SIMPLE,
        },
        {
          id: 'step-2',
          action: 'Implement API endpoints',
          description: 'Create REST API handlers',
          dependencies: ['step-1'],
          complexity: ComplexityLevel.MODERATE,
        },
        {
          id: 'step-3',
          action: 'Add authentication',
          description: 'Implement JWT-based auth',
          dependencies: ['step-2'],
          complexity: ComplexityLevel.COMPLEX,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-12',
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
      expect(result.completenessScore).toBeGreaterThanOrEqual(70);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect missing action verbs', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Database schema for users', // No action verb
          description: 'Tables and relationships',
          dependencies: [],
          complexity: ComplexityLevel.SIMPLE,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-13',
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

      expect(result.warnings.some((w) => w.includes('missing clear action verbs'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('action verbs'))).toBe(true);
    });

    it('should detect circular dependencies', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Create API',
          description: 'Build API',
          dependencies: ['step-2'],
          complexity: ComplexityLevel.SIMPLE,
        },
        {
          id: 'step-2',
          action: 'Create database',
          description: 'Setup DB',
          dependencies: ['step-1'], // Circular!
          complexity: ComplexityLevel.SIMPLE,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-14',
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

      expect(result.errors.some((e) => e.type === 'circular_dependencies')).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('circular dependencies'))).toBe(true);
    });

    it('should warn about missing descriptions', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Create API',
          description: '', // Missing description
          dependencies: [],
          complexity: ComplexityLevel.SIMPLE,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-15',
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

      expect(result.warnings.some((w) => w.includes('insufficient descriptions'))).toBe(true);
    });

    it('should warn about too few steps', () => {
      const steps: Step[] = [
        {
          id: 'step-1',
          action: 'Create everything',
          description: 'Build the entire system',
          dependencies: [],
          complexity: ComplexityLevel.COMPLEX,
        },
      ];

      const artifact: ParsedArtifact = {
        id: 'test-16',
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

      expect(result.warnings.some((w) => w.includes('fewer than 3 steps'))).toBe(true);
    });
  });

  describe('Test Strategy Validation', () => {
    it('should validate complete test strategy', () => {
      const content = `
## Test Strategy

### Coverage Targets
- Unit tests: 90% coverage
- Integration tests: 80% coverage

### Property-Based Testing
- Use Hypothesis for property-based tests
- Define generators for user input
- Test invariants: balance never negative

### Edge Cases
- Empty input
- Maximum values
- Boundary conditions

### Test Data
- Mock user database
- Sample fixtures for API responses
- Test data generators

### Test Types
- Unit tests for business logic
- Integration tests for API endpoints
- End-to-end tests for critical workflows
      `;

      const artifact: ParsedArtifact = {
        id: 'test-17',
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
      expect(result.completenessScore).toBeGreaterThanOrEqual(70);
    });

    it('should detect missing coverage targets', () => {
      const content = `
## Test Strategy

We will write tests for the system.
      `;

      const artifact: ParsedArtifact = {
        id: 'test-18',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.warnings.some((w) => w.includes('coverage targets'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('coverage percentages'))).toBe(true);
    });

    it('should detect missing property-based tests', () => {
      const content = `
## Test Strategy

### Unit Tests
- Test user creation
- Test validation logic
      `;

      const artifact: ParsedArtifact = {
        id: 'test-19',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.warnings.some((w) => w.includes('property-based'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('invariants'))).toBe(true);
    });

    it('should detect missing generators for property tests', () => {
      const content = `
## Test Strategy

### Property-Based Testing
- Test invariants
- Verify properties hold
      `;

      const artifact: ParsedArtifact = {
        id: 'test-20',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.warnings.some((w) => w.includes('generators'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('generators'))).toBe(true);
    });

    it('should detect missing edge cases', () => {
      const content = `
## Test Strategy

### Unit Tests
- Test normal cases
      `;

      const artifact: ParsedArtifact = {
        id: 'test-21',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.warnings.some((w) => w.includes('edge cases'))).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('edge cases'))).toBe(true);
    });

    it('should detect missing test types', () => {
      const content = `
## Test Strategy

We will test the system thoroughly.
      `;

      const artifact: ParsedArtifact = {
        id: 'test-22',
        type: ArtifactType.TEST_STRATEGY,
        content,
        metadata: {
          sourceLocation: { start: 0, end: content.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = validator.validate(artifact);

      expect(result.errors.some((e) => e.type === 'missing_test_types')).toBe(true);
      expect(result.followUpQuestions.some((q) => q.includes('types of tests'))).toBe(true);
    });
  });

  describe('Batch Validation', () => {
    it('should validate multiple artifacts', () => {
      const artifacts: ParsedArtifact[] = [
        {
          id: 'test-23',
          type: ArtifactType.MERMAID_DIAGRAM,
          content: 'graph TD',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date(),
          },
          structured: {
            mermaid: {
              type: 'graph',
              nodes: [
                { id: 'A', label: 'Node A', type: 'rectangle' },
                { id: 'B', label: 'Node B', type: 'rectangle' },
              ],
              edges: [{ from: 'A', to: 'B', label: 'connects' }],
            },
          },
        },
        {
          id: 'test-24',
          type: ArtifactType.TEST_STRATEGY,
          content: 'Test strategy with unit tests',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date(),
          },
        },
      ];

      const results = validator.validateAll(artifacts);

      expect(results).toHaveLength(2);
      expect(results[0].completenessScore).toBeGreaterThan(0);
      expect(results[1].completenessScore).toBeGreaterThan(0);
    });

    it('should generate session summary', () => {
      const results = [
        {
          completenessScore: 85,
          qualityScores: { completeness: 85, clarity: 90, implementability: 80 },
          isValid: true,
          errors: [],
          warnings: [],
          followUpQuestions: [],
        },
        {
          completenessScore: 75,
          qualityScores: { completeness: 75, clarity: 80, implementability: 70 },
          isValid: true,
          errors: [],
          warnings: ['Minor issue'],
          followUpQuestions: [],
        },
      ];

      const summary = validator.getSessionSummary(results);

      expect(summary.averageCompleteness).toBe(80);
      expect(summary.averageQuality.completeness).toBe(80);
      expect(summary.averageQuality.clarity).toBe(85);
      expect(summary.averageQuality.implementability).toBe(75);
      expect(summary.totalErrors).toBe(0);
      expect(summary.totalWarnings).toBe(1);
      expect(summary.readyForImplementation).toBe(true);
    });

    it('should detect not ready for implementation', () => {
      const results = [
        {
          completenessScore: 50,
          qualityScores: { completeness: 50, clarity: 60, implementability: 40 },
          isValid: false,
          errors: [{ type: 'test', message: 'Error', severity: 'error' as const }],
          warnings: [],
          followUpQuestions: [],
        },
      ];

      const summary = validator.getSessionSummary(results);

      expect(summary.readyForImplementation).toBe(false);
      expect(summary.totalErrors).toBe(1);
    });
  });

  describe('Configuration', () => {
    it('should respect custom quality threshold', () => {
      const strictValidator = new ArtifactValidator({ qualityThreshold: 90 });

      const mermaid: MermaidAST = {
        type: 'graph',
        nodes: [
          { id: 'A', label: 'Node A', type: 'rectangle' },
          { id: 'B', label: 'Node B', type: 'rectangle' },
        ],
        edges: [{ from: 'A', to: 'B' }], // Missing label
      };

      const artifact: ParsedArtifact = {
        id: 'test-25',
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

      // With 90% threshold, this should fail
      expect(result.isValid).toBe(false);
    });
  });
});
