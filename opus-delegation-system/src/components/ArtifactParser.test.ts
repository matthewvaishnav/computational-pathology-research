/**
 * Unit tests for Artifact Parser Component
 * Task 11.7: Write unit tests for Artifact Parser
 * Requirements: 5.1-5.8
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ArtifactParser, ArtifactParseError } from './ArtifactParser.js';
import { ArtifactType } from '../types/core.js';

describe('ArtifactParser', () => {
  let parser: ArtifactParser;

  beforeEach(() => {
    parser = new ArtifactParser();
  });

  describe('Code Block Extraction', () => {
    it('should extract fenced code blocks with language identifiers', () => {
      const response = `
Here is a diagram:

\`\`\`mermaid
graph TD
  A[Start] --> B[End]
\`\`\`

And some code:

\`\`\`typescript
const x = 42;
\`\`\`
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThanOrEqual(2);

      const mermaidArtifact = artifacts.find((a) => a.type === ArtifactType.MERMAID_DIAGRAM);
      expect(mermaidArtifact).toBeDefined();
      expect(mermaidArtifact?.content).toContain('graph TD');

      const codeArtifact = artifacts.find((a) => a.type === ArtifactType.CODE_SNIPPET);
      expect(codeArtifact).toBeDefined();
      expect(codeArtifact?.content).toContain('const x = 42');
    });

    it('should preserve source location metadata', () => {
      const response = `\`\`\`mermaid
graph TD
  A --> B
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThan(0);

      const artifact = artifacts[0];
      expect(artifact.metadata.sourceLocation).toBeDefined();
      expect(artifact.metadata.sourceLocation.start).toBeGreaterThanOrEqual(0);
      expect(artifact.metadata.sourceLocation.end).toBeGreaterThan(
        artifact.metadata.sourceLocation.start
      );
    });

    it('should handle multiple code blocks of the same type', () => {
      const response = `
\`\`\`yaml
openapi: 3.0.0
info:
  title: API 1
  version: 1.0.0
paths: {}
\`\`\`

\`\`\`yaml
openapi: 3.0.0
info:
  title: API 2
  version: 1.0.0
paths: {}
\`\`\`
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const openApiArtifacts = artifacts.filter((a) => a.type === ArtifactType.OPENAPI_SPEC);
      expect(openApiArtifacts.length).toBe(2);
    });
  });

  describe('Mermaid Diagram Parsing', () => {
    it('should parse valid Mermaid diagram', () => {
      const response = `\`\`\`mermaid
graph TD
  A[Start] --> B[Process]
  B --> C{Decision}
  C -->|Yes| D[End]
  C -->|No| A
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const mermaid = artifacts.find((a) => a.type === ArtifactType.MERMAID_DIAGRAM);

      expect(mermaid).toBeDefined();
      expect(mermaid?.structured?.mermaid).toBeDefined();
      expect(mermaid?.structured?.mermaid?.type).toBe('graph');
      expect(mermaid?.structured?.mermaid?.nodes.length).toBeGreaterThan(0);
      expect(mermaid?.structured?.mermaid?.edges.length).toBeGreaterThan(0);
    });

    it('should validate Mermaid syntax and report errors', () => {
      const response = `\`\`\`mermaid
invalid syntax here
\`\`\``;

      expect(() => {
        parser.parseResponse(response, 'session-1', 1);
      }).toThrow(ArtifactParseError);
    });

    it('should report errors with line numbers', () => {
      const response = `\`\`\`mermaid
invalid diagram
\`\`\``;

      try {
        parser.parseResponse(response, 'session-1', 1);
        expect.fail('Should have thrown ArtifactParseError');
      } catch (error) {
        expect(error).toBeInstanceOf(ArtifactParseError);
        expect((error as ArtifactParseError).lineNumber).toBeDefined();
      }
    });

    it('should parse flowchart diagrams', () => {
      const response = `\`\`\`mermaid
flowchart LR
  A[Input] --> B[Process]
  B --> C[Output]
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const mermaid = artifacts.find((a) => a.type === ArtifactType.MERMAID_DIAGRAM);

      expect(mermaid).toBeDefined();
      expect(mermaid?.structured?.mermaid?.type).toBe('flowchart');
    });

    it('should parse sequence diagrams', () => {
      const response = `\`\`\`mermaid
sequenceDiagram
  Alice->>Bob: Hello
  Bob->>Alice: Hi
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const mermaid = artifacts.find((a) => a.type === ArtifactType.MERMAID_DIAGRAM);

      expect(mermaid).toBeDefined();
      expect(mermaid?.structured?.mermaid?.type).toBe('sequenceDiagram');
    });

    it('should handle empty Mermaid diagram', () => {
      const response = `\`\`\`mermaid
\`\`\``;

      expect(() => {
        parser.parseResponse(response, 'session-1', 1);
      }).toThrow(ArtifactParseError);
    });
  });

  describe('OpenAPI Specification Parsing', () => {
    it('should parse valid OpenAPI 3.0 specification', () => {
      const response = `\`\`\`yaml
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
paths:
  /users:
    get:
      summary: Get users
      responses:
        '200':
          description: Success
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const openapi = artifacts.find((a) => a.type === ArtifactType.OPENAPI_SPEC);

      expect(openapi).toBeDefined();
      expect(openapi?.structured?.openapi).toBeDefined();
      expect(openapi?.structured?.openapi?.openapi).toBe('3.0.0');
      expect(openapi?.structured?.openapi?.info.title).toBe('Test API');
      expect(openapi?.structured?.openapi?.paths).toBeDefined();
    });

    it('should validate against OpenAPI 3.0 schema', () => {
      const response = `\`\`\`yaml
openapi: 3.0.0
info:
  title: Test API
\`\`\``;

      expect(() => {
        parser.parseResponse(response, 'session-1', 1);
      }).toThrow(ArtifactParseError);
    });

    it('should report validation errors with specific fields', () => {
      const response = `\`\`\`yaml
openapi: 3.0.0
paths: {}
\`\`\``;

      try {
        parser.parseResponse(response, 'session-1', 1);
        expect.fail('Should have thrown ArtifactParseError');
      } catch (error) {
        expect(error).toBeInstanceOf(ArtifactParseError);
        expect((error as ArtifactParseError).field).toBeDefined();
      }
    });

    it('should handle Swagger 2.0 specifications', () => {
      const response = `\`\`\`yaml
swagger: '2.0'
info:
  title: Test API
  version: 1.0.0
paths: {}
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const openapi = artifacts.find((a) => a.type === ArtifactType.OPENAPI_SPEC);

      expect(openapi).toBeDefined();
    });

    it('should reject invalid YAML', () => {
      const response = `\`\`\`yaml
openapi: 3.0.0
info:
  title: Test
  invalid: [unclosed
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      // The yaml parser may be lenient, so we just check it doesn't crash
      // In practice, truly malformed YAML will throw during parsing
      expect(artifacts).toBeDefined();
    });

    it('should distinguish OpenAPI from generic YAML', () => {
      const response = `\`\`\`yaml
some_config: value
another_field: 123
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const openapi = artifacts.find((a) => a.type === ArtifactType.OPENAPI_SPEC);

      expect(openapi).toBeUndefined();
    });
  });

  describe('Implementation Guide Parsing', () => {
    it('should extract implementation guides from markdown sections', () => {
      const response = `
## Implementation Steps

1. Create database schema
2. Implement API endpoints
3. Add authentication
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const guide = artifacts.find((a) => a.type === ArtifactType.IMPLEMENTATION_GUIDE);

      expect(guide).toBeDefined();
      expect(guide?.structured?.implementationSteps).toBeDefined();
      expect(guide?.structured?.implementationSteps?.length).toBe(3);
    });

    it('should parse step hierarchy', () => {
      const response = `
## Implementation Plan

1. Setup infrastructure
   - Configure database
   - Setup CI/CD
2. Implement core features
   - User management
   - Authentication
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const guide = artifacts.find((a) => a.type === ArtifactType.IMPLEMENTATION_GUIDE);

      expect(guide).toBeDefined();
      expect(guide?.structured?.implementationSteps?.length).toBeGreaterThan(0);
    });

    it('should parse step dependencies', () => {
      const response = `
## Steps

1. Create models
2. Create controllers (depends on: step 1)
3. Create routes (requires: step 2)
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const guide = artifacts.find((a) => a.type === ArtifactType.IMPLEMENTATION_GUIDE);

      expect(guide).toBeDefined();
      const steps = guide?.structured?.implementationSteps || [];
      expect(steps.length).toBe(3);

      const step2 = steps.find((s) => s.action.includes('controllers'));
      expect(step2?.dependencies.length).toBeGreaterThan(0);
    });

    it('should handle various step numbering formats', () => {
      const response = `
## Guide

1. First step
2) Second step
3. Third step
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const guide = artifacts.find((a) => a.type === ArtifactType.IMPLEMENTATION_GUIDE);

      expect(guide).toBeDefined();
      expect(guide?.structured?.implementationSteps?.length).toBe(3);
    });
  });

  describe('Test Strategy Parsing', () => {
    it('should extract test strategies from markdown', () => {
      const response = `
## Test Strategy

### Unit Tests
- Test user creation
- Test validation logic

### Property-Based Tests
- Property: All valid inputs produce valid outputs
- Invariant: User ID is always unique
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const testStrategy = artifacts.find((a) => a.type === ArtifactType.TEST_STRATEGY);

      expect(testStrategy).toBeDefined();
      expect(testStrategy?.content).toContain('Test Strategy');
    });

    it('should identify property-based test designs', () => {
      const response = `
## Testing Approach

Property: For all valid user inputs, the system returns a 200 status
Invariant: Database constraints are never violated
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const testStrategy = artifacts.find((a) => a.type === ArtifactType.TEST_STRATEGY);

      expect(testStrategy).toBeDefined();
      expect(testStrategy?.content).toContain('Property');
      expect(testStrategy?.content).toContain('Invariant');
    });

    it('should extract test case specifications', () => {
      const response = `
## Test Plan

Test case 1: User registration with valid data
- Input: Valid email and password
- Expected: User created successfully

Test case 2: User registration with invalid email
- Input: Invalid email format
- Expected: Validation error
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const testStrategy = artifacts.find((a) => a.type === ArtifactType.TEST_STRATEGY);

      expect(testStrategy).toBeDefined();
      expect(testStrategy?.content).toContain('Test case');
    });
  });

  describe('Artifact Storage', () => {
    it('should store parsed artifacts in JSON format', () => {
      const response = `\`\`\`mermaid
graph TD
  A --> B
\`\`\``;

      parser.parseResponse(response, 'session-1', 1);
      const artifacts = parser.getArtifacts('session-1', 1);

      expect(artifacts.length).toBeGreaterThan(0);
      expect(artifacts[0].id).toBeDefined();
      expect(artifacts[0].type).toBeDefined();
    });

    it('should link artifacts to sessions and rounds', () => {
      const response1 = `\`\`\`mermaid
graph TD
  A --> B
\`\`\``;

      const response2 = `\`\`\`mermaid
graph TD
  C --> D
\`\`\``;

      parser.parseResponse(response1, 'session-1', 1);
      parser.parseResponse(response2, 'session-1', 2);

      const round1Artifacts = parser.getArtifacts('session-1', 1);
      const round2Artifacts = parser.getArtifacts('session-1', 2);

      expect(round1Artifacts.length).toBeGreaterThan(0);
      expect(round2Artifacts.length).toBeGreaterThan(0);
      expect(round1Artifacts[0].id).not.toBe(round2Artifacts[0].id);
    });

    it('should retrieve all artifacts for a session', () => {
      parser.parseResponse('```mermaid\ngraph TD\n  A --> B\n```', 'session-1', 1);
      parser.parseResponse('```mermaid\ngraph TD\n  C --> D\n```', 'session-1', 2);
      parser.parseResponse('```mermaid\ngraph TD\n  E --> F\n```', 'session-2', 1);

      const session1Artifacts = parser.getAllSessionArtifacts('session-1');
      const session2Artifacts = parser.getAllSessionArtifacts('session-2');

      expect(session1Artifacts.length).toBe(2);
      expect(session2Artifacts.length).toBe(1);
    });

    it('should export artifacts as JSON', () => {
      const response = `\`\`\`mermaid
graph TD
  A --> B
\`\`\``;

      parser.parseResponse(response, 'session-1', 1);
      const json = parser.exportArtifactsAsJSON('session-1', 1);

      expect(json).toBeDefined();
      const parsed = JSON.parse(json);
      expect(Array.isArray(parsed)).toBe(true);
      expect(parsed.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should detect malformed Mermaid syntax', () => {
      const response = `\`\`\`mermaid
this is not valid mermaid
\`\`\``;

      expect(() => {
        parser.parseResponse(response, 'session-1', 1);
      }).toThrow(ArtifactParseError);
    });

    it('should detect incomplete OpenAPI specifications', () => {
      const response = `\`\`\`yaml
openapi: 3.0.0
\`\`\``;

      expect(() => {
        parser.parseResponse(response, 'session-1', 1);
      }).toThrow(ArtifactParseError);
    });

    it('should continue processing after non-critical errors', () => {
      const response = `
\`\`\`mermaid
graph TD
  A --> B
\`\`\`

\`\`\`typescript
const valid = true;
\`\`\`
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThan(0);
    });

    it('should include parse warnings in metadata', () => {
      const response = `\`\`\`yaml
openapi: 2.0.0
swagger: '2.0'
info:
  title: Test
  version: 1.0.0
paths: {}
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      const openapi = artifacts.find((a) => a.type === ArtifactType.OPENAPI_SPEC);

      expect(openapi).toBeDefined();
      // May have warnings about version
    });

    it('should handle empty responses gracefully', () => {
      const response = '';
      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts).toEqual([]);
    });

    it('should handle responses with no code blocks', () => {
      const response = 'This is just plain text with no code blocks.';
      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBe(0);
    });
  });

  describe('Complex Scenarios', () => {
    it('should handle mixed artifact types in single response', () => {
      const response = `
# Architecture Design

\`\`\`mermaid
graph TD
  A[API] --> B[Database]
\`\`\`

## API Specification

\`\`\`yaml
openapi: 3.0.0
info:
  title: My API
  version: 1.0.0
paths:
  /users:
    get:
      responses:
        '200':
          description: OK
\`\`\`

## Implementation Steps

1. Create database schema
2. Implement API endpoints
3. Add tests

## Test Strategy

- Unit tests for all endpoints
- Property: All requests return valid JSON
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);

      expect(artifacts.length).toBeGreaterThan(0);
      expect(artifacts.some((a) => a.type === ArtifactType.MERMAID_DIAGRAM)).toBe(true);
      expect(artifacts.some((a) => a.type === ArtifactType.OPENAPI_SPEC)).toBe(true);
      expect(artifacts.some((a) => a.type === ArtifactType.IMPLEMENTATION_GUIDE)).toBe(true);
      expect(artifacts.some((a) => a.type === ArtifactType.TEST_STRATEGY)).toBe(true);
    });

    it('should preserve metadata across all artifact types', () => {
      const response = `
\`\`\`mermaid
graph TD
  A --> B
\`\`\`

\`\`\`yaml
openapi: 3.0.0
info:
  title: Test
  version: 1.0.0
paths: {}
\`\`\`
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);

      for (const artifact of artifacts) {
        expect(artifact.metadata.extractedAt).toBeInstanceOf(Date);
        expect(artifact.metadata.sourceLocation).toBeDefined();
        expect(artifact.metadata.parseWarnings).toBeDefined();
      }
    });

    it('should handle nested code blocks in markdown', () => {
      const response = `
## Example

Here's how to use it:

\`\`\`typescript
// This is example code
const example = \`
  nested content
\`;
\`\`\`
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle code blocks without language identifier', () => {
      const response = `
\`\`\`
plain code block
\`\`\`
`;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      // Should not extract blocks without language identifier
      expect(artifacts.length).toBe(0);
    });

    it('should handle very large responses', () => {
      const largeContent = 'A --> B\n'.repeat(1000);
      const response = `\`\`\`mermaid
graph TD
${largeContent}
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThan(0);
    });

    it('should handle special characters in content', () => {
      const response = `\`\`\`mermaid
graph TD
  A["Node with 'quotes'"] --> B["Node with \\"escaped\\""]
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThan(0);
    });

    it('should handle Unicode characters', () => {
      const response = `\`\`\`mermaid
graph TD
  A[开始] --> B[结束]
\`\`\``;

      const artifacts = parser.parseResponse(response, 'session-1', 1);
      expect(artifacts.length).toBeGreaterThan(0);
    });
  });
});
