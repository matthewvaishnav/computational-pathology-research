# SpecWorkflowAdapter Documentation

## Overview

The `SpecWorkflowAdapter` component integrates Opus artifacts with the existing spec workflow by generating `requirements.md`, `design.md`, and `tasks.md` from Opus artifacts. This enables seamless integration between Opus-generated designs and the local development workflow.

## Features

### Requirements.md Generation

- **EARS Pattern Conversion**: Converts Opus requirements to EARS (Easy Approach to Requirements Syntax) patterns
- **Multi-Artifact Support**: Extracts requirements from various artifact types:
  - Implementation guides
  - OpenAPI specifications  
  - Mermaid architecture diagrams
  - Test strategies
  - General content
- **EARS Compliance Validation**: Validates generated requirements against EARS patterns
- **Property-Based Testing Integration**: Includes property-based testing guidance when available

## Usage

### Basic Requirements Generation

```typescript
import { SpecWorkflowAdapter } from 'opus-delegation-system';

const adapter = new SpecWorkflowAdapter();

// Generate requirements from artifacts
const result = await adapter.generateRequirements(artifacts, {
  projectName: 'My Project',
  includePropertyBasedTesting: true,
  earsValidation: true,
  requirementIdPrefix: 'REQ'
});

console.log(result.content); // Generated requirements.md content
console.log(result.metadata.earsCompliant); // EARS compliance status
```

### Configuration Options

```typescript
interface RequirementsGenerationOptions {
  projectName?: string;                    // Default: 'Generated Project'
  includePropertyBasedTesting?: boolean;   // Default: true
  earsValidation?: boolean;                // Default: true
  requirementIdPrefix?: string;            // Default: 'REQ'
}
```

## EARS Pattern Support

The adapter converts requirements to EARS (Easy Approach to Requirements Syntax) patterns:

### Supported EARS Patterns

1. **Basic Requirements**
   ```
   THE <system> SHALL <action>
   ```

2. **Conditional Requirements**
   ```
   WHEN <condition>, THE <system> SHALL <action>
   ```

3. **Alternative Conditions**
   ```
   IF <condition>, THEN THE <system> SHALL <action>
   ```

4. **Contextual Requirements**
   ```
   WHERE <context>, THE <system> SHALL <action>
   ```

### EARS Validation

The adapter validates generated requirements against EARS patterns and reports:
- Non-compliant statements
- Missing user stories
- Empty acceptance criteria

## Artifact Processing

### Implementation Guide Artifacts

Extracts requirements from implementation steps:

```typescript
// Input: Implementation steps
{
  id: 'step-1',
  phase: 'Setup',
  title: 'Initialize Database',
  description: 'Set up database schema and connections',
  action: 'create',
  file: 'database.ts'
}

// Output: EARS requirement
"THE database.ts component SHALL implement initialize database"
```

### OpenAPI Specification Artifacts

Extracts API requirements from endpoints:

```typescript
// Input: OpenAPI paths
{
  "/users": {
    "get": { "summary": "Get users" },
    "post": { "summary": "Create user" }
  }
}

// Output: EARS requirements
"THE API SHALL provide GET /users endpoint to retrieve users data"
"THE API SHALL provide POST /users endpoint to create new users"
```

### Mermaid Architecture Artifacts

Extracts component and relationship requirements:

```typescript
// Input: Mermaid diagram
graph TD
  A[Frontend] --> B[API Gateway]
  B --> C[Database]

// Output: EARS requirements
"THE System SHALL implement Frontend component with defined interfaces"
"THE System SHALL implement API Gateway component with defined interfaces"
"THE Frontend component SHALL send data to API Gateway component"
```

### Test Strategy Artifacts

Extracts testing requirements:

```typescript
// Input: Test strategy content
"Unit tests: 90% coverage for core modules
Integration tests: 80% coverage for API endpoints"

// Output: EARS requirements
"THE System SHALL provide unit tests with 90% coverage for core modules"
"THE System SHALL provide integration tests with 80% coverage for API endpoints"
```

## Property-Based Testing Integration

When `includePropertyBasedTesting` is enabled, the adapter extracts and includes:

- **Invariants**: Properties that must always hold
- **Round-trip properties**: Operations that should be reversible
- **Metamorphic properties**: Relationships between inputs and outputs
- **Error conditions**: Expected failure scenarios

Example output:
```markdown
**Property-Based Testing Guidance:**
- **Invariant**: User ID is always positive
- **Round-trip**: Serialize/deserialize preserves data
- **Metamorphic**: Sorting twice gives same result
```

## Output Format

### Generated Requirements Document Structure

```markdown
# Requirements Document: Project Name

## Introduction

This document specifies requirements for Project Name. The requirements are written using EARS (Easy Approach to Requirements Syntax) patterns to ensure clarity and testability.

## Requirements

### Requirement 1: Component Implementation

**User Story:** As a developer, I want to implement core components, so that the system provides required functionality.

#### Acceptance Criteria

1. THE System SHALL implement authentication component
2. THE System SHALL implement data storage component
3. THE authentication component SHALL validate user credentials

**Property-Based Testing Guidance:**
- **Invariant**: Authentication tokens are always valid
- **Round-trip**: Token generation and validation are consistent
```

### Metadata

```typescript
interface SpecDocument {
  title: string;
  content: string;
  metadata: {
    generatedAt: Date;
    sourceArtifacts: string[];      // IDs of source artifacts
    earsCompliant: boolean;         // Overall EARS compliance
    validationErrors: string[];     // Specific validation errors
  };
}
```

## Error Handling

### Common Issues

1. **Malformed JSON in OpenAPI artifacts**
   - Falls back to content text extraction
   - Continues processing other artifacts

2. **Invalid Mermaid syntax**
   - Attempts pattern matching on content
   - Extracts what components are identifiable

3. **EARS validation failures**
   - Reports specific non-compliant statements
   - Provides suggestions for improvement
   - Sets `earsCompliant: false` in metadata

### Validation Errors

```typescript
// Example validation errors
[
  "Requirement REQ-01: 'System should work' does not follow EARS pattern",
  "Requirement REQ-02: User story is missing or too short",
  "Requirement REQ-03: No acceptance criteria defined"
]
```

## Integration with Spec Workflow

The SpecWorkflowAdapter is designed to integrate with the existing spec workflow:

1. **Opus generates artifacts** → Implementation guides, API specs, diagrams
2. **SpecWorkflowAdapter processes artifacts** → Extracts requirements
3. **Generates requirements.md** → EARS-compliant requirements document
4. **Validates compliance** → Reports issues and suggestions
5. **Integrates with workflow** → Ready for spec task execution

## Best Practices

### For Optimal Results

1. **Provide detailed artifacts**: More detailed Opus artifacts produce better requirements
2. **Use consistent naming**: Consistent component and API naming improves extraction
3. **Include descriptions**: Rich descriptions in artifacts improve requirement quality
4. **Validate early**: Run EARS validation to catch issues before implementation

### Artifact Quality Guidelines

1. **Implementation Guides**: Include clear action verbs and component names
2. **API Specifications**: Provide complete endpoint definitions with descriptions
3. **Architecture Diagrams**: Use descriptive node labels and relationship annotations
4. **Test Strategies**: Specify coverage targets and test types explicitly

## Examples

See `examples/requirements-generation-example.ts` for complete usage examples including:
- Multi-artifact processing
- EARS validation demonstration
- Property-based testing integration
- Error handling scenarios

## Requirements Traceability

This component implements:
- **Requirement 16.1**: Generate requirements.md from Opus-provided requirements using EARS patterns
- **Requirement 16.4**: Validate generated spec documents against project standards (EARS compliance)