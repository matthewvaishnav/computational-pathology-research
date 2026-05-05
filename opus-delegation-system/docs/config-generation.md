# .config.kiro Generation

The SpecWorkflowAdapter component now supports generating `.config.kiro` files with appropriate workflow type and spec metadata, fulfilling Requirement 16.5.

## Overview

The `.config.kiro` file is a JSON configuration file that contains:
- `specId`: A unique UUID for the specification
- `workflowType`: The workflow type (`requirements-first`, `design-first`, or `bugfix`)
- `specType`: The type of specification (`feature` or `bugfix`)

## Usage

### Basic Usage

```typescript
import { SpecWorkflowAdapter } from './components/SpecWorkflowAdapter.js';

const adapter = new SpecWorkflowAdapter();

// Generate config from artifacts
const config = await adapter.generateConfig(artifacts);

// Export as JSON string for .config.kiro file
const configContent = adapter.exportConfigKiro(config);
```

### Custom Options

```typescript
const config = await adapter.generateConfig(artifacts, {
  specId: 'custom-spec-id',
  workflowType: 'design-first',
  specType: 'feature',
  projectName: 'My Project'
});
```

## Automatic Workflow Detection

The generator automatically detects the appropriate workflow type based on artifact content:

### Design-First Detection
- **Architecture Diagrams**: Mermaid diagrams with complex component structures
- **API Specifications**: OpenAPI specs with multiple endpoints
- **Detailed Implementation Guides**: Guides with 5+ implementation steps

### Bugfix Detection
- **Bug Keywords**: Content containing multiple bug-related terms (bug, fix, error, issue, problem, defect, failure, timeout, crash, exception, incorrect, broken, failing)
- **Specific Patterns**: "bug fix", "error fix", "issue fix", "timeout fix", "failure fix"

### Requirements-First (Default)
- All other content defaults to requirements-first workflow

## Spec Type Detection

- **Bugfix**: Detected when bugfix workflow is identified
- **Feature**: Default for all other workflows

## File Format

The exported `.config.kiro` file contains only the essential configuration:

```json
{
  "specId": "a9c00ef1-c181-45e7-943c-ac669f6dedd0",
  "workflowType": "design-first", 
  "specType": "feature"
}
```

Metadata (generation date, source artifacts, project name) is available in the full `ConfigKiro` object but excluded from the file export to match existing format.

## Integration with Spec Workflow

This generator integrates with the existing spec workflow components:
- **Task 19.1**: Requirements.md generator
- **Task 19.2**: Design.md generator  
- **Task 19.3**: Tasks.md generator
- **Task 19.4**: .config.kiro generator (this component)

Together, these components enable complete spec workflow integration for Opus-generated artifacts.

## Examples

See `examples/config-generation-example.ts` for complete usage examples demonstrating:
- Design-first workflow detection from architecture diagrams
- Bugfix workflow detection from bug-related content
- Requirements-first workflow for general content
- OpenAPI specification handling
- Custom configuration options