# Opus Delegation System

TypeScript system for delegating complex architectural problems to Claude Opus 4.5 via use.ai. Automates context extraction, delegation request formatting, artifact parsing, and implementation guide generation.

## Overview

Opus Delegation System bridges gap between Opus 4.5's superior reasoning and lack of repository access. Enables leveraging Opus for challenging design problems (federated learning architectures, PACS integration, property-based test strategies) through structured workflow.

## Features

### Core Components

- **Problem Classifier** - Identifies problems suitable for Opus delegation, categorizes by type
- **Context Extractor** - Extracts relevant code/docs using semantic search and dependency analysis
- **Context Packager** - Formats context as copy-paste ready markdown with size management
- **Template Library** - Pre-built templates for common delegation scenarios
- **Opus Delegator** - Generates delegation requests, manages multi-round sessions
- **Artifact Parser** - Extracts Mermaid diagrams, OpenAPI specs, implementation guides, test strategies
- **Artifact Validator** - Validates completeness, quality, generates follow-up questions
- **Implementation Guide Generator** - Transforms artifacts into actionable implementation steps
- **Session History Manager** - Tracks delegations, enables context reuse
- **Artifact Versioning** - Versions artifacts, compares changes, supports reversion
- **Artifact Exporter** - Exports artifacts as YAML, HTML, markdown, test templates

### Delegation Types

- `architecture_design` - System structure, component relationships, data flow
- `api_design` - Endpoint design, schemas, versioning strategy
- `test_strategy` - Property-based tests, coverage approach
- `integration_design` - External system integration, protocol handling
- `refactoring_analysis` - Code restructuring, dependency untangling
- `formal_verification` - Invariants, correctness properties

### Built-in Templates

- Federated learning architecture
- PACS/DICOM integration design
- Property-based test suite design
- WSI streaming architecture
- Refactoring analysis

## Installation

```bash
npm install
npm run build
```

## Usage

### Basic Workflow

```typescript
import { ProblemClassifier } from './src/components/ProblemClassifier.js';
import { ContextExtractor } from './src/components/ContextExtractor.js';
import { ContextPackager } from './src/components/ContextPackager.js';
import { OpusDelegator } from './src/components/OpusDelegator.js';
import { ArtifactParser } from './src/components/ArtifactParser.js';
import { ArtifactValidator } from './src/components/ArtifactValidator.js';

// 1. Classify problem
const classifier = new ProblemClassifier();
const classification = classifier.classifyProblem(
  'Design federated learning architecture for medical imaging'
);

// 2. Extract context
const extractor = new ContextExtractor('/path/to/repo');
const extractedFiles = extractor.extractContext(
  'federated learning medical imaging',
  classification.classification.delegationType
);

// 3. Package context
const packager = new ContextPackager();
const contextBundle = packager.packageContext(
  extractedFiles,
  'Federated Learning Architecture'
);

// 4. Generate delegation request
const delegator = new OpusDelegator();
const session = delegator.createSession(
  'Federated Learning Architecture',
  'Design federated learning system',
  classification.classification.delegationType,
  'moderate'
);

const request = delegator.generateDelegationRequest(
  session.id,
  'federated_learning_architecture',
  { system_name: 'MedicalFL' },
  contextBundle.markdown
);

// Copy request to use.ai, get Opus response

// 5. Parse Opus response
const parser = new ArtifactParser();
const artifacts = parser.parseResponse(opusResponse, session.id, 1);

// 6. Validate artifacts
const validator = new ArtifactValidator();
const validationResults = validator.validateAll(artifacts);

// 7. Generate implementation guide
const guideGenerator = new ImplementationGuideGenerator();
const guide = guideGenerator.generateGuide(artifacts, 'MedicalFL');
```

### Session Management

```typescript
import { SessionHistoryManager } from './src/components/SessionHistoryManager.js';

const historyManager = new SessionHistoryManager();

// Create session
const session = historyManager.createSession(
  'API Design',
  'Design REST API for patient data',
  'api_design',
  'moderate'
);

// Add round
historyManager.addRound(
  session.id,
  delegationRequest,
  opusResponse,
  artifacts,
  contextSize
);

// Search sessions
const sessions = historyManager.searchSessions({
  problemType: 'api_design',
  keywords: ['patient', 'REST'],
});

// Generate report
const report = historyManager.generateSessionReport(session.id);
```

### Artifact Export

```typescript
import { ArtifactExporter } from './src/components/ArtifactExporter.js';

const exporter = new ArtifactExporter('./exports');

// Export Mermaid diagram
exporter.exportMermaidDiagram(artifact, 'architecture', 'mmd');

// Export OpenAPI spec
exporter.exportOpenAPISpec(artifact, 'api-spec', 'yaml');
exporter.exportOpenAPISpec(artifact, 'api-docs', 'html');

// Export implementation guide
exporter.exportImplementationGuide(guide, 'implementation-guide');

// Export complete package
exporter.exportDelegationPackage(
  sessionId,
  artifacts,
  contextBundle,
  guide
);
```

## Testing

```bash
npm test
```

## Architecture

```
┌─────────────────┐
│ Problem Input   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Problem         │
│ Classifier      │
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌─────────────────┐
│ Context         │────>│ Context         │
│ Extractor       │     │ Packager        │
└─────────────────┘     └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ Opus Delegator  │
                        │ + Templates     │
                        └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ use.ai          │
                        │ (Manual Copy)   │
                        └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ Artifact Parser │
                        └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ Artifact        │
                        │ Validator       │
                        └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ Implementation  │
                        │ Guide Generator │
                        └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ Artifact        │
                        │ Exporter        │
                        └─────────────────┘
```

## Storage Structure

```
.opus-delegation/
├── config.yaml
├── templates/
│   └── custom_template.yaml
└── sessions/
    └── session-{id}/
        ├── session.json
        ├── context.md
        ├── rounds/
        │   ├── 01_request.md
        │   ├── 01_response.md
        │   └── 01_artifacts.json
        └── artifacts/
            ├── architecture.mermaid
            └── api.yaml
```

## Requirements Satisfied

Implements 18 comprehensive requirements:
- ✅ Problem identification and classification
- ✅ Context extraction and packaging
- ✅ Delegation request generation
- ✅ Template library management
- ✅ Artifact reception and parsing
- ✅ Artifact validation and completeness checking
- ✅ Implementation guide generation
- ✅ Session history and context management
- ✅ Multi-round delegation support
- ✅ Artifact export and integration
- ✅ Problem-specific context extraction
- ✅ Artifact quality assessment
- ✅ Artifact versioning and comparison

## License

MIT