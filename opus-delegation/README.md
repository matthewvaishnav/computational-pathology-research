# Opus Delegation System

A TypeScript-based CLI tool for delegating complex architectural and design problems to Claude Opus 4.5 via use.ai. The system automates context extraction from your codebase, formats delegation requests, parses Opus-generated artifacts, and generates implementation guides.

## Overview

The Opus Delegation System bridges the gap between Claude Opus 4.5's superior reasoning capabilities and its lack of repository access. It enables you to:

- **Extract relevant context** from your codebase automatically
- **Package problems** with structured delegation requests
- **Parse artifacts** from Opus (architecture diagrams, API specs, implementation plans)
- **Generate implementation guides** from Opus designs
- **Track multi-round conversations** with session history

## Features

### Core Capabilities
- Problem identification and classification for delegation suitability
- Automated context extraction and packaging from codebase
- Structured delegation request generation with templates
- Artifact reception, parsing, and validation
- Implementation guide generation from artifacts
- Session history and multi-round delegation support

### Advanced Features
- Problem-specific context extraction strategies
- Artifact quality assessment and completeness checking
- Workflow automation with CLI interface
- Context size optimization for Opus limits
- Artifact versioning and comparison
- Integration with existing spec workflow
- Cost tracking and efficiency metrics

## Project Structure

```
opus-delegation/
├── src/
│   ├── types/           # Core type definitions
│   ├── components/      # Main components (to be implemented)
│   ├── cli/             # CLI interface (to be implemented)
│   └── index.ts         # Main entry point
├── dist/                # Compiled output
├── tests/               # Test files
├── package.json         # Project dependencies
├── tsconfig.json        # TypeScript configuration
└── README.md            # This file
```

## Installation

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Check test coverage
npm run test:coverage

# Lint code
npm run lint

# Format code
npm run format
```

## Development Status

This project is currently in initial development. **Task 1 (project structure and core types) has been completed successfully.**

### Completed ✅
- **TypeScript project setup** with tsconfig.json, proper build configuration
- **Core type definitions and interfaces** for all system components
- **Zod schemas** for runtime validation and type guards
- **Testing framework (Vitest)** with comprehensive unit tests (28 tests passing)
- **Linting (ESLint) and formatting (Prettier)** setup with strict TypeScript rules
- **CLI framework** with Commander.js and all planned commands stubbed
- **Build system** producing clean JavaScript output with type declarations
- **Test coverage** at 97%+ with proper validation of all core types

### Next Steps 🚧
- Core data structures implementation (Task 2)
- Problem Classifier component (Task 3)
- Context Extractor component (Task 5)
- Context Packager component (Task 6)
- Template Library component (Task 8)
- Full CLI implementation with actual functionality

## Technology Stack

- **Language**: TypeScript 5.3+
- **Runtime**: Node.js 20+
- **Testing**: Vitest
- **Linting**: ESLint with TypeScript support
- **Formatting**: Prettier
- **CLI Framework**: Commander.js (planned)
- **Validation**: Zod (planned)
- **YAML Parsing**: js-yaml (planned)

## Core Types

The system defines several core types:

- **DelegationType**: Categories of problems (architecture_design, api_design, test_strategy, etc.)
- **ComplexityLevel**: Problem complexity (simple, moderate, complex)
- **ContextType**: Types of context to extract (code_snippets, requirements_docs, etc.)
- **ArtifactType**: Types of artifacts Opus generates (mermaid_diagram, openapi_spec, etc.)
- **DelegationSession**: Complete session with rounds, artifacts, and metrics
- **ContextBundle**: Packaged context ready for Opus
- **ParsedArtifact**: Structured artifact from Opus response

## Contributing

This project follows strict TypeScript practices:

- All code must pass type checking (`npm run typecheck`)
- All code must pass linting (`npm run lint`)
- All code must be formatted with Prettier (`npm run format`)
- All new features must include unit tests
- Test coverage should be maintained above 80%

## License

MIT

## Related Documentation

- [Requirements Document](.kiro/specs/opus-delegation-system/requirements.md)
- [Design Document](.kiro/specs/opus-delegation-system/design.md)
- [Implementation Tasks](.kiro/specs/opus-delegation-system/tasks.md)
