# Opus Delegation System

A TypeScript system for delegating complex architectural problems to Claude Opus 4.5 via use.ai.

## Task 3 Implementation: Problem Classifier Component

This implementation completes **Task 3** from the opus-delegation-system spec:

### ✅ Task 3.1: Problem Classification Engine
- **Complexity indicator detection**: Identifies architectural scope, formal reasoning, novel patterns, integration complexity, and strategic decisions
- **Delegation type categorization**: Classifies problems into 6 types (architecture_design, api_design, test_strategy, integration_design, refactoring_analysis, formal_verification)
- **Delegation suitability scores**: Calculates 0-100 scores based on complexity indicators and type-specific multipliers

### ✅ Task 3.2: Context Recommendation Logic  
- **Problem type mapping**: Maps each delegation type to required context types (architecture_docs, api_endpoints, test_files, etc.)
- **Session complexity estimation**: Estimates simple/moderate/complex based on problem scope and type multipliers
- **Delegation recommendations**: Generates recommendations with artifact types and estimated rounds

### ✅ Task 3.3: Unit Tests
- **Classification logic tests**: Tests all 6 delegation types with various problem descriptions
- **Context recommendation accuracy**: Validates appropriate context types are recommended
- **Edge case handling**: Tests empty descriptions, ambiguous problems, multiple type indicators
- **Property-based testing properties**: Validates invariants, metamorphic properties, and error conditions

## Core Features

### Problem Classification
The `ProblemClassifier` analyzes problem descriptions using:
- **Complexity Indicators** (weighted scoring):
  - Architectural scope (30% weight)
  - Formal reasoning (30% weight) 
  - Novel patterns (20% weight)
  - Integration complexity (15% weight)
  - Strategic decisions (5% weight)

### Delegation Types
Supports 6 delegation types with specific context requirements:
- `architecture_design` → architecture_docs, existing_designs, constraints
- `api_design` → api_endpoints, code_snippets, existing_designs  
- `test_strategy` → test_files, code_snippets, requirements_docs
- `integration_design` → external_interfaces, architecture_docs, constraints
- `refactoring_analysis` → code_snippets, dependency_graphs, architecture_docs
- `formal_verification` → code_snippets, requirements_docs, constraints

### Suitability Assessment
- **Suitability threshold**: 60+ score and 30+ confidence for delegation recommendation
- **Complexity multipliers**: Different types have different complexity factors (formal_verification: 1.5x, refactoring: 0.8x)
- **Context estimation**: Estimates context size (15K-50K chars) and extraction complexity

## Usage

```typescript
import { ProblemClassifier } from './src/components/ProblemClassifier.js';

const classifier = new ProblemClassifier();

const result = classifier.classifyProblem(
  'Design a distributed microservices architecture for federated learning'
);

console.log(result.classification.delegationType); // 'architecture_design'
console.log(result.suitable); // true/false
console.log(result.reasoning); // Human-readable explanation
```

## Requirements Satisfied

This implementation satisfies **Requirements 1.1-1.6** from the spec:
- ✅ 1.1: Complexity indicator detection
- ✅ 1.2: Delegation type categorization  
- ✅ 1.3: Context type recommendations
- ✅ 1.4: Session complexity estimation
- ✅ 1.5: Delegation recommendations with artifact types
- ✅ 1.6: Completeness property (sufficient context for implementable artifacts)

## Testing

Run the test suite:
```bash
npm test
```

The implementation includes 46 comprehensive tests covering:
- All delegation types and complexity levels
- Context recommendation accuracy
- Edge cases and error conditions
- Property-based testing invariants

## Next Steps

This completes Task 3. The next tasks in the implementation plan are:
- Task 4: Checkpoint - Ensure all tests pass ✅
- Task 5: Context Extractor component
- Task 6: Context Packager component
- Task 7: Template Library component

## Architecture

The Problem Classifier is part of the larger Opus Delegation System architecture:

```
Problem Description → Problem Classifier → Context Extractor → Context Packager → Opus Delegator
                           ↓
                    Classification + Recommendations
```

The classifier provides the foundation for the entire delegation workflow by determining problem suitability and required context types.