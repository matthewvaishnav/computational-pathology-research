/**
 * Example: Requirements.md Generation from Opus Artifacts
 * 
 * This example demonstrates how to use the SpecWorkflowAdapter to convert
 * Opus-generated artifacts into EARS-compliant requirements.md documents.
 */

import { SpecWorkflowAdapter } from '../src/components/SpecWorkflowAdapter.js';
import { ParsedArtifact } from '../src/types/core.js';

async function demonstrateRequirementsGeneration() {
  const adapter = new SpecWorkflowAdapter();

  // Example 1: Implementation Guide Artifact
  const implementationGuide: ParsedArtifact = {
    id: 'federated-learning-impl',
    type: 'implementation_guide',
    content: 'Federated Learning Implementation Guide',
    metadata: {
      sourceLocation: { start: 0, end: 1000 },
      parseWarnings: [],
      extractedAt: new Date()
    },
    structured: {
      implementationSteps: [
        {
          id: 'step-1',
          phase: 'Setup',
          title: 'Initialize Federated Network',
          description: 'Set up the federated learning network infrastructure',
          action: 'create',
          file: 'network.ts',
          dependencies: [],
          complexity: 'moderate'
        },
        {
          id: 'step-2',
          phase: 'Setup',
          title: 'Configure Client Nodes',
          description: 'Configure individual client nodes for federated training',
          action: 'configure',
          file: 'client.ts',
          dependencies: ['step-1'],
          complexity: 'complex'
        },
        {
          id: 'step-3',
          phase: 'Implementation',
          title: 'Implement Model Aggregation',
          description: 'Implement federated averaging algorithm',
          action: 'implement',
          file: 'aggregator.ts',
          dependencies: ['step-1', 'step-2'],
          complexity: 'complex'
        }
      ]
    }
  };

  // Example 2: API Specification Artifact
  const apiSpec: ParsedArtifact = {
    id: 'federated-api-spec',
    type: 'openapi_spec',
    content: JSON.stringify({
      openapi: '3.0.0',
      info: {
        title: 'Federated Learning API',
        version: '1.0.0',
        description: 'API for federated learning coordination'
      },
      paths: {
        '/models': {
          get: { summary: 'List available models' },
          post: { summary: 'Create new model' }
        },
        '/models/{id}/train': {
          post: { summary: 'Start federated training' }
        },
        '/clients': {
          get: { summary: 'List registered clients' },
          post: { summary: 'Register new client' }
        },
        '/clients/{id}/weights': {
          get: { summary: 'Get client model weights' },
          put: { summary: 'Update client model weights' }
        }
      }
    }),
    metadata: {
      sourceLocation: { start: 0, end: 2000 },
      parseWarnings: [],
      extractedAt: new Date()
    }
  };

  // Example 3: Architecture Diagram Artifact
  const architectureDiagram: ParsedArtifact = {
    id: 'federated-architecture',
    type: 'mermaid_diagram',
    content: `
      graph TB
        subgraph "Federated Learning System"
          Coordinator[Central Coordinator] --> ClientA[Client A]
          Coordinator --> ClientB[Client B]
          Coordinator --> ClientC[Client C]
          ClientA --> LocalData1[Local Dataset A]
          ClientB --> LocalData2[Local Dataset B]
          ClientC --> LocalData3[Local Dataset C]
          Coordinator --> ModelRegistry[Model Registry]
          Coordinator --> AggregationEngine[Aggregation Engine]
        end
    `,
    metadata: {
      sourceLocation: { start: 0, end: 500 },
      parseWarnings: [],
      extractedAt: new Date()
    }
  };

  // Generate requirements document
  console.log('🔄 Generating requirements.md from Opus artifacts...\n');

  const result = await adapter.generateRequirements(
    [implementationGuide, apiSpec, architectureDiagram],
    {
      projectName: 'Federated Learning System',
      includePropertyBasedTesting: true,
      earsValidation: true,
      requirementIdPrefix: 'FL-REQ'
    }
  );

  console.log('📋 Generated Requirements Document:');
  console.log('=' .repeat(50));
  console.log(result.content);
  console.log('=' .repeat(50));

  console.log('\n📊 Generation Metadata:');
  console.log(`- Title: ${result.title}`);
  console.log(`- Generated at: ${result.metadata.generatedAt.toISOString()}`);
  console.log(`- Source artifacts: ${result.metadata.sourceArtifacts.length}`);
  console.log(`- EARS compliant: ${result.metadata.earsCompliant ? '✅' : '❌'}`);
  
  if (result.metadata.validationErrors.length > 0) {
    console.log('\n⚠️  Validation Errors:');
    result.metadata.validationErrors.forEach((error, index) => {
      console.log(`  ${index + 1}. ${error}`);
    });
  }

  console.log('\n✅ Requirements generation complete!');
  
  return result;
}

// Example of EARS pattern validation
async function demonstrateEARSValidation() {
  const adapter = new SpecWorkflowAdapter();

  console.log('\n🔍 EARS Pattern Validation Examples:');
  console.log('-'.repeat(40));

  const validStatements = [
    'THE System SHALL authenticate users before granting access',
    'WHEN a user submits invalid credentials, THE System SHALL log the attempt',
    'IF the database connection fails, THEN THE System SHALL retry up to 3 times',
    'WHERE user has admin privileges, THE System SHALL allow configuration changes'
  ];

  const invalidStatements = [
    'System should work correctly',
    'Users can access the application',
    'The system will be fast',
    'Must validate all inputs'
  ];

  console.log('✅ Valid EARS Patterns:');
  validStatements.forEach((statement, index) => {
    const isValid = (adapter as any).isValidEARSStatement(statement);
    console.log(`  ${index + 1}. ${statement} - ${isValid ? '✅' : '❌'}`);
  });

  console.log('\n❌ Invalid EARS Patterns:');
  invalidStatements.forEach((statement, index) => {
    const isValid = (adapter as any).isValidEARSStatement(statement);
    console.log(`  ${index + 1}. ${statement} - ${isValid ? '✅' : '❌'}`);
  });
}

// Run the examples
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateRequirementsGeneration()
    .then(() => demonstrateEARSValidation())
    .catch(console.error);
}

export { demonstrateRequirementsGeneration, demonstrateEARSValidation };