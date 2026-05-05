/**
 * Example demonstrating .config.kiro generation functionality
 * 
 * This example shows how to use the SpecWorkflowAdapter to generate
 * .config.kiro files with appropriate workflow type and spec metadata.
 */

import { SpecWorkflowAdapter } from '../src/components/SpecWorkflowAdapter.js';
import { ParsedArtifact } from '../src/types/core.js';

async function demonstrateConfigGeneration() {
  const adapter = new SpecWorkflowAdapter();

  console.log('=== .config.kiro Generation Examples ===\n');

  // Example 1: Feature with architecture diagram (design-first)
  console.log('1. Feature with Architecture Diagram (Design-First):');
  const architectureArtifact: ParsedArtifact = {
    id: 'arch-diagram-1',
    type: 'mermaid_diagram',
    content: `
      graph TB
        subgraph "Frontend Layer"
          A[React App]
          B[Redux Store]
        end
        subgraph "Backend Layer"
          C[API Gateway]
          D[Auth Service]
          E[Data Service]
        end
        A --> C
        C --> D
        C --> E
    `,
    metadata: {
      sourceLocation: { start: 0, end: 200 },
      parseWarnings: [],
      extractedAt: new Date()
    }
  };

  const designFirstConfig = await adapter.generateConfig([architectureArtifact], {
    projectName: 'User Management System'
  });

  console.log('Generated Config:', JSON.stringify({
    specId: designFirstConfig.specId,
    workflowType: designFirstConfig.workflowType,
    specType: designFirstConfig.specType
  }, null, 2));

  console.log('Exported .config.kiro content:');
  console.log(adapter.exportConfigKiro(designFirstConfig));
  console.log();

  // Example 2: Bugfix workflow
  console.log('2. Bugfix Workflow:');
  const bugfixArtifact: ParsedArtifact = {
    id: 'bugfix-1',
    type: 'implementation_guide',
    content: `
      Bug Fix: Timeout issue in CI pipeline
      
      Problem: Tests are failing due to timeout errors in the CI system.
      The issue occurs when running integration tests that take longer than expected.
      
      Solution: Increase timeout values and optimize test execution.
    `,
    metadata: {
      sourceLocation: { start: 0, end: 200 },
      parseWarnings: [],
      extractedAt: new Date()
    }
  };

  const bugfixConfig = await adapter.generateConfig([bugfixArtifact], {
    projectName: 'CI Pipeline Fix'
  });

  console.log('Generated Config:', JSON.stringify({
    specId: bugfixConfig.specId,
    workflowType: bugfixConfig.workflowType,
    specType: bugfixConfig.specType
  }, null, 2));

  console.log('Exported .config.kiro content:');
  console.log(adapter.exportConfigKiro(bugfixConfig));
  console.log();

  // Example 3: Requirements-first workflow
  console.log('3. Requirements-First Workflow:');
  const requirementsArtifact: ParsedArtifact = {
    id: 'requirements-1',
    type: 'code_snippet',
    content: `
      User Story: As a user, I want to be able to login to the system.
      
      Acceptance Criteria:
      - User can enter username and password
      - System validates credentials
      - User is redirected to dashboard on success
      - Failed login attempts are logged
    `,
    metadata: {
      sourceLocation: { start: 0, end: 150 },
      parseWarnings: [],
      extractedAt: new Date()
    }
  };

  const requirementsFirstConfig = await adapter.generateConfig([requirementsArtifact], {
    projectName: 'Authentication System'
  });

  console.log('Generated Config:', JSON.stringify({
    specId: requirementsFirstConfig.specId,
    workflowType: requirementsFirstConfig.workflowType,
    specType: requirementsFirstConfig.specType
  }, null, 2));

  console.log('Exported .config.kiro content:');
  console.log(adapter.exportConfigKiro(requirementsFirstConfig));
  console.log();

  // Example 4: OpenAPI spec (design-first)
  console.log('4. OpenAPI Specification (Design-First):');
  const apiSpecArtifact: ParsedArtifact = {
    id: 'api-spec-1',
    type: 'openapi_spec',
    content: JSON.stringify({
      openapi: '3.0.0',
      info: { title: 'User API', version: '1.0.0' },
      paths: {
        '/users': {
          get: { summary: 'Get users' },
          post: { summary: 'Create user' }
        },
        '/users/{id}': {
          get: { summary: 'Get user' },
          put: { summary: 'Update user' },
          delete: { summary: 'Delete user' }
        }
      }
    }),
    metadata: {
      sourceLocation: { start: 0, end: 300 },
      parseWarnings: [],
      extractedAt: new Date()
    }
  };

  const apiSpecConfig = await adapter.generateConfig([apiSpecArtifact], {
    projectName: 'REST API Service'
  });

  console.log('Generated Config:', JSON.stringify({
    specId: apiSpecConfig.specId,
    workflowType: apiSpecConfig.workflowType,
    specType: apiSpecConfig.specType
  }, null, 2));

  console.log('Exported .config.kiro content:');
  console.log(adapter.exportConfigKiro(apiSpecConfig));
  console.log();

  console.log('=== Summary ===');
  console.log('✅ Successfully generated .config.kiro files for different workflow types');
  console.log('✅ Automatic workflow detection based on artifact content');
  console.log('✅ Support for feature and bugfix spec types');
  console.log('✅ Clean JSON export format matching existing .config.kiro files');
}

// Run the example
demonstrateConfigGeneration().catch(console.error);