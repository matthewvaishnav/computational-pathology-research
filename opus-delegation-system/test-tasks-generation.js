/**
 * Simple test script to demonstrate the generateTasks functionality
 */

import { SpecWorkflowAdapter } from './dist/components/SpecWorkflowAdapter.js';

// Create test artifacts
const implementationGuide = {
  id: 'impl-guide-1',
  type: 'implementation_guide',
  content: 'Implementation guide content',
  metadata: {
    sourceLocation: { start: 0, end: 100 },
    parseWarnings: [],
    extractedAt: new Date()
  },
  structured: {
    implementationSteps: [
      {
        id: 'step-1',
        phase: 'Setup',
        title: 'Initialize Database',
        description: 'Set up the database schema and connections for requirement 1.2',
        action: 'create',
        file: 'database.ts',
        dependencies: [],
        complexity: 'simple',
        estimate: '2 hours'
      },
      {
        id: 'step-2',
        phase: 'Setup',
        title: 'Configure Authentication',
        description: 'Implement user authentication system for requirement 2.1',
        action: 'implement',
        file: 'auth.ts',
        dependencies: ['step-1'],
        complexity: 'moderate',
        estimate: '4 hours'
      },
      {
        id: 'step-3',
        phase: 'Implementation',
        title: 'Build API Endpoints',
        description: 'Create REST API endpoints for requirement 3.1',
        action: 'implement',
        file: 'api.ts',
        dependencies: ['step-1', 'step-2'],
        complexity: 'complex',
        estimate: '8 hours'
      }
    ]
  }
};

const apiSpec = {
  id: 'api-spec-1',
  type: 'openapi_spec',
  content: JSON.stringify({
    openapi: '3.0.0',
    info: { title: 'Test API', version: '1.0.0' },
    paths: {
      '/users': {
        get: { summary: 'Get users' },
        post: { summary: 'Create user' }
      },
      '/users/{id}': {
        get: { summary: 'Get user by ID' },
        put: { summary: 'Update user' },
        delete: { summary: 'Delete user' }
      }
    },
    components: {
      schemas: {
        User: {
          type: 'object',
          properties: {
            id: { type: 'integer' },
            name: { type: 'string' }
          }
        }
      }
    }
  }),
  metadata: {
    sourceLocation: { start: 0, end: 500 },
    parseWarnings: [],
    extractedAt: new Date()
  }
};

async function testTasksGeneration() {
  console.log('Testing Tasks Generation...\n');
  
  const adapter = new SpecWorkflowAdapter();
  
  // Test with implementation guide
  console.log('=== Implementation Guide to Tasks ===');
  const implResult = await adapter.generateTasks([implementationGuide], {
    projectName: 'Test Project',
    includeComplexityEstimates: true,
    includeRequirementsReferences: true,
    includeDependencies: true
  });
  
  console.log('Title:', implResult.title);
  console.log('Task Count:', implResult.metadata.taskCount);
  console.log('Validation Errors:', implResult.metadata.validationErrors.length);
  console.log('\nGenerated Tasks Document:');
  console.log(implResult.content);
  
  console.log('\n' + '='.repeat(60) + '\n');
  
  // Test with API spec
  console.log('=== API Specification to Tasks ===');
  const apiResult = await adapter.generateTasks([apiSpec], {
    projectName: 'API Project',
    includeComplexityEstimates: true
  });
  
  console.log('Title:', apiResult.title);
  console.log('Task Count:', apiResult.metadata.taskCount);
  console.log('\nGenerated Tasks Document:');
  console.log(apiResult.content);
  
  console.log('\n' + '='.repeat(60) + '\n');
  
  // Test with combined artifacts
  console.log('=== Combined Artifacts to Tasks ===');
  const combinedResult = await adapter.generateTasks([implementationGuide, apiSpec], {
    projectName: 'Full System',
    groupByPhase: true
  });
  
  console.log('Title:', combinedResult.title);
  console.log('Task Count:', combinedResult.metadata.taskCount);
  console.log('Source Artifacts:', combinedResult.metadata.sourceArtifacts);
  console.log('\nGenerated Tasks Document:');
  console.log(combinedResult.content);
}

testTasksGeneration().catch(console.error);