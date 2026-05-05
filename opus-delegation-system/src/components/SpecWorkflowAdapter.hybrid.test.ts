/**
 * Unit tests for SpecWorkflowAdapter hybrid workflow support
 * 
 * Tests hybrid workflow functionality for Task 19.5
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SpecWorkflowAdapter, HybridWorkflowOptions, HybridSpecPackage } from './SpecWorkflowAdapter.js';
import { ParsedArtifact, ArtifactType } from '../types/core.js';

describe('SpecWorkflowAdapter - Hybrid Workflow Support', () => {
  let adapter: SpecWorkflowAdapter;
  let opusDesignArtifacts: ParsedArtifact[];
  let localRequirementsArtifacts: ParsedArtifact[];
  let opusRequirementsArtifacts: ParsedArtifact[];
  let localTaskArtifacts: ParsedArtifact[];

  beforeEach(() => {
    adapter = new SpecWorkflowAdapter();

    // Opus design artifacts
    opusDesignArtifacts = [
      {
        id: 'opus-arch-1',
        type: 'mermaid_diagram',
        content: `graph TB
          A[Frontend] --> B[API Gateway]
          B --> C[Auth Service]
          B --> D[Data Service]
          C --> E[Database]
          D --> E`,
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      },
      {
        id: 'opus-api-1',
        type: 'openapi_spec',
        content: JSON.stringify({
          openapi: '3.0.0',
          info: { title: 'Hybrid API', version: '1.0.0' },
          paths: {
            '/users': {
              get: { summary: 'Get users' },
              post: { summary: 'Create user' }
            }
          }
        }),
        metadata: {
          sourceLocation: { start: 0, end: 200 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      }
    ];

    // Local requirements artifacts
    localRequirementsArtifacts = [
      {
        id: 'local-req-1',
        type: 'code_snippet',
        content: `
          The system must authenticate users before allowing access.
          The system shall validate all input data.
          Users must be able to reset their passwords.
        `,
        metadata: {
          sourceLocation: { start: 0, end: 150 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      }
    ];

    // Opus requirements artifacts
    opusRequirementsArtifacts = [
      {
        id: 'opus-req-1',
        type: 'implementation_guide',
        content: 'Requirements from Opus',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date()
        },
        structured: {
          implementationSteps: [
            {
              id: 'req-step-1',
              phase: 'Authentication',
              title: 'User Authentication',
              description: 'Implement secure user authentication',
              action: 'implement',
              dependencies: [],
              complexity: 'moderate'
            }
          ]
        }
      }
    ];

    // Local task artifacts
    localTaskArtifacts = [
      {
        id: 'local-task-1',
        type: 'implementation_guide',
        content: 'Local implementation tasks',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date()
        },
        structured: {
          implementationSteps: [
            {
              id: 'task-step-1',
              phase: 'Setup',
              title: 'Database Setup',
              description: 'Configure database connections',
              action: 'configure',
              dependencies: [],
              complexity: 'simple'
            }
          ]
        }
      }
    ];
  });

  describe('generateHybridWorkflow', () => {
    it('should generate hybrid workflow with Opus design and local requirements', async () => {
      const result = await adapter.generateHybridWorkflow(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        {
          projectName: 'Hybrid System',
          opusSource: 'design',
          localSource: 'requirements',
          includeSourceTracking: true,
          validateConsistency: true
        }
      );

      expect(result.design).toBeDefined();
      expect(result.requirements).toBeDefined();
      expect(result.tasks).toBeDefined();
      expect(result.config).toBeDefined();

      expect(result.design!.title).toBe('Hybrid System Design Document');
      expect(result.requirements!.title).toBe('Hybrid System Requirements');
      expect(result.tasks!.title).toBe('Hybrid System Implementation Plan');

      expect(result.metadata.hybridType).toBe('design-opus-requirements-local');
      expect(result.metadata.opusArtifacts).toEqual(['opus-arch-1', 'opus-api-1']);
      expect(result.metadata.localArtifacts).toEqual(['local-req-1']);

      // Check source tracking annotations
      expect(result.design!.content).toContain('<!-- HYBRID WORKFLOW: design-opus-requirements-local -->');
      expect(result.requirements!.content).toContain('<!-- HYBRID WORKFLOW: design-opus-requirements-local -->');
      expect(result.tasks!.content).toContain('<!-- HYBRID WORKFLOW: design-opus-requirements-local -->');
    });

    it('should generate hybrid workflow with Opus requirements and local tasks', async () => {
      const result = await adapter.generateHybridWorkflow(
        opusRequirementsArtifacts,
        localTaskArtifacts,
        {
          projectName: 'Task Hybrid',
          opusSource: 'requirements',
          localSource: 'tasks',
          includeSourceTracking: true,
          validateConsistency: false
        }
      );

      expect(result.requirements).toBeDefined();
      expect(result.tasks).toBeDefined();
      expect(result.design).toBeDefined();

      expect(result.metadata.hybridType).toBe('requirements-opus-tasks-local');
      expect(result.metadata.opusArtifacts).toEqual(['opus-req-1']);
      expect(result.metadata.localArtifacts).toEqual(['local-task-1']);

      // Should not have consistency validation since it's disabled
      expect(result.metadata.consistencyValidation).toBeUndefined();
    });

    it('should handle Opus tasks with local design', async () => {
      const opusTaskArtifacts = [
        {
          id: 'opus-task-1',
          type: 'implementation_guide' as ArtifactType,
          content: 'Opus task content',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          },
          structured: {
            implementationSteps: [
              {
                id: 'opus-step-1',
                phase: 'Implementation',
                title: 'Core Logic',
                description: 'Implement core business logic',
                action: 'implement',
                dependencies: [],
                complexity: 'complex'
              }
            ]
          }
        }
      ];

      const localDesignArtifacts = [
        {
          id: 'local-design-1',
          type: 'mermaid_diagram' as ArtifactType,
          content: 'graph TD\n  A[Local Component] --> B[Local Service]',
          metadata: {
            sourceLocation: { start: 0, end: 50 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const result = await adapter.generateHybridWorkflow(
        opusTaskArtifacts,
        localDesignArtifacts,
        {
          projectName: 'Mixed System',
          opusSource: 'tasks',
          localSource: 'design'
        }
      );

      expect(result.tasks).toBeDefined();
      expect(result.design).toBeDefined();
      expect(result.requirements).toBeDefined();
      expect(result.metadata.hybridType).toBe('tasks-opus-design-local');
    });
  });

  describe('generateOpusDesignLocalRequirements', () => {
    it('should generate hybrid workflow with convenience method', async () => {
      const result = await adapter.generateOpusDesignLocalRequirements(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        'Convenience Test'
      );

      expect(result.design).toBeDefined();
      expect(result.requirements).toBeDefined();
      expect(result.tasks).toBeDefined();
      expect(result.metadata.hybridType).toBe('design-opus-requirements-local');
      expect(result.design!.title).toContain('Convenience Test');
    });
  });

  describe('generateOpusRequirementsLocalTasks', () => {
    it('should generate hybrid workflow with convenience method', async () => {
      const result = await adapter.generateOpusRequirementsLocalTasks(
        opusRequirementsArtifacts,
        localTaskArtifacts,
        'Task Convenience Test'
      );

      expect(result.requirements).toBeDefined();
      expect(result.tasks).toBeDefined();
      expect(result.design).toBeDefined();
      expect(result.metadata.hybridType).toBe('requirements-opus-tasks-local');
      expect(result.requirements!.title).toContain('Task Convenience Test');
    });
  });

  describe('source tracking', () => {
    it('should add source annotations when enabled', async () => {
      const result = await adapter.generateHybridWorkflow(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        {
          projectName: 'Source Tracking Test',
          opusSource: 'design',
          localSource: 'requirements',
          includeSourceTracking: true
        }
      );

      expect(result.design!.content).toContain('<!-- HYBRID WORKFLOW:');
      expect(result.design!.content).toContain('<!-- Generated:');
      expect(result.design!.content).toContain('<!-- Opus Source: artifacts [opus-arch-1, opus-api-1]');

      expect(result.requirements!.content).toContain('<!-- Local Source: artifacts [local-req-1]');
    });

    it('should not add source annotations when disabled', async () => {
      const result = await adapter.generateHybridWorkflow(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        {
          projectName: 'No Tracking Test',
          opusSource: 'design',
          localSource: 'requirements',
          includeSourceTracking: false
        }
      );

      expect(result.design!.content).not.toContain('<!-- HYBRID WORKFLOW:');
      expect(result.requirements!.content).not.toContain('<!-- Local Source:');
    });

    it('should handle mixed source documents', async () => {
      // Create a scenario where tasks are generated from both sources
      const result = await adapter.generateHybridWorkflow(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        {
          projectName: 'Mixed Source Test',
          opusSource: 'design',
          localSource: 'requirements',
          includeSourceTracking: true
        }
      );

      // Tasks should be generated from both Opus and local artifacts
      expect(result.tasks!.content).toContain('<!-- Mixed Sources:');
    });
  });

  describe('consistency validation', () => {
    it('should validate consistency between requirements and design', async () => {
      const inconsistentDesign = [
        {
          id: 'inconsistent-design',
          type: 'mermaid_diagram' as ArtifactType,
          content: 'graph TD\n  X[Unrelated] --> Y[Components]',
          metadata: {
            sourceLocation: { start: 0, end: 50 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const detailedRequirements = [
        {
          id: 'detailed-req',
          type: 'code_snippet' as ArtifactType,
          content: `
            The system must implement user authentication with JWT tokens.
            The system shall provide user management endpoints.
            The system must use PostgreSQL database for data persistence.
          `,
          metadata: {
            sourceLocation: { start: 0, end: 200 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const result = await adapter.generateHybridWorkflow(
        inconsistentDesign,
        detailedRequirements,
        {
          projectName: 'Consistency Test',
          opusSource: 'design',
          localSource: 'requirements',
          validateConsistency: true
        }
      );

      expect(result.metadata.consistencyValidation).toBeDefined();
      expect(result.metadata.consistencyValidation!.length).toBeGreaterThan(0);
    });

    it('should validate hybrid-specific consistency issues', async () => {
      const result = await adapter.generateHybridWorkflow(
        [], // No Opus artifacts
        localRequirementsArtifacts,
        {
          projectName: 'Empty Opus Test',
          opusSource: 'design',
          localSource: 'requirements',
          validateConsistency: true
        }
      );

      expect(result.metadata.consistencyValidation).toBeDefined();
      expect(result.metadata.consistencyValidation!.some(error => 
        error.includes('no Opus artifacts')
      )).toBe(true);
    });

    it('should validate workflow type consistency', async () => {
      // This test checks if the workflow type matches the hybrid configuration
      const result = await adapter.generateHybridWorkflow(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        {
          projectName: 'Workflow Consistency Test',
          opusSource: 'design',
          localSource: 'requirements',
          validateConsistency: true
        }
      );

      // Should be design-first since Opus provides design
      expect(result.config.workflowType).toBe('design-first');
      
      // Should not have workflow type mismatch errors
      const hasWorkflowError = result.metadata.consistencyValidation?.some(error => 
        error.includes('Workflow type mismatch')
      ) || false;
      expect(hasWorkflowError).toBe(false);
    });
  });

  describe('hybrid configuration generation', () => {
    it('should generate appropriate workflow type for design-first hybrid', async () => {
      const result = await adapter.generateHybridWorkflow(
        opusDesignArtifacts,
        localRequirementsArtifacts,
        {
          opusSource: 'design',
          localSource: 'requirements'
        }
      );

      expect(result.config.workflowType).toBe('design-first');
      expect(result.config.specType).toBe('feature');
      expect(result.config.metadata?.sourceArtifacts).toHaveLength(3); // 2 opus + 1 local
    });

    it('should generate appropriate workflow type for requirements-first hybrid', async () => {
      const result = await adapter.generateHybridWorkflow(
        opusRequirementsArtifacts,
        localTaskArtifacts,
        {
          opusSource: 'requirements',
          localSource: 'tasks'
        }
      );

      expect(result.config.workflowType).toBe('requirements-first');
      expect(result.config.specType).toBe('feature');
    });
  });

  describe('error handling', () => {
    it('should handle empty artifact arrays gracefully', async () => {
      const result = await adapter.generateHybridWorkflow(
        [],
        [],
        {
          projectName: 'Empty Test',
          opusSource: 'design',
          localSource: 'requirements'
        }
      );

      expect(result.design).toBeDefined();
      expect(result.requirements).toBeDefined();
      expect(result.tasks).toBeDefined();
      expect(result.metadata.opusArtifacts).toHaveLength(0);
      expect(result.metadata.localArtifacts).toHaveLength(0);
    });

    it('should handle malformed artifacts in hybrid workflow', async () => {
      const malformedArtifacts = [
        {
          id: 'malformed-1',
          type: 'openapi_spec' as ArtifactType,
          content: 'invalid json {',
          metadata: {
            sourceLocation: { start: 0, end: 20 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const result = await adapter.generateHybridWorkflow(
        malformedArtifacts,
        localRequirementsArtifacts,
        {
          projectName: 'Malformed Test',
          opusSource: 'design',
          localSource: 'requirements'
        }
      );

      // Should still generate documents despite malformed input
      expect(result.design).toBeDefined();
      expect(result.requirements).toBeDefined();
      expect(result.tasks).toBeDefined();
    });
  });

  describe('consistency validation helpers', () => {
    it('should extract key terms from document content', () => {
      const content = `
        # User Management System
        
        ## Authentication Module
        
        **JWT Tokens** are used for authentication.
        **User Roles** define access levels.
        
        ### Database Schema
      `;

      const terms = (adapter as any).extractKeyTerms(content);

      // The method extracts headers and bold text
      expect(terms.length).toBeGreaterThan(0);
      expect(terms).toContain('JWT Tokens');
      expect(terms).toContain('User Roles');
      
      // Just verify we get some terms - the exact extraction logic may vary
      expect(terms.length).toBeGreaterThanOrEqual(2);
    });

    it('should extract component names from content', () => {
      const content = `
        Component: UserService handles user operations
        Module: AuthModule provides authentication
        Service: DataService manages data access
        Class: UserController handles HTTP requests
      `;

      const components = (adapter as any).extractComponentNamesFromContent(content);

      expect(components).toContain('UserService');
      expect(components).toContain('AuthModule');
      expect(components).toContain('DataService');
      expect(components).toContain('UserController');
    });

    it('should extract API endpoints from content', () => {
      const content = `
        GET /api/users - retrieve users
        POST /api/users - create user
        Endpoint: /api/auth/login
        Path: /api/data/{id}
      `;

      const endpoints = (adapter as any).extractAPIEndpointsFromContent(content);

      expect(endpoints).toContain('/api/users');
      expect(endpoints).toContain('/api/auth/login');
      expect(endpoints).toContain('/api/data/{id}');
    });

    it('should extract requirement IDs from content', () => {
      const content = `
        ### Requirement 1.1: User Authentication
        REQ-2.3: Data validation
        R-3.1: Performance requirements
        Requirement: 4.2 Security measures
      `;

      const reqIds = (adapter as any).extractRequirementIds(content);

      expect(reqIds).toContain('Requirement 1.1');
      expect(reqIds).toContain('Requirement 2.3');
      expect(reqIds).toContain('Requirement 3.1');
      expect(reqIds).toContain('Requirement 4.2');
    });
  });
});