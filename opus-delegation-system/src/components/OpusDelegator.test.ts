/**
 * Unit tests for Opus Delegator Component
 * Tests Task 9.5 - Comprehensive unit tests for Opus Delegator
 * Requirements: 3.1-3.7 (Delegation Request Generation), 9.1-9.7 (Multi-Round Delegation Support)
 * 
 * Test Coverage:
 * - Delegation request generation (3.1, 3.2, 3.3, 3.6)
 * - Context bundle integration (3.4, 3.5)
 * - Multi-round session management (9.1, 9.2, 9.3, 9.4, 9.5)
 * - Automatic follow-up generation (9.6)
 * - Session completion detection (9.7)
 * - Property-based testing patterns and invariants
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { OpusDelegator } from './OpusDelegator.js';
import { TemplateLibrary } from './TemplateLibrary.js';
import { 
  DelegationType, 
  ComplexityLevel, 
  ArtifactType,
  ParsedArtifact,
  ValidationResult
} from '../types/core.js';
import { ContextBundle } from './ContextPackager.js';

describe('OpusDelegator', () => {
  let delegator: OpusDelegator;
  let templateLibrary: TemplateLibrary;

  beforeEach(() => {
    templateLibrary = new TemplateLibrary('./templates');
    templateLibrary.loadAllTemplates();
    delegator = new OpusDelegator(templateLibrary);
  });

  describe('generateDelegationRequest', () => {
    it('should generate delegation request with all required fields', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test Problem',
        problemSummary: 'Test summary',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: ['Constraint 1', 'Constraint 2'],
        contextManifest: {
          sources: [],
          totalFiles: 0,
          totalSize: 0
        },
        totalSize: 100,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Federated Learning Architecture',
        'Design a federated learning system',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle,
        'federated_learning_architecture',
        {
          system_name: 'FedLearn',
          node_types: ['coordinator', 'worker']
        }
      );

      expect(request.sessionId).toBe('session-1');
      expect(request.roundNumber).toBe(1);
      expect(request.problemDescription).toBe('Design a federated learning system');
      expect(request.objectives.length).toBeGreaterThan(0);
      expect(request.constraints).toEqual(['Constraint 1', 'Constraint 2']);
      expect(request.expectedArtifacts.length).toBeGreaterThan(0);
      expect(request.outputFormatRequirements.length).toBeGreaterThan(0);
      expect(request.questionsToAddress.length).toBeGreaterThan(0);
      expect(request.contextBundle).toBeDefined();
      expect(request.generatedAt).toBeInstanceOf(Date);
    });

    it('should include expected artifacts based on template', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle,
        'federated_learning_architecture'
      );

      expect(request.expectedArtifacts).toContainEqual(
        expect.objectContaining({
          type: ArtifactType.MERMAID_DIAGRAM
        })
      );
      expect(request.expectedArtifacts).toContainEqual(
        expect.objectContaining({
          type: ArtifactType.OPENAPI_SPEC
        })
      );
    });

    it('should include output format requirements', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test',
        'Test problem',
        DelegationType.API_DESIGN,
        contextBundle,
        'pacs_integration_design'
      );

      expect(request.outputFormatRequirements.length).toBeGreaterThan(0);
      expect(request.outputFormatRequirements[0]).toHaveProperty('artifactType');
      expect(request.outputFormatRequirements[0]).toHaveProperty('format');
      expect(request.outputFormatRequirements[0]).toHaveProperty('instructions');
    });

    it('should generate appropriate questions for delegation type', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      const archRequest = delegator.generateDelegationRequest(
        'session-1',
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle,
        'federated_learning_architecture'
      );

      // Check that questions are generated for architecture design
      expect(archRequest.questionsToAddress.length).toBeGreaterThan(0);
      const hasComponentQuestion = archRequest.questionsToAddress.some(q => 
        q.toLowerCase().includes('component')
      );
      expect(hasComponentQuestion).toBe(true);

      const integrationRequest = delegator.generateDelegationRequest(
        'session-2',
        'Test',
        'Test problem',
        DelegationType.INTEGRATION_DESIGN,
        contextBundle,
        'pacs_integration_design'
      );

      // Check that questions are generated for integration design
      expect(integrationRequest.questionsToAddress.length).toBeGreaterThan(0);
    });

    it('should throw error if template not found', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      expect(() => {
        delegator.generateDelegationRequest(
          'session-1',
          'Test',
          'Test problem',
          DelegationType.ARCHITECTURE_DESIGN,
          contextBundle,
          'nonexistent_template'
        );
      }).toThrow('Template not found');
    });
  });

  describe('formatDelegationRequestAsText', () => {
    it('should format request as copy-paste ready markdown', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: ['Constraint 1'],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test',
        'Design a system',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle
      );

      const formatted = delegator.formatDelegationRequestAsText(request);

      expect(formatted).toContain('# Delegation Request: Round 1');
      expect(formatted).toContain('## Objective');
      expect(formatted).toContain('Design a system');
      expect(formatted).toContain('## Expected Artifacts');
      expect(formatted).toContain('## Output Format Requirements');
      expect(formatted).toContain('## Context');
      expect(formatted).toContain('Constraint 1');
    });

    it('should include previous round summary for multi-round requests', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle
      );

      request.roundNumber = 2;
      request.previousRoundSummary = 'Round 1 produced 2 artifacts';
      request.artifactReferences = ['mermaid_diagram (version 1)'];

      const formatted = delegator.formatDelegationRequestAsText(request);

      expect(formatted).toContain('## Previous Round Summary');
      expect(formatted).toContain('Round 1 produced 2 artifacts');
      expect(formatted).toContain('## Previous Artifacts to Refine');
      expect(formatted).toContain('mermaid_diagram (version 1)');
    });

    it('should format expected artifacts with descriptions', () => {
      const contextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 100,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle
      );

      const formatted = delegator.formatDelegationRequestAsText(request);

      expect(formatted).toMatch(/\d+\.\s+\*\*.+\*\*\s+—\s+.+/); // Numbered artifact with description
    });
  });

  describe('initializeSession', () => {
    it('should create new session with correct structure', () => {
      const session = delegator.initializeSession(
        'Test Problem',
        'Design a federated learning system',
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.COMPLEX
      );

      expect(session.id).toBeDefined();
      expect(session.createdAt).toBeInstanceOf(Date);
      expect(session.updatedAt).toBeInstanceOf(Date);
      expect(session.problem.title).toBe('Test Problem');
      expect(session.problem.description).toBe('Design a federated learning system');
      expect(session.problem.type).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(session.problem.complexity).toBe(ComplexityLevel.COMPLEX);
      expect(session.rounds).toEqual([]);
      expect(session.finalArtifacts).toEqual([]);
      expect(session.metrics.roundCount).toBe(0);
    });

    it('should use default complexity if not provided', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.API_DESIGN
      );

      expect(session.problem.complexity).toBe(ComplexityLevel.MODERATE);
    });

    it('should initialize session state internally', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      // Session should be retrievable
      const retrieved = delegator.getSession(session.id);
      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe(session.id);
    });
  });

  describe('addRound', () => {
    it('should add round to session', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        {
          id: 'artifact-1',
          type: ArtifactType.MERMAID_DIAGRAM,
          content: 'graph TD\nA-->B',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const validation: ValidationResult = {
        completenessScore: 85,
        qualityScores: {
          completeness: 85,
          clarity: 80,
          implementability: 75
        },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(
        session.id,
        'Initial request',
        'Opus response',
        artifacts,
        validation
      );

      const updated = delegator.getSession(session.id);
      expect(updated?.rounds.length).toBe(1);
      expect(updated?.rounds[0].roundNumber).toBe(1);
      expect(updated?.rounds[0].artifacts).toEqual(artifacts);
      expect(updated?.rounds[0].validation).toEqual(validation);
      expect(updated?.metrics.roundCount).toBe(1);
    });

    it('should update final artifacts if validation passed', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        {
          id: 'artifact-1',
          type: ArtifactType.MERMAID_DIAGRAM,
          content: 'graph TD\nA-->B',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const validation: ValidationResult = {
        completenessScore: 90,
        qualityScores: {
          completeness: 90,
          clarity: 85,
          implementability: 80
        },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      const updated = delegator.getSession(session.id);
      expect(updated?.finalArtifacts).toEqual(artifacts);
      expect(updated?.metrics.finalCompleteness).toBe(90);
    });

    it('should track artifact version history', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifact1: ParsedArtifact = {
        id: 'artifact-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\nA-->B',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      };

      const validation: ValidationResult = {
        completenessScore: 70,
        qualityScores: { completeness: 70, clarity: 70, implementability: 70 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request 1', 'Response 1', [artifact1], validation);

      const artifact2: ParsedArtifact = {
        id: 'artifact-2',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\nA-->B\nB-->C',
        metadata: {
          sourceLocation: { start: 0, end: 150 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      };

      delegator.addRound(session.id, 'Request 2', 'Response 2', [artifact2], validation);

      const history = delegator.getArtifactVersionHistory(session.id, ArtifactType.MERMAID_DIAGRAM);
      expect(history.length).toBe(2);
      expect(history[0].version).toBe(1);
      expect(history[0].roundNumber).toBe(1);
      expect(history[1].version).toBe(2);
      expect(history[1].roundNumber).toBe(2);
      expect(history[1].changes).toBeDefined();
    });

    it('should throw error if session not found', () => {
      const artifacts: ParsedArtifact[] = [];
      const validation: ValidationResult = {
        completenessScore: 0,
        qualityScores: { completeness: 0, clarity: 0, implementability: 0 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      expect(() => {
        delegator.addRound('nonexistent', 'Request', 'Response', artifacts, validation);
      }).toThrow('Session not found');
    });
  });

  describe('generateFollowUpRequest', () => {
    it('should generate follow-up request for incomplete artifacts', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        {
          id: 'artifact-1',
          type: ArtifactType.MERMAID_DIAGRAM,
          content: 'graph TD\nA-->B',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const validation: ValidationResult = {
        completenessScore: 60,
        qualityScores: {
          completeness: 60,
          clarity: 70,
          implementability: 65
        },
        isValid: false,
        errors: [
          {
            type: 'missing_element',
            message: 'Missing component descriptions',
            location: 'artifact-1',
            severity: 'error'
          }
        ],
        warnings: [],
        followUpQuestions: ['What are the component responsibilities?']
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      const followUp = delegator.generateFollowUpRequest(session.id, validation, artifacts);

      expect(followUp.sessionId).toBe(session.id);
      expect(followUp.roundNumber).toBe(2);
      expect(followUp.previousRoundSummary).toBeDefined();
      expect(followUp.artifactReferences).toBeDefined();
      expect(followUp.artifactReferences?.length).toBeGreaterThan(0);
      expect(followUp.questionsToAddress.length).toBeGreaterThan(0);
    });

    it('should include clarifying questions from validation', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.API_DESIGN
      );

      const artifacts: ParsedArtifact[] = [];
      const validation: ValidationResult = {
        completenessScore: 50,
        qualityScores: { completeness: 50, clarity: 60, implementability: 55 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: [
          'What authentication method should be used?',
          'What are the rate limiting requirements?'
        ]
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      const followUp = delegator.generateFollowUpRequest(session.id, validation, artifacts);

      expect(followUp.questionsToAddress).toContain('What authentication method should be used?');
      expect(followUp.questionsToAddress).toContain('What are the rate limiting requirements?');
    });

    it('should reference previous artifacts', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        {
          id: 'artifact-1',
          type: ArtifactType.MERMAID_DIAGRAM,
          content: 'graph TD\nA-->B',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const validation: ValidationResult = {
        completenessScore: 70,
        qualityScores: { completeness: 70, clarity: 70, implementability: 70 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      const followUp = delegator.generateFollowUpRequest(session.id, validation, artifacts);

      expect(followUp.artifactReferences).toBeDefined();
      expect(followUp.artifactReferences?.length).toBe(1);
      expect(followUp.artifactReferences?.[0]).toContain('mermaid_diagram');
      expect(followUp.artifactReferences?.[0]).toContain('version 1');
    });

    it('should throw error if session not found', () => {
      const validation: ValidationResult = {
        completenessScore: 0,
        qualityScores: { completeness: 0, clarity: 0, implementability: 0 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      expect(() => {
        delegator.generateFollowUpRequest('nonexistent', validation, []);
      }).toThrow('Session not found');
    });
  });

  describe('detectSessionCompletion', () => {
    it('should detect completion when criteria met', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        {
          id: 'artifact-1',
          type: ArtifactType.MERMAID_DIAGRAM,
          content: 'graph TD\nA-->B',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        },
        {
          id: 'artifact-2',
          type: ArtifactType.IMPLEMENTATION_GUIDE,
          content: '# Implementation Guide',
          metadata: {
            sourceLocation: { start: 0, end: 100 },
            parseWarnings: [],
            extractedAt: new Date()
          }
        }
      ];

      const validation: ValidationResult = {
        completenessScore: 85,
        qualityScores: {
          completeness: 85,
          clarity: 80,
          implementability: 75
        },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(true);
    });

    it('should not detect completion if completeness below threshold', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const validation: ValidationResult = {
        completenessScore: 70, // Below 80% threshold
        qualityScores: {
          completeness: 70,
          clarity: 80,
          implementability: 75
        },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', [], validation);

      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(false);
    });

    it('should not detect completion if quality below threshold', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const validation: ValidationResult = {
        completenessScore: 85,
        qualityScores: {
          completeness: 85,
          clarity: 80,
          implementability: 65 // Below 70% threshold
        },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', [], validation);

      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(false);
    });

    it('should detect completion if max rounds reached', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const validation: ValidationResult = {
        completenessScore: 70,
        qualityScores: { completeness: 70, clarity: 70, implementability: 70 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      // Add 5 rounds (max rounds = 5)
      for (let i = 0; i < 5; i++) {
        delegator.addRound(session.id, 'Request', 'Response', [], validation);
      }

      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(true);
    });

    it('should return false if session not found', () => {
      const validation: ValidationResult = {
        completenessScore: 100,
        qualityScores: { completeness: 100, clarity: 100, implementability: 100 },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      const isComplete = delegator.detectSessionCompletion('nonexistent', validation);
      expect(isComplete).toBe(false);
    });
  });

  describe('getArtifactVersionHistory', () => {
    it('should return version history for artifact type', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifact1: ParsedArtifact = {
        id: 'artifact-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'version 1',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      };

      const artifact2: ParsedArtifact = {
        id: 'artifact-2',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'version 2',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      };

      const validation: ValidationResult = {
        completenessScore: 70,
        qualityScores: { completeness: 70, clarity: 70, implementability: 70 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request 1', 'Response 1', [artifact1], validation);
      delegator.addRound(session.id, 'Request 2', 'Response 2', [artifact2], validation);

      const history = delegator.getArtifactVersionHistory(session.id, ArtifactType.MERMAID_DIAGRAM);

      expect(history.length).toBe(2);
      expect(history[0].version).toBe(1);
      expect(history[0].artifact.content).toBe('version 1');
      expect(history[1].version).toBe(2);
      expect(history[1].artifact.content).toBe('version 2');
    });

    it('should return empty array if no versions exist', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const history = delegator.getArtifactVersionHistory(session.id, ArtifactType.MERMAID_DIAGRAM);
      expect(history).toEqual([]);
    });

    it('should return empty array if session not found', () => {
      const history = delegator.getArtifactVersionHistory('nonexistent', ArtifactType.MERMAID_DIAGRAM);
      expect(history).toEqual([]);
    });
  });

  // ========== ADDITIONAL COMPREHENSIVE TESTS FOR TASK 9.5 ==========

  describe('Property-Based Testing Patterns', () => {
    describe('Delegation Request Generation Invariants (Requirement 3)', () => {
      it('should always include problem description, context bundle, and expected artifacts (Invariant)', () => {
        const contextBundle = createTestContextBundle();
        
        // Test with delegation types that have corresponding templates
        const delegationTypesWithTemplates = [
          { type: DelegationType.ARCHITECTURE_DESIGN, template: 'federated_learning_architecture' },
          { type: DelegationType.INTEGRATION_DESIGN, template: 'pacs_integration_design' },
          { type: DelegationType.TEST_STRATEGY, template: 'property_based_test_suite' },
          { type: DelegationType.REFACTORING_ANALYSIS, template: 'refactoring_analysis' }
        ];

        for (const { type, template } of delegationTypesWithTemplates) {
          const request = delegator.generateDelegationRequest(
            `session-${type}`,
            'Test Problem',
            'Test description',
            type,
            contextBundle,
            template
          );

          // Invariant: All delegation requests include required fields
          expect(request.problemDescription).toBeDefined();
          expect(request.problemDescription.length).toBeGreaterThan(0);
          expect(request.contextBundle).toBeDefined();
          expect(request.expectedArtifacts.length).toBeGreaterThan(0);
        }
      });

      it('should generate valid markdown format (Invariant)', () => {
        const contextBundle = createTestContextBundle();
        const request = delegator.generateDelegationRequest(
          'session-1',
          'Test Problem',
          'Test description',
          DelegationType.ARCHITECTURE_DESIGN,
          contextBundle
        );

        const formatted = delegator.formatDelegationRequestAsText(request);

        // Invariant: Delegation request format is valid markdown
        expect(formatted).toContain('# Delegation Request:');
        expect(formatted).toContain('## Objective');
        expect(formatted).toContain('## Expected Artifacts');
        expect(formatted).toContain('## Context');
        
        // Should have proper markdown structure
        const lines = formatted.split('\n');
        const headerLines = lines.filter(line => line.startsWith('#'));
        expect(headerLines.length).toBeGreaterThan(0);
      });

      it('should produce more specific artifact requirements with detailed descriptions (Metamorphic)', () => {
        const contextBundle = createTestContextBundle();
        
        const simpleRequest = delegator.generateDelegationRequest(
          'session-1',
          'Simple Problem',
          'Simple description',
          DelegationType.ARCHITECTURE_DESIGN,
          contextBundle,
          'federated_learning_architecture'
        );

        const detailedRequest = delegator.generateDelegationRequest(
          'session-2',
          'Complex Federated Learning Architecture',
          'Design a comprehensive federated learning system with multiple node types, privacy-preserving aggregation, and fault tolerance mechanisms',
          DelegationType.ARCHITECTURE_DESIGN,
          contextBundle,
          'federated_learning_architecture'
        );

        // Metamorphic: More detailed problem descriptions should produce more specific requirements
        expect(detailedRequest.questionsToAddress.length).toBeGreaterThanOrEqual(simpleRequest.questionsToAddress.length);
        expect(detailedRequest.expectedArtifacts.length).toBeGreaterThanOrEqual(simpleRequest.expectedArtifacts.length);
      });
    });

    describe('Multi-Round Session Invariants (Requirement 9)', () => {
      it('should reference previous round artifacts in follow-up requests (Invariant)', () => {
        const session = delegator.initializeSession(
          'Test',
          'Test problem',
          DelegationType.ARCHITECTURE_DESIGN
        );

        const artifacts: ParsedArtifact[] = [
          createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'graph TD\nA-->B')
        ];

        const validation: ValidationResult = {
          completenessScore: 70,
          qualityScores: { completeness: 70, clarity: 70, implementability: 70 },
          isValid: false,
          errors: [],
          warnings: [],
          followUpQuestions: ['Need more details']
        };

        delegator.addRound(session.id, 'Request 1', 'Response 1', artifacts, validation);
        const followUp = delegator.generateFollowUpRequest(session.id, validation, artifacts);

        // Invariant: Each round references previous round's artifacts
        expect(followUp.artifactReferences).toBeDefined();
        expect(followUp.artifactReferences!.length).toBeGreaterThan(0);
        expect(followUp.previousRoundSummary).toBeDefined();
        expect(followUp.roundNumber).toBe(2);
      });

      it('should maintain or increase artifact completeness across rounds (Invariant)', () => {
        const session = delegator.initializeSession(
          'Test',
          'Test problem',
          DelegationType.ARCHITECTURE_DESIGN
        );

        const artifacts1 = [createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'basic content')];
        const validation1: ValidationResult = {
          completenessScore: 60,
          qualityScores: { completeness: 60, clarity: 60, implementability: 60 },
          isValid: false,
          errors: [],
          warnings: [],
          followUpQuestions: []
        };

        delegator.addRound(session.id, 'Request 1', 'Response 1', artifacts1, validation1);

        const artifacts2 = [createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'enhanced content with more details')];
        const validation2: ValidationResult = {
          completenessScore: 80,
          qualityScores: { completeness: 80, clarity: 75, implementability: 70 },
          isValid: true,
          errors: [],
          warnings: [],
          followUpQuestions: []
        };

        delegator.addRound(session.id, 'Request 2', 'Response 2', artifacts2, validation2);

        const updatedSession = delegator.getSession(session.id);
        
        // Invariant: Artifact completeness score increases or stays same across rounds
        expect(updatedSession!.rounds[1].validation.completenessScore)
          .toBeGreaterThanOrEqual(updatedSession!.rounds[0].validation.completenessScore);
      });

      it('should produce more detailed artifacts with more rounds (Metamorphic)', () => {
        const session = delegator.initializeSession(
          'Test',
          'Test problem',
          DelegationType.ARCHITECTURE_DESIGN
        );

        // Round 1 - Basic artifact
        const artifact1 = createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'graph TD\nA-->B');
        const validation1: ValidationResult = {
          completenessScore: 50,
          qualityScores: { completeness: 50, clarity: 50, implementability: 50 },
          isValid: false,
          errors: [],
          warnings: [],
          followUpQuestions: []
        };

        delegator.addRound(session.id, 'Request 1', 'Response 1', [artifact1], validation1);

        // Round 2 - Enhanced artifact
        const artifact2 = createTestArtifact(
          ArtifactType.MERMAID_DIAGRAM, 
          'graph TD\nA[Component A]-->B[Component B]\nB-->C[Component C]\nA-->D[Database]'
        );
        const validation2: ValidationResult = {
          completenessScore: 85,
          qualityScores: { completeness: 85, clarity: 80, implementability: 75 },
          isValid: true,
          errors: [],
          warnings: [],
          followUpQuestions: []
        };

        delegator.addRound(session.id, 'Request 2', 'Response 2', [artifact2], validation2);

        const history = delegator.getArtifactVersionHistory(session.id, ArtifactType.MERMAID_DIAGRAM);
        
        // Metamorphic: More rounds produce more detailed artifacts
        expect(history[1].artifact.content.length).toBeGreaterThan(history[0].artifact.content.length);
        expect(history.length).toBe(2);
      });
    });
  });

  describe('Context Bundle Integration (Requirements 3.4, 3.5)', () => {
    it('should include context bundle in delegation request', () => {
      const contextBundle = createTestContextBundle([
        { path: 'src/main.ts', content: 'console.log("test")' },
        { path: 'src/utils.ts', content: 'export function helper() {}' }
      ]);

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test Problem',
        'Test description',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle
      );

      // Requirement 3.4: Include Context_Bundle with all relevant code and documentation
      expect(request.contextBundle).toBeDefined();
      expect(request.contextBundle.length).toBeGreaterThan(0);
    });

    it('should provide artifact structure guidance', () => {
      const contextBundle = createTestContextBundle();
      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test Problem',
        'Test description',
        DelegationType.ARCHITECTURE_DESIGN,
        contextBundle
      );

      // Requirement 3.5: Provide guidance on artifact structure
      for (const artifact of request.expectedArtifacts) {
        expect(artifact.structureGuidance).toBeDefined();
        expect(artifact.structureGuidance.length).toBeGreaterThan(0);
      }

      for (const formatReq of request.outputFormatRequirements) {
        expect(formatReq.instructions).toBeDefined();
        expect(formatReq.instructions.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Session Completion Detection (Requirement 9.5, 9.7)', () => {
    it('should detect completion when all criteria are met', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      // Add required artifacts with high quality scores
      const artifacts: ParsedArtifact[] = [
        createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'complete diagram'),
        createTestArtifact(ArtifactType.IMPLEMENTATION_GUIDE, 'complete guide')
      ];

      const validation: ValidationResult = {
        completenessScore: 90,
        qualityScores: {
          completeness: 90,
          clarity: 85,
          implementability: 80
        },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      // Requirement 9.5: Detect when Opus has provided sufficient detail
      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(true);
    });

    it('should recommend ending session when completion criteria met', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.API_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        createTestArtifact(ArtifactType.OPENAPI_SPEC, 'complete API spec'),
        createTestArtifact(ArtifactType.IMPLEMENTATION_GUIDE, 'complete guide')
      ];

      const validation: ValidationResult = {
        completenessScore: 95,
        qualityScores: {
          completeness: 95,
          clarity: 90,
          implementability: 85
        },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      // Should recommend ending the session
      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(true);
    });
  });

  describe('Automatic Follow-up Generation (Requirement 9.6)', () => {
    it('should automatically generate clarifying questions for incomplete artifacts', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'incomplete diagram')
      ];

      const validation: ValidationResult = {
        completenessScore: 60,
        qualityScores: { completeness: 60, clarity: 65, implementability: 55 },
        isValid: false,
        errors: [
          {
            type: 'missing_element',
            message: 'Missing component descriptions',
            location: 'artifact-1',
            severity: 'error'
          }
        ],
        warnings: [],
        followUpQuestions: [
          'What are the responsibilities of each component?',
          'How do components communicate?'
        ]
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);

      // Requirement 9.6: Automatically generate clarifying questions for next round
      const followUp = delegator.generateFollowUpRequest(session.id, validation, artifacts);

      expect(followUp.questionsToAddress.length).toBeGreaterThan(0);
      expect(followUp.questionsToAddress).toContain('What are the responsibilities of each component?');
      expect(followUp.questionsToAddress).toContain('How do components communicate?');
    });

    it('should generate refinement requests for quality issues', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.INTEGRATION_DESIGN
      );

      const artifacts: ParsedArtifact[] = [
        createTestArtifact(ArtifactType.OPENAPI_SPEC, 'basic API spec')
      ];

      const validation: ValidationResult = {
        completenessScore: 75,
        qualityScores: { completeness: 75, clarity: 60, implementability: 65 },
        isValid: false,
        errors: [
          {
            type: 'quality_issue',
            message: 'Missing error response schemas',
            location: 'artifact-1',
            severity: 'warning'
          }
        ],
        warnings: [],
        followUpQuestions: ['Please add error handling details', 'What retry mechanisms should be implemented?']
      };

      delegator.addRound(session.id, 'Request', 'Response', artifacts, validation);
      const followUp = delegator.generateFollowUpRequest(session.id, validation, artifacts);

      expect(followUp.questionsToAddress.length).toBeGreaterThan(0);
      expect(followUp.questionsToAddress).toContain('Please add error handling details');
    });
  });

  describe('Conversation Context Management (Requirement 9.2)', () => {
    it('should maintain conversation context across rounds', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      // Round 1
      const request1 = 'What are the main components? How do they interact?';
      const response1 = 'The system has three main components: A, B, and C...';
      const artifacts1 = [createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'basic diagram')];
      const validation1: ValidationResult = {
        completenessScore: 60,
        qualityScores: { completeness: 60, clarity: 60, implementability: 60 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, request1, response1, artifacts1, validation1);

      // Round 2
      const request2 = 'Please add more details about component interactions';
      const response2 = 'Component A communicates with B via REST API...';
      const artifacts2 = [createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'detailed diagram')];
      const validation2: ValidationResult = {
        completenessScore: 85,
        qualityScores: { completeness: 85, clarity: 80, implementability: 75 },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, request2, response2, artifacts2, validation2);

      // Requirement 9.2: Maintain conversation context across rounds
      const updatedSession = delegator.getSession(session.id);
      expect(updatedSession!.rounds.length).toBe(2);
      expect(updatedSession!.rounds[0].request).toBe(request1);
      expect(updatedSession!.rounds[0].response).toBe(response1);
      expect(updatedSession!.rounds[1].request).toBe(request2);
      expect(updatedSession!.rounds[1].response).toBe(response2);
    });
  });

  describe('Artifact Version History (Requirement 9.4)', () => {
    it('should track artifact refinements across rounds', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      // Round 1 - Initial version
      const artifact1 = createTestArtifact(ArtifactType.MERMAID_DIAGRAM, 'graph TD\nA-->B');
      const validation1: ValidationResult = {
        completenessScore: 50,
        qualityScores: { completeness: 50, clarity: 50, implementability: 50 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request 1', 'Response 1', [artifact1], validation1);

      // Round 2 - Refined version
      const artifact2 = createTestArtifact(
        ArtifactType.MERMAID_DIAGRAM, 
        'graph TD\nA[Frontend]-->B[API Gateway]\nB-->C[Database]'
      );
      const validation2: ValidationResult = {
        completenessScore: 80,
        qualityScores: { completeness: 80, clarity: 75, implementability: 70 },
        isValid: true,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      delegator.addRound(session.id, 'Request 2', 'Response 2', [artifact2], validation2);

      // Requirement 9.4: Track which artifacts were refined in each round
      const history = delegator.getArtifactVersionHistory(session.id, ArtifactType.MERMAID_DIAGRAM);
      
      expect(history.length).toBe(2);
      expect(history[0].version).toBe(1);
      expect(history[0].roundNumber).toBe(1);
      expect(history[1].version).toBe(2);
      expect(history[1].roundNumber).toBe(2);
      expect(history[1].changes).toBeDefined();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty context bundles gracefully', () => {
      const emptyContextBundle: ContextBundle = {
        problemTitle: 'Test',
        problemSummary: 'Test',
        codeSnippets: [],
        documentationExcerpts: [],
        constraints: [],
        contextManifest: { sources: [], totalFiles: 0, totalSize: 0 },
        totalSize: 0,
        compressionApplied: false
      };

      const request = delegator.generateDelegationRequest(
        'session-1',
        'Test Problem',
        'Test description',
        DelegationType.ARCHITECTURE_DESIGN,
        emptyContextBundle
      );

      expect(request).toBeDefined();
      expect(request.contextBundle).toBeDefined();
    });

    it('should handle sessions with no rounds', () => {
      const session = delegator.initializeSession(
        'Test',
        'Test problem',
        DelegationType.ARCHITECTURE_DESIGN
      );

      const history = delegator.getArtifactVersionHistory(session.id, ArtifactType.MERMAID_DIAGRAM);
      expect(history).toEqual([]);

      const validation: ValidationResult = {
        completenessScore: 0,
        qualityScores: { completeness: 0, clarity: 0, implementability: 0 },
        isValid: false,
        errors: [],
        warnings: [],
        followUpQuestions: []
      };

      const isComplete = delegator.detectSessionCompletion(session.id, validation);
      expect(isComplete).toBe(false);
    });

    it('should handle different delegation types appropriately', () => {
      const contextBundle = createTestContextBundle();
      const delegationTypesWithTemplates = [
        { type: DelegationType.ARCHITECTURE_DESIGN, template: 'federated_learning_architecture' },
        { type: DelegationType.INTEGRATION_DESIGN, template: 'pacs_integration_design' },
        { type: DelegationType.TEST_STRATEGY, template: 'property_based_test_suite' },
        { type: DelegationType.REFACTORING_ANALYSIS, template: 'refactoring_analysis' }
      ];

      for (const { type, template } of delegationTypesWithTemplates) {
        const session = delegator.initializeSession(
          `Test ${type}`,
          `Test problem for ${type}`,
          type
        );

        expect(session.problem.type).toBe(type);
        
        const request = delegator.generateDelegationRequest(
          session.id,
          `Test ${type}`,
          `Test description for ${type}`,
          type,
          contextBundle,
          template
        );

        expect(request.expectedArtifacts.length).toBeGreaterThan(0);
        expect(request.questionsToAddress.length).toBeGreaterThan(0);
      }
    });
  });

  // ========== HELPER FUNCTIONS ==========

  function createTestContextBundle(codeSnippets?: Array<{ path: string; content: string }>): ContextBundle {
    return {
      problemTitle: 'Test Problem',
      problemSummary: 'Test summary',
      codeSnippets: codeSnippets?.map(snippet => ({
        filePath: snippet.path,
        content: snippet.content,
        startLine: 1,
        endLine: 10,
        language: 'typescript'
      })) || [],
      documentationExcerpts: [
        {
          source: 'docs/architecture.md',
          content: 'System architecture overview',
          relevance: 'high'
        }
      ],
      constraints: ['Performance requirement: < 100ms response time', 'Security: OAuth 2.0 authentication'],
      contextManifest: {
        sources: [
          { path: 'src/main.ts', type: 'code', size: 1024, relevance: 'high' },
          { path: 'docs/architecture.md', type: 'documentation', size: 2048, relevance: 'medium' }
        ],
        totalFiles: 2,
        totalSize: 3072
      },
      totalSize: 3072,
      compressionApplied: false
    };
  }

  function createTestArtifact(type: ArtifactType, content: string): ParsedArtifact {
    return {
      id: `artifact-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      content,
      metadata: {
        sourceLocation: { start: 0, end: content.length },
        parseWarnings: [],
        extractedAt: new Date()
      }
    };
  }
});
