/**
 * Unit Tests for Core Data Structures
 * Implements Task 2.4 - Test type guards and data structure validation
 */

import { describe, it, expect } from 'vitest';
import {
  DelegationType,
  ComplexityLevel,
  ContextType,
  ArtifactType,
  isDelegationType,
  isComplexityLevel,
  isContextType,
  isArtifactType
} from './core.js';

describe('Core Data Structures', () => {
  describe('Type Guards', () => {
    describe('isDelegationType', () => {
      it('should return true for valid delegation types', () => {
        expect(isDelegationType('architecture_design')).toBe(true);
        expect(isDelegationType('api_design')).toBe(true);
        expect(isDelegationType('test_strategy')).toBe(true);
        expect(isDelegationType('integration_design')).toBe(true);
        expect(isDelegationType('refactoring_analysis')).toBe(true);
        expect(isDelegationType('formal_verification')).toBe(true);
      });

      it('should return false for invalid delegation types', () => {
        expect(isDelegationType('invalid_type')).toBe(false);
        expect(isDelegationType('')).toBe(false);
        expect(isDelegationType('ARCHITECTURE_DESIGN')).toBe(false); // Wrong case
      });
    });

    describe('isComplexityLevel', () => {
      it('should return true for valid complexity levels', () => {
        expect(isComplexityLevel('simple')).toBe(true);
        expect(isComplexityLevel('moderate')).toBe(true);
        expect(isComplexityLevel('complex')).toBe(true);
      });

      it('should return false for invalid complexity levels', () => {
        expect(isComplexityLevel('easy')).toBe(false);
        expect(isComplexityLevel('hard')).toBe(false);
        expect(isComplexityLevel('')).toBe(false);
      });
    });

    describe('isContextType', () => {
      it('should return true for valid context types', () => {
        expect(isContextType('code_snippets')).toBe(true);
        expect(isContextType('requirements_docs')).toBe(true);
        expect(isContextType('existing_designs')).toBe(true);
        expect(isContextType('constraints')).toBe(true);
        expect(isContextType('architecture_docs')).toBe(true);
        expect(isContextType('api_endpoints')).toBe(true);
        expect(isContextType('test_files')).toBe(true);
        expect(isContextType('external_interfaces')).toBe(true);
        expect(isContextType('dependency_graphs')).toBe(true);
      });

      it('should return false for invalid context types', () => {
        expect(isContextType('invalid_context')).toBe(false);
        expect(isContextType('')).toBe(false);
      });
    });

    describe('isArtifactType', () => {
      it('should return true for valid artifact types', () => {
        expect(isArtifactType('mermaid_diagram')).toBe(true);
        expect(isArtifactType('openapi_spec')).toBe(true);
        expect(isArtifactType('implementation_guide')).toBe(true);
        expect(isArtifactType('test_strategy')).toBe(true);
        expect(isArtifactType('code_snippet')).toBe(true);
      });

      it('should return false for invalid artifact types', () => {
        expect(isArtifactType('invalid_artifact')).toBe(false);
        expect(isArtifactType('')).toBe(false);
      });
    });
  });

  describe('Enum Values', () => {
    it('should have correct DelegationType values', () => {
      expect(DelegationType.ARCHITECTURE_DESIGN).toBe('architecture_design');
      expect(DelegationType.API_DESIGN).toBe('api_design');
      expect(DelegationType.TEST_STRATEGY).toBe('test_strategy');
      expect(DelegationType.INTEGRATION_DESIGN).toBe('integration_design');
      expect(DelegationType.REFACTORING_ANALYSIS).toBe('refactoring_analysis');
      expect(DelegationType.FORMAL_VERIFICATION).toBe('formal_verification');
    });

    it('should have correct ComplexityLevel values', () => {
      expect(ComplexityLevel.SIMPLE).toBe('simple');
      expect(ComplexityLevel.MODERATE).toBe('moderate');
      expect(ComplexityLevel.COMPLEX).toBe('complex');
    });

    it('should have correct ContextType values', () => {
      expect(ContextType.CODE_SNIPPETS).toBe('code_snippets');
      expect(ContextType.REQUIREMENTS_DOCS).toBe('requirements_docs');
      expect(ContextType.EXISTING_DESIGNS).toBe('existing_designs');
      expect(ContextType.CONSTRAINTS).toBe('constraints');
      expect(ContextType.ARCHITECTURE_DOCS).toBe('architecture_docs');
      expect(ContextType.API_ENDPOINTS).toBe('api_endpoints');
      expect(ContextType.TEST_FILES).toBe('test_files');
      expect(ContextType.EXTERNAL_INTERFACES).toBe('external_interfaces');
      expect(ContextType.DEPENDENCY_GRAPHS).toBe('dependency_graphs');
    });

    it('should have correct ArtifactType values', () => {
      expect(ArtifactType.MERMAID_DIAGRAM).toBe('mermaid_diagram');
      expect(ArtifactType.OPENAPI_SPEC).toBe('openapi_spec');
      expect(ArtifactType.IMPLEMENTATION_GUIDE).toBe('implementation_guide');
      expect(ArtifactType.TEST_STRATEGY).toBe('test_strategy');
      expect(ArtifactType.CODE_SNIPPET).toBe('code_snippet');
    });
  });

  describe('Data Structure Validation', () => {
    it('should validate ProblemClassification structure', () => {
      const classification = {
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        suitabilityScore: 85,
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.ARCHITECTURE_DOCS, ContextType.CODE_SNIPPETS],
        expectedArtifactTypes: [ArtifactType.MERMAID_DIAGRAM],
        estimatedRounds: 3,
        confidence: 90
      };

      expect(classification.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(classification.suitabilityScore).toBe(85);
      expect(classification.complexity).toBe(ComplexityLevel.COMPLEX);
      expect(classification.requiredContextTypes).toHaveLength(2);
      expect(classification.expectedArtifactTypes).toHaveLength(1);
      expect(classification.estimatedRounds).toBe(3);
      expect(classification.confidence).toBe(90);
    });

    it('should validate DelegationRecommendation structure', () => {
      const recommendation = {
        suitable: true,
        classification: {
          delegationType: DelegationType.API_DESIGN,
          suitabilityScore: 75,
          complexity: ComplexityLevel.MODERATE,
          requiredContextTypes: [ContextType.API_ENDPOINTS],
          expectedArtifactTypes: [ArtifactType.OPENAPI_SPEC],
          estimatedRounds: 2,
          confidence: 80
        },
        reasoning: 'Test reasoning',
        contextEstimate: {
          estimatedSize: 30000,
          extractionComplexity: ComplexityLevel.MODERATE
        }
      };

      expect(recommendation.suitable).toBe(true);
      expect(recommendation.classification.delegationType).toBe(DelegationType.API_DESIGN);
      expect(recommendation.reasoning).toBe('Test reasoning');
      expect(recommendation.contextEstimate.estimatedSize).toBe(30000);
    });

    it('should validate ParsedArtifact structure', () => {
      const artifact = {
        id: 'test-artifact-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\n  A --> B',
        metadata: {
          sourceLocation: { start: 0, end: 15 },
          parseWarnings: [],
          extractedAt: new Date()
        }
      };

      expect(artifact.id).toBe('test-artifact-1');
      expect(artifact.type).toBe(ArtifactType.MERMAID_DIAGRAM);
      expect(artifact.content).toBe('graph TD\n  A --> B');
      expect(artifact.metadata.sourceLocation.start).toBe(0);
      expect(artifact.metadata.sourceLocation.end).toBe(15);
      expect(artifact.metadata.parseWarnings).toHaveLength(0);
      expect(artifact.metadata.extractedAt).toBeInstanceOf(Date);
    });

    it('should validate DelegationSession structure', () => {
      const session = {
        id: 'session-123',
        createdAt: new Date(),
        updatedAt: new Date(),
        problem: {
          title: 'Test Problem',
          description: 'Test description',
          type: DelegationType.ARCHITECTURE_DESIGN,
          complexity: ComplexityLevel.MODERATE
        },
        rounds: [],
        finalArtifacts: [],
        metrics: {
          totalTime: 3600,
          contextSize: 25000,
          roundCount: 2,
          finalCompleteness: 85
        }
      };

      expect(session.id).toBe('session-123');
      expect(session.problem.title).toBe('Test Problem');
      expect(session.problem.type).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(session.metrics.totalTime).toBe(3600);
      expect(session.rounds).toHaveLength(0);
      expect(session.finalArtifacts).toHaveLength(0);
    });
  });
});