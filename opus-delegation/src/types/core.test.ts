import { describe, it, expect } from 'vitest';
import {
  DelegationType,
  ComplexityLevel,
  ContextType,
  ArtifactType,
  type ProblemClassification,
  type ContextBundle,
  type DelegationSession,
  type ParsedArtifact,
} from './core';

describe('Core Types', () => {
  describe('Enums', () => {
    it('should have all delegation types defined', () => {
      expect(DelegationType.ARCHITECTURE_DESIGN).toBe('architecture_design');
      expect(DelegationType.API_DESIGN).toBe('api_design');
      expect(DelegationType.TEST_STRATEGY).toBe('test_strategy');
      expect(DelegationType.INTEGRATION_DESIGN).toBe('integration_design');
      expect(DelegationType.REFACTORING_ANALYSIS).toBe('refactoring_analysis');
      expect(DelegationType.FORMAL_VERIFICATION).toBe('formal_verification');
    });

    it('should have all complexity levels defined', () => {
      expect(ComplexityLevel.SIMPLE).toBe('simple');
      expect(ComplexityLevel.MODERATE).toBe('moderate');
      expect(ComplexityLevel.COMPLEX).toBe('complex');
    });

    it('should have all context types defined', () => {
      expect(ContextType.CODE_SNIPPETS).toBe('code_snippets');
      expect(ContextType.REQUIREMENTS_DOCS).toBe('requirements_docs');
      expect(ContextType.EXISTING_DESIGNS).toBe('existing_designs');
      expect(ContextType.CONSTRAINTS).toBe('constraints');
    });

    it('should have all artifact types defined', () => {
      expect(ArtifactType.MERMAID_DIAGRAM).toBe('mermaid_diagram');
      expect(ArtifactType.OPENAPI_SPEC).toBe('openapi_spec');
      expect(ArtifactType.IMPLEMENTATION_GUIDE).toBe('implementation_guide');
      expect(ArtifactType.TEST_STRATEGY).toBe('test_strategy');
    });
  });

  describe('Type Guards', () => {
    it('should validate ProblemClassification structure', () => {
      const classification: ProblemClassification = {
        suitable: true,
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.CODE_SNIPPETS, ContextType.ARCHITECTURE_DOCS],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM, ArtifactType.IMPLEMENTATION_GUIDE],
        suitabilityScore: 85,
        reasoning: 'Complex architectural problem requiring system design',
      };

      expect(classification.suitable).toBe(true);
      expect(classification.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(classification.suitabilityScore).toBeGreaterThanOrEqual(0);
      expect(classification.suitabilityScore).toBeLessThanOrEqual(100);
    });

    it('should validate ContextBundle structure', () => {
      const bundle: ContextBundle = {
        id: 'test-bundle-1',
        problemDescription: 'Design a federated learning system',
        codeSnippets: [
          {
            filePath: 'src/ml/model.py',
            startLine: 1,
            endLine: 50,
            content: 'class Model: pass',
            language: 'python',
            relevance: 0.9,
          },
        ],
        documentation: [],
        constraints: ['Must support differential privacy'],
        totalSize: 5000,
        truncated: false,
        manifest: [
          {
            source: 'src/ml/model.py',
            type: 'code',
            size: 1200,
            relevance: 'high',
          },
        ],
      };

      expect(bundle.id).toBe('test-bundle-1');
      expect(bundle.codeSnippets).toHaveLength(1);
      expect(bundle.totalSize).toBeGreaterThan(0);
      expect(bundle.truncated).toBe(false);
    });

    it('should validate DelegationSession structure', () => {
      const now = new Date();
      const session: DelegationSession = {
        id: 'session-1',
        createdAt: now,
        updatedAt: now,
        problem: {
          title: 'Federated Learning Architecture',
          description: 'Design a federated learning system',
          type: DelegationType.ARCHITECTURE_DESIGN,
          complexity: ComplexityLevel.COMPLEX,
        },
        rounds: [],
        finalArtifacts: [],
        metrics: {
          totalTime: 0,
          contextSize: 0,
          roundCount: 0,
          finalCompleteness: 0,
        },
        status: 'active',
      };

      expect(session.id).toBe('session-1');
      expect(session.problem.type).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(session.status).toBe('active');
      expect(session.rounds).toHaveLength(0);
    });
  });

  describe('Data Validation', () => {
    it('should ensure suitability scores are in valid range', () => {
      const validScores = [0, 50, 100];
      validScores.forEach((score) => {
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(100);
      });
    });

    it('should ensure relevance scores are in valid range', () => {
      const validRelevance = [0, 0.5, 1.0];
      validRelevance.forEach((relevance) => {
        expect(relevance).toBeGreaterThanOrEqual(0);
        expect(relevance).toBeLessThanOrEqual(1);
      });
    });

    it('should ensure completeness scores are in valid range', () => {
      const validScores = [0, 50, 100];
      validScores.forEach((score) => {
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(100);
      });
    });

    it('should validate artifact type discriminators', () => {
      const mermaidArtifact: ParsedArtifact = {
        id: 'artifact-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\n  A --> B',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: {
          diagramType: 'graph',
          syntax: 'graph TD\n  A --> B',
          valid: true,
        },
      };

      const openApiArtifact: ParsedArtifact = {
        id: 'artifact-2',
        type: ArtifactType.OPENAPI_SPEC,
        content: 'openapi: 3.0.0\ninfo:\n  title: Test API\n  version: 1.0.0',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
        structured: {
          version: '3.0.0',
          spec: { openapi: '3.0.0', info: { title: 'Test API', version: '1.0.0' } },
          valid: true,
        },
      };

      expect(mermaidArtifact.type).toBe(ArtifactType.MERMAID_DIAGRAM);
      expect(openApiArtifact.type).toBe(ArtifactType.OPENAPI_SPEC);
      
      // Type discriminator should work
      if (mermaidArtifact.structured && 'diagramType' in mermaidArtifact.structured) {
        expect(mermaidArtifact.structured.diagramType).toBe('graph');
      }
      
      if (openApiArtifact.structured && 'version' in openApiArtifact.structured) {
        expect(openApiArtifact.structured.version).toBe('3.0.0');
      }
    });

    it('should validate delegation type enum values', () => {
      const allTypes = Object.values(DelegationType);
      expect(allTypes).toContain('architecture_design');
      expect(allTypes).toContain('api_design');
      expect(allTypes).toContain('test_strategy');
      expect(allTypes).toContain('integration_design');
      expect(allTypes).toContain('refactoring_analysis');
      expect(allTypes).toContain('formal_verification');
      expect(allTypes).toHaveLength(6);
    });

    it('should validate complexity level enum values', () => {
      const allLevels = Object.values(ComplexityLevel);
      expect(allLevels).toContain('simple');
      expect(allLevels).toContain('moderate');
      expect(allLevels).toContain('complex');
      expect(allLevels).toHaveLength(3);
    });

    it('should validate context type enum values', () => {
      const allContextTypes = Object.values(ContextType);
      expect(allContextTypes).toContain('code_snippets');
      expect(allContextTypes).toContain('requirements_docs');
      expect(allContextTypes).toContain('existing_designs');
      expect(allContextTypes).toContain('constraints');
      expect(allContextTypes).toContain('architecture_docs');
      expect(allContextTypes).toContain('api_definitions');
      expect(allContextTypes).toContain('test_files');
      expect(allContextTypes).toContain('config_files');
      expect(allContextTypes).toContain('protocol_specs');
      expect(allContextTypes).toContain('dependency_graphs');
      expect(allContextTypes).toHaveLength(10);
    });

    it('should validate artifact type enum values', () => {
      const allArtifactTypes = Object.values(ArtifactType);
      expect(allArtifactTypes).toContain('mermaid_diagram');
      expect(allArtifactTypes).toContain('openapi_spec');
      expect(allArtifactTypes).toContain('implementation_guide');
      expect(allArtifactTypes).toContain('test_strategy');
      expect(allArtifactTypes).toContain('code_snippet');
      expect(allArtifactTypes).toContain('requirements');
      expect(allArtifactTypes).toContain('design_doc');
      expect(allArtifactTypes).toHaveLength(7);
    });
  });
});
