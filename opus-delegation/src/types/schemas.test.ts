import { describe, it, expect } from 'vitest';
import { DelegationType, ComplexityLevel, ContextType, ArtifactType } from './core';
import {
  ProblemClassificationSchema,
  CodeSnippetSchema,
  ContextBundleSchema,
  DelegationTemplateSchema,
  ValidationResultSchema,
  SessionMetricsSchema,
  ParsedArtifactSchema,
  DelegationRequestSchema,
  ImplementationGuideSchema,
  TestStrategySchema,
  isProblemClassification,
  isContextBundle,
  isDelegationTemplate,
  isValidationResult,
  isDelegationSession,
  isParsedArtifact,
  isDelegationRequest,
  isDelegationRound,
  isImplementationGuide,
  isMermaidDiagram,
  isOpenAPISpec,
  isTestStrategy,
} from './schemas';

describe('Schemas', () => {
  describe('ProblemClassificationSchema', () => {
    it('should validate a valid problem classification', () => {
      const validClassification = {
        suitable: true,
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.CODE_SNIPPETS, ContextType.ARCHITECTURE_DOCS],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM, ArtifactType.IMPLEMENTATION_GUIDE],
        suitabilityScore: 85,
        reasoning: 'Complex architectural problem requiring system design',
      };

      const result = ProblemClassificationSchema.safeParse(validClassification);
      expect(result.success).toBe(true);
    });

    it('should reject invalid suitability scores', () => {
      const invalidClassification = {
        suitable: true,
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.CODE_SNIPPETS],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM],
        suitabilityScore: 150, // Invalid: > 100
        reasoning: 'Test',
      };

      const result = ProblemClassificationSchema.safeParse(invalidClassification);
      expect(result.success).toBe(false);
    });

    it('should reject empty reasoning', () => {
      const invalidClassification = {
        suitable: true,
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.CODE_SNIPPETS],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM],
        suitabilityScore: 85,
        reasoning: '', // Invalid: empty string
      };

      const result = ProblemClassificationSchema.safeParse(invalidClassification);
      expect(result.success).toBe(false);
    });

    it('should accept optional alternative types', () => {
      const classificationWithAlternatives = {
        suitable: true,
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        alternativeTypes: [DelegationType.API_DESIGN, DelegationType.INTEGRATION_DESIGN],
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.CODE_SNIPPETS],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM],
        suitabilityScore: 85,
        reasoning: 'Valid reasoning with alternatives',
      };

      const result = ProblemClassificationSchema.safeParse(classificationWithAlternatives);
      expect(result.success).toBe(true);
    });
  });

  describe('CodeSnippetSchema', () => {
    it('should validate a valid code snippet', () => {
      const validSnippet = {
        filePath: 'src/main.ts',
        startLine: 1,
        endLine: 50,
        content: 'export class Main {}',
        language: 'typescript',
        relevance: 0.9,
      };

      const result = CodeSnippetSchema.safeParse(validSnippet);
      expect(result.success).toBe(true);
    });

    it('should reject invalid line numbers', () => {
      const invalidSnippet = {
        filePath: 'src/main.ts',
        startLine: 0, // Invalid: must be positive
        endLine: 50,
        content: 'export class Main {}',
        language: 'typescript',
        relevance: 0.9,
      };

      const result = CodeSnippetSchema.safeParse(invalidSnippet);
      expect(result.success).toBe(false);
    });

    it('should reject invalid relevance scores', () => {
      const invalidSnippet = {
        filePath: 'src/main.ts',
        startLine: 1,
        endLine: 50,
        content: 'export class Main {}',
        language: 'typescript',
        relevance: 1.5, // Invalid: > 1
      };

      const result = CodeSnippetSchema.safeParse(invalidSnippet);
      expect(result.success).toBe(false);
    });
  });

  describe('ContextBundleSchema', () => {
    it('should validate a valid context bundle', () => {
      const validBundle = {
        id: 'bundle-1',
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
            type: 'code' as const,
            size: 1200,
            relevance: 'high' as const,
          },
        ],
      };

      const result = ContextBundleSchema.safeParse(validBundle);
      expect(result.success).toBe(true);
    });

    it('should reject negative total size', () => {
      const invalidBundle = {
        id: 'bundle-1',
        problemDescription: 'Design a federated learning system',
        codeSnippets: [],
        documentation: [],
        constraints: [],
        totalSize: -100, // Invalid: negative
        truncated: false,
        manifest: [],
      };

      const result = ContextBundleSchema.safeParse(invalidBundle);
      expect(result.success).toBe(false);
    });
  });

  describe('DelegationTemplateSchema', () => {
    it('should validate a valid delegation template', () => {
      const validTemplate = {
        templateId: 'federated-learning-arch',
        name: 'Federated Learning Architecture',
        category: DelegationType.ARCHITECTURE_DESIGN,
        version: '1.0.0',
        parameters: [
          {
            name: 'system_name',
            required: true,
            type: 'string' as const,
            description: 'Name of the system',
          },
        ],
        contextRequirements: [ContextType.CODE_SNIPPETS, ContextType.ARCHITECTURE_DOCS],
        expectedArtifacts: [
          {
            type: ArtifactType.MERMAID_DIAGRAM,
            subtype: 'architecture',
          },
        ],
        promptTemplate: 'Design a federated learning system for {{system_name}}',
      };

      const result = DelegationTemplateSchema.safeParse(validTemplate);
      expect(result.success).toBe(true);
    });

    it('should reject empty template ID', () => {
      const invalidTemplate = {
        templateId: '', // Invalid: empty
        name: 'Test Template',
        category: DelegationType.ARCHITECTURE_DESIGN,
        version: '1.0.0',
        parameters: [],
        contextRequirements: [],
        expectedArtifacts: [],
        promptTemplate: 'Test prompt',
      };

      const result = DelegationTemplateSchema.safeParse(invalidTemplate);
      expect(result.success).toBe(false);
    });
  });

  describe('ValidationResultSchema', () => {
    it('should validate a valid validation result', () => {
      const validResult = {
        valid: true,
        completenessScore: 85,
        qualityScores: {
          completeness: 90,
          clarity: 80,
          implementability: 85,
        },
        missingElements: [],
        errors: [],
        warnings: ['Minor formatting issue'],
      };

      const result = ValidationResultSchema.safeParse(validResult);
      expect(result.success).toBe(true);
    });

    it('should reject invalid quality scores', () => {
      const invalidResult = {
        valid: true,
        completenessScore: 85,
        qualityScores: {
          completeness: 150, // Invalid: > 100
          clarity: 80,
          implementability: 85,
        },
        missingElements: [],
        errors: [],
        warnings: [],
      };

      const result = ValidationResultSchema.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });
  });

  describe('SessionMetricsSchema', () => {
    it('should validate valid session metrics', () => {
      const validMetrics = {
        totalTime: 300000,
        contextSize: 50000,
        roundCount: 3,
        finalCompleteness: 95,
        estimatedCost: 2.5,
      };

      const result = SessionMetricsSchema.safeParse(validMetrics);
      expect(result.success).toBe(true);
    });

    it('should reject negative values', () => {
      const invalidMetrics = {
        totalTime: -100, // Invalid: negative
        contextSize: 50000,
        roundCount: 3,
        finalCompleteness: 95,
      };

      const result = SessionMetricsSchema.safeParse(invalidMetrics);
      expect(result.success).toBe(false);
    });
  });

  describe('ParsedArtifactSchema', () => {
    it('should validate a valid parsed artifact', () => {
      const validArtifact = {
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

      const result = ParsedArtifactSchema.safeParse(validArtifact);
      expect(result.success).toBe(true);
    });

    it('should reject invalid source location', () => {
      const invalidArtifact = {
        id: 'artifact-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\n  A --> B',
        metadata: {
          sourceLocation: { start: -1, end: 100 }, // Invalid: negative start
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      const result = ParsedArtifactSchema.safeParse(invalidArtifact);
      expect(result.success).toBe(false);
    });
  });

  describe('DelegationRequestSchema', () => {
    it('should validate a valid delegation request', () => {
      const validRequest = {
        id: 'request-1',
        sessionId: 'session-1',
        roundNumber: 1,
        title: 'Design Architecture',
        description: 'Design a federated learning architecture',
        objectives: ['Create scalable system', 'Ensure privacy'],
        constraints: ['Must use existing infrastructure'],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM, ArtifactType.IMPLEMENTATION_GUIDE],
        formatRequirements: 'Use Mermaid for diagrams',
        contextBundle: {
          id: 'bundle-1',
          problemDescription: 'Test problem',
          codeSnippets: [],
          documentation: [],
          constraints: [],
          totalSize: 1000,
          truncated: false,
          manifest: [],
        },
        formattedText: 'Formatted delegation request text',
        createdAt: new Date(),
      };

      const result = DelegationRequestSchema.safeParse(validRequest);
      expect(result.success).toBe(true);
    });

    it('should reject invalid round number', () => {
      const invalidRequest = {
        id: 'request-1',
        sessionId: 'session-1',
        roundNumber: 0, // Invalid: must be positive
        title: 'Design Architecture',
        description: 'Design a federated learning architecture',
        objectives: [],
        constraints: [],
        expectedArtifacts: [],
        formatRequirements: 'Use Mermaid for diagrams',
        contextBundle: {
          id: 'bundle-1',
          problemDescription: 'Test problem',
          codeSnippets: [],
          documentation: [],
          constraints: [],
          totalSize: 1000,
          truncated: false,
          manifest: [],
        },
        formattedText: 'Formatted delegation request text',
        createdAt: new Date(),
      };

      const result = DelegationRequestSchema.safeParse(invalidRequest);
      expect(result.success).toBe(false);
    });
  });

  describe('ImplementationGuideSchema', () => {
    it('should validate a valid implementation guide', () => {
      const validGuide = {
        title: 'Federated Learning Implementation',
        prerequisites: ['Python 3.8+', 'Docker'],
        phases: [
          {
            name: 'Foundation',
            complexity: ComplexityLevel.SIMPLE,
            steps: [
              {
                id: 'step-1',
                number: '1.1',
                action: 'Create base interfaces',
                dependencies: [],
                complexity: ComplexityLevel.SIMPLE,
              },
            ],
          },
        ],
        risks: [
          {
            risk: 'Performance bottleneck',
            mitigation: 'Use caching',
            owner: 'Backend team',
          },
        ],
      };

      const result = ImplementationGuideSchema.safeParse(validGuide);
      expect(result.success).toBe(true);
    });

    it('should reject empty title', () => {
      const invalidGuide = {
        title: '', // Invalid: empty
        prerequisites: [],
        phases: [],
      };

      const result = ImplementationGuideSchema.safeParse(invalidGuide);
      expect(result.success).toBe(false);
    });
  });

  describe('TestStrategySchema', () => {
    it('should validate a valid test strategy', () => {
      const validStrategy = {
        coverageTargets: {
          unit: 90,
          integration: 80,
          e2e: 70,
        },
        propertyTests: [
          {
            name: 'Aggregation correctness',
            property: 'Aggregated weights preserve model accuracy',
            generator: 'random_weights',
          },
        ],
        edgeCases: ['Empty dataset', 'Single client'],
        testDataRequirements: ['Synthetic federated data', 'Privacy-preserving test cases'],
      };

      const result = TestStrategySchema.safeParse(validStrategy);
      expect(result.success).toBe(true);
    });

    it('should reject invalid coverage targets', () => {
      const invalidStrategy = {
        coverageTargets: {
          unit: 150, // Invalid: > 100
        },
        propertyTests: [],
        edgeCases: [],
        testDataRequirements: [],
      };

      const result = TestStrategySchema.safeParse(invalidStrategy);
      expect(result.success).toBe(false);
    });
  });

  describe('Type Guards', () => {
    it('should correctly identify valid problem classifications', () => {
      const validClassification = {
        suitable: true,
        delegationType: DelegationType.ARCHITECTURE_DESIGN,
        complexity: ComplexityLevel.COMPLEX,
        requiredContextTypes: [ContextType.CODE_SNIPPETS],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM],
        suitabilityScore: 85,
        reasoning: 'Valid reasoning',
      };

      expect(isProblemClassification(validClassification)).toBe(true);
      expect(isProblemClassification({})).toBe(false);
      expect(isProblemClassification(null)).toBe(false);
    });

    it('should correctly identify valid context bundles', () => {
      const validBundle = {
        id: 'bundle-1',
        problemDescription: 'Test problem',
        codeSnippets: [],
        documentation: [],
        constraints: [],
        totalSize: 1000,
        truncated: false,
        manifest: [],
      };

      expect(isContextBundle(validBundle)).toBe(true);
      expect(isContextBundle({})).toBe(false);
      expect(isContextBundle(null)).toBe(false);
    });

    it('should correctly identify valid delegation templates', () => {
      const validTemplate = {
        templateId: 'test-template',
        name: 'Test Template',
        category: DelegationType.ARCHITECTURE_DESIGN,
        version: '1.0.0',
        parameters: [],
        contextRequirements: [],
        expectedArtifacts: [],
        promptTemplate: 'Test prompt',
      };

      expect(isDelegationTemplate(validTemplate)).toBe(true);
      expect(isDelegationTemplate({})).toBe(false);
      expect(isDelegationTemplate(null)).toBe(false);
    });

    it('should correctly identify valid validation results', () => {
      const validResult = {
        valid: true,
        completenessScore: 85,
        qualityScores: {
          completeness: 90,
          clarity: 80,
          implementability: 85,
        },
        missingElements: [],
        errors: [],
        warnings: [],
      };

      expect(isValidationResult(validResult)).toBe(true);
      expect(isValidationResult({})).toBe(false);
      expect(isValidationResult(null)).toBe(false);
    });

    it('should correctly identify valid delegation sessions', () => {
      const validSession = {
        id: 'session-1',
        createdAt: new Date(),
        updatedAt: new Date(),
        problem: {
          title: 'Test Problem',
          description: 'Test description',
          type: DelegationType.ARCHITECTURE_DESIGN,
          complexity: ComplexityLevel.SIMPLE,
        },
        rounds: [],
        finalArtifacts: [],
        metrics: {
          totalTime: 1000,
          contextSize: 5000,
          roundCount: 1,
          finalCompleteness: 85,
        },
        status: 'active' as const,
      };

      expect(isDelegationSession(validSession)).toBe(true);
      expect(isDelegationSession({})).toBe(false);
      expect(isDelegationSession(null)).toBe(false);
    });

    it('should correctly identify valid parsed artifacts', () => {
      const validArtifact = {
        id: 'artifact-1',
        type: ArtifactType.MERMAID_DIAGRAM,
        content: 'graph TD\n  A --> B',
        metadata: {
          sourceLocation: { start: 0, end: 100 },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      };

      expect(isParsedArtifact(validArtifact)).toBe(true);
      expect(isParsedArtifact({})).toBe(false);
      expect(isParsedArtifact(null)).toBe(false);
    });

    it('should correctly identify valid delegation requests', () => {
      const validRequest = {
        id: 'request-1',
        sessionId: 'session-1',
        roundNumber: 1,
        title: 'Test Request',
        description: 'Test description',
        objectives: ['Objective 1'],
        constraints: ['Constraint 1'],
        expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM],
        formatRequirements: 'Use Mermaid',
        contextBundle: {
          id: 'bundle-1',
          problemDescription: 'Test problem',
          codeSnippets: [],
          documentation: [],
          constraints: [],
          totalSize: 1000,
          truncated: false,
          manifest: [],
        },
        formattedText: 'Formatted text',
        createdAt: new Date(),
      };

      expect(isDelegationRequest(validRequest)).toBe(true);
      expect(isDelegationRequest({})).toBe(false);
      expect(isDelegationRequest(null)).toBe(false);
    });

    it('should correctly identify valid delegation rounds', () => {
      const validRound = {
        roundNumber: 1,
        request: {
          id: 'request-1',
          sessionId: 'session-1',
          roundNumber: 1,
          title: 'Test Request',
          description: 'Test description',
          objectives: ['Objective 1'],
          constraints: ['Constraint 1'],
          expectedArtifacts: [ArtifactType.MERMAID_DIAGRAM],
          formatRequirements: 'Use Mermaid',
          contextBundle: {
            id: 'bundle-1',
            problemDescription: 'Test problem',
            codeSnippets: [],
            documentation: [],
            constraints: [],
            totalSize: 1000,
            truncated: false,
            manifest: [],
          },
          formattedText: 'Formatted text',
          createdAt: new Date(),
        },
        artifacts: [],
        timestamp: new Date(),
      };

      expect(isDelegationRound(validRound)).toBe(true);
      expect(isDelegationRound({})).toBe(false);
      expect(isDelegationRound(null)).toBe(false);
    });

    it('should correctly identify valid implementation guides', () => {
      const validGuide = {
        title: 'Test Guide',
        prerequisites: ['Prerequisite 1'],
        phases: [
          {
            name: 'Phase 1',
            complexity: ComplexityLevel.SIMPLE,
            steps: [
              {
                id: 'step-1',
                number: '1.1',
                action: 'Do something',
                dependencies: [],
                complexity: ComplexityLevel.SIMPLE,
              },
            ],
          },
        ],
      };

      expect(isImplementationGuide(validGuide)).toBe(true);
      expect(isImplementationGuide({})).toBe(false);
      expect(isImplementationGuide(null)).toBe(false);
    });

    it('should correctly identify valid Mermaid diagrams', () => {
      const validDiagram = {
        diagramType: 'graph',
        syntax: 'graph TD\n  A --> B',
        valid: true,
      };

      expect(isMermaidDiagram(validDiagram)).toBe(true);
      expect(isMermaidDiagram({})).toBe(false);
      expect(isMermaidDiagram(null)).toBe(false);
    });

    it('should correctly identify valid OpenAPI specs', () => {
      const validSpec = {
        version: '3.0.0',
        spec: {
          openapi: '3.0.0',
          info: { title: 'Test API', version: '1.0.0' },
          paths: {},
        },
        valid: true,
      };

      expect(isOpenAPISpec(validSpec)).toBe(true);
      expect(isOpenAPISpec({})).toBe(false);
      expect(isOpenAPISpec(null)).toBe(false);
    });

    it('should correctly identify valid test strategies', () => {
      const validStrategy = {
        coverageTargets: {
          unit: 90,
        },
        propertyTests: [
          {
            name: 'Test property',
            property: 'Property description',
          },
        ],
        edgeCases: ['Edge case 1'],
        testDataRequirements: ['Requirement 1'],
      };

      expect(isTestStrategy(validStrategy)).toBe(true);
      expect(isTestStrategy({})).toBe(false);
      expect(isTestStrategy(null)).toBe(false);
    });
  });
});
