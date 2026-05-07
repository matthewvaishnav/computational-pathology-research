import { describe, it, expect, beforeEach } from 'vitest';
import { ProblemClassifier, classifyProblem, generateExtractionStrategy } from './problem-classifier';
import { DelegationType, ComplexityLevel, ContextType, ArtifactType } from '../types';

describe('ProblemClassifier', () => {
  let classifier: ProblemClassifier;

  beforeEach(() => {
    classifier = new ProblemClassifier();
  });

  describe('Architecture Design Classification', () => {
    it('should classify architecture problems correctly', () => {
      const result = classifier.classifyProblem(
        'Design a distributed microservices architecture for a federated learning system with scalability and performance requirements'
      );

      expect(result.suitable).toBe(true);
      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(result.complexity).toBe(ComplexityLevel.COMPLEX);
      expect(result.suitabilityScore).toBeGreaterThan(60);
      expect(result.requiredContextTypes).toContain(ContextType.ARCHITECTURE_DOCS);
      expect(result.expectedArtifacts).toContain(ArtifactType.MERMAID_DIAGRAM);
      expect(result.reasoning).toContain('architecture');
    });

    it('should detect architectural scope indicators', () => {
      const result = classifier.classifyProblem(
        'Create a system design for multi-tier application with load balancer and database design'
      );

      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(result.suitabilityScore).toBeGreaterThan(50);
      expect(result.reasoning).toContain('architectural scope');
    });
  });

  describe('API Design Classification', () => {
    it('should classify API design problems correctly', () => {
      const result = classifier.classifyProblem(
        'Design REST API endpoints for user management with GraphQL integration'
      );

      expect(result.suitable).toBe(true);
      expect(result.delegationType).toBe(DelegationType.API_DESIGN);
      expect(result.requiredContextTypes).toContain(ContextType.API_DEFINITIONS);
      expect(result.expectedArtifacts).toContain(ArtifactType.OPENAPI_SPEC);
    });

    it('should detect API-specific keywords', () => {
      const result = classifier.classifyProblem('Create endpoint specifications for REST API');

      expect(result.delegationType).toBe(DelegationType.API_DESIGN);
    });
  });

  describe('Test Strategy Classification', () => {
    it('should classify test strategy problems correctly', () => {
      const result = classifier.classifyProblem(
        'Design comprehensive testing strategy with property-based tests and unit test coverage'
      );

      expect(result.suitable).toBe(true);
      expect(result.delegationType).toBe(DelegationType.TEST_STRATEGY);
      expect(result.requiredContextTypes).toContain(ContextType.TEST_FILES);
      expect(result.expectedArtifacts).toContain(ArtifactType.TEST_STRATEGY);
    });

    it('should detect testing keywords', () => {
      const result = classifier.classifyProblem('Create integration test suite for the system');

      expect(result.delegationType).toBe(DelegationType.TEST_STRATEGY);
    });
  });

  describe('Integration Design Classification', () => {
    it('should classify integration problems correctly', () => {
      const result = classifier.classifyProblem(
        'Design integration with external API using webhook protocols and real-time messaging'
      );

      expect(result.suitable).toBe(true);
      expect(result.delegationType).toBe(DelegationType.INTEGRATION_DESIGN);
      expect(result.requiredContextTypes).toContain(ContextType.PROTOCOL_SPECS);
      expect(result.expectedArtifacts).toContain(ArtifactType.MERMAID_DIAGRAM);
    });

    it('should detect integration complexity indicators', () => {
      const result = classifier.classifyProblem(
        'Implement third-party integration with event-driven messaging and streaming protocols'
      );

      expect(result.delegationType).toBe(DelegationType.INTEGRATION_DESIGN);
      expect(result.reasoning).toContain('integration complexity');
    });
  });

  describe('Refactoring Analysis Classification', () => {
    it('should classify refactoring problems correctly', () => {
      const result = classifier.classifyProblem(
        'Analyze code smells and technical debt for refactoring legacy system'
      );

      expect(result.suitable).toBe(true);
      expect(result.delegationType).toBe(DelegationType.REFACTORING_ANALYSIS);
      expect(result.requiredContextTypes).toContain(ContextType.DEPENDENCY_GRAPHS);
      expect(result.expectedArtifacts).toContain(ArtifactType.IMPLEMENTATION_GUIDE);
    });

    it('should detect refactoring keywords', () => {
      const result = classifier.classifyProblem('Refactor the codebase to eliminate technical debt');

      expect(result.delegationType).toBe(DelegationType.REFACTORING_ANALYSIS);
    });
  });

  describe('Formal Verification Classification', () => {
    it('should classify formal verification problems correctly', () => {
      const result = classifier.classifyProblem(
        'Prove correctness of algorithm with mathematical verification and invariant analysis'
      );

      expect(result.suitable).toBe(true);
      expect(result.delegationType).toBe(DelegationType.FORMAL_VERIFICATION);
      expect(result.requiredContextTypes).toContain(ContextType.REQUIREMENTS_DOCS);
      expect(result.expectedArtifacts).toContain(ArtifactType.REQUIREMENTS);
    });

    it('should detect formal reasoning indicators', () => {
      const result = classifier.classifyProblem(
        'Design verification strategy with proof of correctness and invariant checking'
      );

      expect(result.delegationType).toBe(DelegationType.FORMAL_VERIFICATION);
      expect(result.reasoning).toContain('formal reasoning');
    });
  });

  describe('Complexity Assessment', () => {
    it('should assess simple complexity correctly', () => {
      const result = classifier.classifyProblem('Create a basic function');

      expect(result.complexity).toBe(ComplexityLevel.SIMPLE);
      expect(result.suitabilityScore).toBeLessThan(60);
      expect(result.suitable).toBe(false);
    });

    it('should assess moderate complexity correctly', () => {
      const result = classifier.classifyProblem(
        'Design API endpoints with some integration requirements'
      );

      expect(result.complexity).toBe(ComplexityLevel.MODERATE);
      expect(result.suitabilityScore).toBeGreaterThanOrEqual(60);
      expect(result.suitabilityScore).toBeLessThan(80);
    });

    it('should assess complex problems correctly', () => {
      const result = classifier.classifyProblem(
        'Design distributed architecture with formal verification, novel machine learning algorithms, and complex integration patterns'
      );

      expect(result.complexity).toBe(ComplexityLevel.COMPLEX);
      expect(result.suitabilityScore).toBeGreaterThanOrEqual(80);
    });
  });

  describe('Alternative Types Detection', () => {
    it('should detect alternative delegation types', () => {
      const result = classifier.classifyProblem(
        'Design architecture with comprehensive API endpoints and testing strategy'
      );

      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(result.alternativeTypes).toBeDefined();
      expect(result.alternativeTypes).toContain(DelegationType.API_DESIGN);
    });

    it('should limit alternative types to 2', () => {
      const result = classifier.classifyProblem(
        'Design architecture with API endpoints, testing strategy, integration patterns, and refactoring analysis'
      );

      expect(result.alternativeTypes?.length).toBeLessThanOrEqual(2);
    });
  });

  describe('Suitability Scoring', () => {
    it('should return scores in valid range (0-100)', () => {
      const testCases = [
        'Simple task',
        'Moderate complexity with some architecture',
        'Complex distributed system with formal verification and novel algorithms'
      ];

      testCases.forEach(description => {
        const result = classifier.classifyProblem(description);
        expect(result.suitabilityScore).toBeGreaterThanOrEqual(0);
        expect(result.suitabilityScore).toBeLessThanOrEqual(100);
      });
    });

    it('should have higher scores for more complex problems', () => {
      const simple = classifier.classifyProblem('Create a basic function');
      const complex = classifier.classifyProblem(
        'Design distributed federated learning architecture with formal verification'
      );

      expect(complex.suitabilityScore).toBeGreaterThan(simple.suitabilityScore);
    });
  });

  describe('Context Requirements', () => {
    it('should provide appropriate context types for each delegation type', () => {
      const architectureResult = classifier.classifyProblem('Design system architecture');
      expect(architectureResult.requiredContextTypes).toContain(ContextType.ARCHITECTURE_DOCS);
      expect(architectureResult.requiredContextTypes).toContain(ContextType.CODE_SNIPPETS);

      const apiResult = classifier.classifyProblem('Design REST API');
      expect(apiResult.requiredContextTypes).toContain(ContextType.API_DEFINITIONS);

      const testResult = classifier.classifyProblem('Create testing strategy');
      expect(testResult.requiredContextTypes).toContain(ContextType.TEST_FILES);
    });

    it('should provide non-empty context requirements', () => {
      const result = classifier.classifyProblem('Design architecture');
      expect(result.requiredContextTypes.length).toBeGreaterThan(0);
    });
  });

  describe('Expected Artifacts', () => {
    it('should provide appropriate artifacts for each delegation type', () => {
      const architectureResult = classifier.classifyProblem('Design system architecture');
      expect(architectureResult.expectedArtifacts).toContain(ArtifactType.MERMAID_DIAGRAM);

      const apiResult = classifier.classifyProblem('Design REST API');
      expect(apiResult.expectedArtifacts).toContain(ArtifactType.OPENAPI_SPEC);

      const testResult = classifier.classifyProblem('Create testing strategy');
      expect(testResult.expectedArtifacts).toContain(ArtifactType.TEST_STRATEGY);
    });

    it('should provide non-empty artifact expectations', () => {
      const result = classifier.classifyProblem('Design architecture');
      expect(result.expectedArtifacts.length).toBeGreaterThan(0);
    });
  });

  describe('Reasoning Generation', () => {
    it('should provide clear reasoning for suitable problems', () => {
      const result = classifier.classifyProblem(
        'Design distributed architecture with formal verification'
      );

      expect(result.reasoning).toBeTruthy();
      expect(result.reasoning).toContain('suitable');
      expect(result.reasoning.length).toBeGreaterThan(20);
    });

    it('should provide clear reasoning for unsuitable problems', () => {
      const result = classifier.classifyProblem('Simple task');

      expect(result.reasoning).toBeTruthy();
      expect(result.reasoning).toContain('low complexity');
      expect(result.reasoning).toContain('manual implementation');
    });

    it('should mention complexity indicators in reasoning', () => {
      const result = classifier.classifyProblem(
        'Design architecture with formal verification requirements'
      );

      expect(result.reasoning).toMatch(/architectural scope|formal reasoning/);
    });
  });

  describe('Title Integration', () => {
    it('should consider title in classification', () => {
      const withTitle = classifier.classifyProblem(
        'Create a simple function',
        'Federated Learning Architecture Design'
      );
      const withoutTitle = classifier.classifyProblem('Create a simple function');

      expect(withTitle.suitabilityScore).toBeGreaterThan(withoutTitle.suitabilityScore);
      expect(withTitle.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
    });
  });

  describe('Custom Configuration', () => {
    it('should accept custom minimum suitability score', () => {
      const strictClassifier = new ProblemClassifier({ minSuitabilityScore: 80 });
      const lenientClassifier = new ProblemClassifier({ minSuitabilityScore: 40 });

      const problem = 'Design API with moderate complexity';
      const strictResult = strictClassifier.classifyProblem(problem);
      const lenientResult = lenientClassifier.classifyProblem(problem);

      // Same score, different suitability due to threshold
      expect(strictResult.suitabilityScore).toBe(lenientResult.suitabilityScore);
      expect(strictResult.suitable).not.toBe(lenientResult.suitable);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty problem description', () => {
      const result = classifier.classifyProblem('');

      expect(result.suitable).toBe(false);
      expect(result.suitabilityScore).toBe(0);
      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN); // Default fallback
    });

    it('should handle very long problem descriptions', () => {
      const longDescription = 'architecture '.repeat(1000) + 'design system';
      const result = classifier.classifyProblem(longDescription);

      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(result.suitabilityScore).toBeGreaterThan(0);
    });

    it('should handle special characters and numbers', () => {
      const result = classifier.classifyProblem(
        'Design API v2.0 with 99% uptime & real-time features @scale'
      );

      expect(result.delegationType).toBe(DelegationType.API_DESIGN);
      expect(result.suitable).toBe(true);
    });
  });
});

describe('Extraction Strategy Generation', () => {
  let classifier: ProblemClassifier;

  beforeEach(() => {
    classifier = new ProblemClassifier();
  });

  describe('Strategy by Delegation Type', () => {
    it('should generate appropriate strategy for architecture design', () => {
      const strategy = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.MODERATE
      );

      expect(strategy.primaryPatterns).toContain('**/*.md');
      expect(strategy.primaryPatterns).toContain('**/architecture/**');
      expect(strategy.keywords).toContain('architecture');
      expect(strategy.maxFiles).toBe(20);
      expect(strategy.includeDependencies).toBe(true);
    });

    it('should generate appropriate strategy for API design', () => {
      const strategy = classifier.generateExtractionStrategy(
        DelegationType.API_DESIGN,
        ComplexityLevel.SIMPLE
      );

      expect(strategy.primaryPatterns).toContain('**/api/**');
      expect(strategy.keywords).toContain('api');
      expect(strategy.maxFiles).toBe(10);
      expect(strategy.includeDependencies).toBe(false);
    });

    it('should generate appropriate strategy for test strategy', () => {
      const strategy = classifier.generateExtractionStrategy(
        DelegationType.TEST_STRATEGY,
        ComplexityLevel.COMPLEX
      );

      expect(strategy.primaryPatterns).toContain('**/test/**');
      expect(strategy.keywords).toContain('test');
      expect(strategy.maxFiles).toBe(30);
      expect(strategy.includeDependencies).toBe(true);
    });
  });

  describe('Complexity-based Adjustments', () => {
    it('should adjust max files based on complexity', () => {
      const simple = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.SIMPLE
      );
      const moderate = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.MODERATE
      );
      const complex = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.COMPLEX
      );

      expect(simple.maxFiles).toBeLessThan(moderate.maxFiles);
      expect(moderate.maxFiles).toBeLessThan(complex.maxFiles);
    });

    it('should adjust dependency inclusion based on complexity', () => {
      const simple = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.SIMPLE
      );
      const complex = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.COMPLEX
      );

      expect(simple.includeDependencies).toBe(false);
      expect(complex.includeDependencies).toBe(true);
    });
  });

  describe('Pattern Validation', () => {
    it('should provide valid glob patterns', () => {
      const strategy = classifier.generateExtractionStrategy(
        DelegationType.ARCHITECTURE_DESIGN,
        ComplexityLevel.MODERATE
      );

      strategy.primaryPatterns.forEach(pattern => {
        expect(pattern).toBeTruthy();
        expect(typeof pattern).toBe('string');
      });

      strategy.secondaryPatterns.forEach(pattern => {
        expect(pattern).toBeTruthy();
        expect(typeof pattern).toBe('string');
      });
    });

    it('should provide relevant keywords', () => {
      const strategy = classifier.generateExtractionStrategy(
        DelegationType.API_DESIGN,
        ComplexityLevel.MODERATE
      );

      expect(strategy.keywords.length).toBeGreaterThan(0);
      strategy.keywords.forEach(keyword => {
        expect(keyword).toBeTruthy();
        expect(typeof keyword).toBe('string');
      });
    });
  });
});

describe('Convenience Functions', () => {
  describe('classifyProblem', () => {
    it('should work as a convenience function', () => {
      const result = classifyProblem('Design architecture for federated learning');

      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(result.suitable).toBe(true);
    });

    it('should handle title parameter', () => {
      const result = classifyProblem('Simple task', 'Architecture Design');

      expect(result.delegationType).toBe(DelegationType.ARCHITECTURE_DESIGN);
    });
  });

  describe('generateExtractionStrategy', () => {
    it('should work as a convenience function', () => {
      const strategy = generateExtractionStrategy(
        DelegationType.API_DESIGN,
        ComplexityLevel.MODERATE
      );

      expect(strategy.primaryPatterns).toContain('**/api/**');
      expect(strategy.maxFiles).toBe(20);
    });
  });
});

describe('Property-Based Test Scenarios', () => {
  describe('Invariant: All classifications have delegation type', () => {
    it('should always assign a delegation type', () => {
      const testCases = [
        '',
        'simple task',
        'complex architecture with verification',
        'api design with testing',
        'random text without keywords',
        '12345 numbers only',
        '!@#$% special chars only'
      ];

      testCases.forEach(description => {
        const result = classifyProblem(description);
        expect(Object.values(DelegationType)).toContain(result.delegationType);
      });
    });
  });

  describe('Invariant: Recommended context types are non-empty for suitable problems', () => {
    it('should provide context types for suitable problems', () => {
      const suitableProblems = [
        'Design distributed architecture',
        'Create comprehensive API design',
        'Develop testing strategy with property-based tests'
      ];

      suitableProblems.forEach(description => {
        const result = classifyProblem(description);
        if (result.suitable) {
          expect(result.requiredContextTypes.length).toBeGreaterThan(0);
        }
      });
    });
  });

  describe('Metamorphic: Adding complexity indicators increases suitability', () => {
    it('should increase suitability with more complexity indicators', () => {
      const simple = classifyProblem('Create function');
      const withArchitecture = classifyProblem('Create function for distributed architecture');
      const withMultiple = classifyProblem(
        'Create function for distributed architecture with formal verification and novel algorithms'
      );

      expect(withArchitecture.suitabilityScore).toBeGreaterThanOrEqual(simple.suitabilityScore);
      expect(withMultiple.suitabilityScore).toBeGreaterThanOrEqual(withArchitecture.suitabilityScore);
    });
  });

  describe('Error Condition: Problems with insufficient context flagged as unsuitable', () => {
    it('should flag very simple problems as unsuitable', () => {
      const verySimpleProblems = [
        'hello',
        'test',
        'simple',
        'basic task',
        'create file'
      ];

      verySimpleProblems.forEach(description => {
        const result = classifyProblem(description);
        expect(result.suitable).toBe(false);
        expect(result.reasoning).toContain('low complexity');
      });
    });
  });
});