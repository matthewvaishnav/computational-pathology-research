/**
 * Unit Tests for Context Extractor Component
 * Implements Task 5.4 - Test semantic search accuracy, problem-specific extraction strategies, snippet extraction
 * Requirements: 2.1-2.3, 11.1-11.7
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { ContextExtractor, FileMatch, CodeSnippet, ExtractionConfig } from './ContextExtractor.js';
import { DelegationType, ContextType } from '../types/core.js';

describe('ContextExtractor', () => {
  let extractor: ContextExtractor;

  beforeEach(() => {
    extractor = new ContextExtractor();
  });

  describe('Problem-Specific Extraction Strategies (Task 5.2)', () => {
    it('should prioritize architecture docs for architecture_design problems', () => {
      const strategy = extractor.getExtractionStrategy(DelegationType.ARCHITECTURE_DESIGN);
      
      expect(strategy).toBeDefined();
      expect(strategy?.name).toBe('Architecture Design');
      expect(strategy?.primarySources).toContain('architecture docs');
      expect(strategy?.contextTypes).toContain(ContextType.ARCHITECTURE_DOCS);
      expect(strategy?.keywords).toContain('architecture');
    });

    it('should prioritize API endpoints for api_design problems', () => {
      const strategy = extractor.getExtractionStrategy(DelegationType.API_DESIGN);
      
      expect(strategy).toBeDefined();
      expect(strategy?.name).toBe('API Design');
      expect(strategy?.primarySources).toContain('existing endpoints');
      expect(strategy?.contextTypes).toContain(ContextType.API_ENDPOINTS);
      expect(strategy?.keywords).toContain('api');
    });

    it('should prioritize test files for test_strategy problems', () => {
      const strategy = extractor.getExtractionStrategy(DelegationType.TEST_STRATEGY);
      
      expect(strategy).toBeDefined();
      expect(strategy?.name).toBe('Test Strategy');
      expect(strategy?.primarySources).toContain('test files');
      expect(strategy?.contextTypes).toContain(ContextType.TEST_FILES);
      expect(strategy?.keywords).toContain('test');
    });

    it('should prioritize external interfaces for integration_design problems', () => {
      const strategy = extractor.getExtractionStrategy(DelegationType.INTEGRATION_DESIGN);
      
      expect(strategy).toBeDefined();
      expect(strategy?.name).toBe('Integration Design');
      expect(strategy?.primarySources).toContain('external interfaces');
      expect(strategy?.contextTypes).toContain(ContextType.EXTERNAL_INTERFACES);
      expect(strategy?.keywords).toContain('integration');
    });

    it('should prioritize code metrics for refactoring_analysis problems', () => {
      const strategy = extractor.getExtractionStrategy(DelegationType.REFACTORING_ANALYSIS);
      
      expect(strategy).toBeDefined();
      expect(strategy?.name).toBe('Refactoring Analysis');
      expect(strategy?.primarySources).toContain('target modules');
      expect(strategy?.contextTypes).toContain(ContextType.DEPENDENCY_GRAPHS);
      expect(strategy?.keywords).toContain('refactor');
    });
  });

  describe('Configuration and Edge Cases', () => {
    it('should support all delegation types', () => {
      const supportedTypes = extractor.getSupportedDelegationTypes();
      
      expect(supportedTypes).toContain(DelegationType.ARCHITECTURE_DESIGN);
      expect(supportedTypes).toContain(DelegationType.API_DESIGN);
      expect(supportedTypes).toContain(DelegationType.TEST_STRATEGY);
      expect(supportedTypes).toContain(DelegationType.INTEGRATION_DESIGN);
      expect(supportedTypes).toContain(DelegationType.REFACTORING_ANALYSIS);
      
      // Verify each type has a strategy
      for (const type of supportedTypes) {
        const strategy = extractor.getExtractionStrategy(type);
        expect(strategy).toBeDefined();
        expect(strategy?.name).toBeDefined();
        expect(strategy?.keywords.length).toBeGreaterThan(0);
      }
    });

    it('should throw error for unsupported delegation types', async () => {
      await expect(
        extractor.extractContext(
          DelegationType.FORMAL_VERIFICATION, // Not implemented in strategies
          'formal verification',
          '/repo'
        )
      ).rejects.toThrow('No extraction strategy found');
    });
  });

  describe('Internal Helper Methods', () => {
    it('should detect programming language from file extension', () => {
      // Access private method through type assertion for testing
      const extractorAny = extractor as any;
      
      expect(extractorAny.detectLanguage('test.ts')).toBe('typescript');
      expect(extractorAny.detectLanguage('test.js')).toBe('javascript');
      expect(extractorAny.detectLanguage('test.py')).toBe('python');
      expect(extractorAny.detectLanguage('test.java')).toBe('java');
      expect(extractorAny.detectLanguage('test.md')).toBe('markdown');
      expect(extractorAny.detectLanguage('test.yaml')).toBe('yaml');
      expect(extractorAny.detectLanguage('test.unknown')).toBe('text');
    });

    it('should match glob patterns correctly', () => {
      const extractorAny = extractor as any;
      
      // Test basic patterns
      expect(extractorAny.matchesPattern('test.ts', '*.ts')).toBe(true);
      expect(extractorAny.matchesPattern('test.js', '*.ts')).toBe(false);
      
      // Test recursive patterns
      expect(extractorAny.matchesPattern('src/components/test.ts', '**/test.ts')).toBe(true);
      expect(extractorAny.matchesPattern('src/components/other.ts', '**/test.ts')).toBe(false);
      
      // Test directory patterns
      expect(extractorAny.matchesPattern('src/api/routes.ts', '**/api/**')).toBe(true);
      expect(extractorAny.matchesPattern('src/components/Button.ts', '**/api/**')).toBe(false);
    });

    it('should handle include/exclude patterns correctly', () => {
      const extractorAny = extractor as any;
      
      // Test include patterns
      expect(extractorAny.shouldInclude('test.ts', ['*.ts', '*.js'])).toBe(true);
      expect(extractorAny.shouldInclude('test.py', ['*.ts', '*.js'])).toBe(false);
      expect(extractorAny.shouldInclude('test.ts', [])).toBe(true); // Empty patterns should include all
      
      // Test exclude patterns
      expect(extractorAny.shouldExclude('node_modules/test.ts', ['**/node_modules/**'])).toBe(true);
      expect(extractorAny.shouldExclude('src/test.ts', ['**/node_modules/**'])).toBe(false);
      expect(extractorAny.shouldExclude('test.ts', [])).toBe(false); // Empty patterns should exclude none
    });

    it('should add file path annotations correctly', () => {
      const extractorAny = extractor as any;
      const content = 'const x = 1;\nconst y = 2;';
      const filePath = '/repo/test.ts';
      
      const annotated = extractorAny.addFilePathAnnotations(content, filePath);
      
      expect(annotated).toContain('### /repo/test.ts');
      expect(annotated).toContain('```typescript');
      expect(annotated).toContain('  1: const x = 1;');
      expect(annotated).toContain('  2: const y = 2;');
    });

    it('should calculate relevance scores correctly', () => {
      const extractorAny = extractor as any;
      const strategy = extractor.getExtractionStrategy(DelegationType.ARCHITECTURE_DESIGN)!;
      const keywords = ['architecture', 'component', 'system'];
      const lastModified = new Date();
      
      // Test high relevance (content matches keywords)
      const highScore = extractorAny.calculateRelevanceScore(
        '/repo/architecture/system.ts',
        'This file contains system architecture and component design.',
        keywords,
        strategy,
        lastModified,
        0.3
      );
      
      // Test low relevance (no keyword matches)
      const lowScore = extractorAny.calculateRelevanceScore(
        '/repo/utils/helper.ts',
        'This is a utility helper function.',
        keywords,
        strategy,
        lastModified,
        0.3
      );
      
      expect(highScore).toBeGreaterThan(lowScore);
      expect(highScore).toBeGreaterThan(0);
      expect(lowScore).toBeGreaterThanOrEqual(0);
    });

    it('should generate match reasons correctly', () => {
      const extractorAny = extractor as any;
      const strategy = extractor.getExtractionStrategy(DelegationType.ARCHITECTURE_DESIGN)!;
      const keywords = ['architecture', 'component'];
      
      const reasons = extractorAny.getMatchReasons(
        '/repo/architecture/system.ts',
        'This file contains system architecture and component design.',
        keywords,
        strategy
      );
      
      expect(reasons.length).toBeGreaterThan(0);
      expect(reasons.some(reason => reason.includes('keyword'))).toBe(true);
    });
  });

  describe('Strategy Configuration Validation', () => {
    it('should have valid strategies for all supported delegation types', () => {
      const supportedTypes = extractor.getSupportedDelegationTypes();
      
      for (const type of supportedTypes) {
        const strategy = extractor.getExtractionStrategy(type);
        
        expect(strategy).toBeDefined();
        expect(strategy!.name).toBeTruthy();
        expect(strategy!.primarySources.length).toBeGreaterThan(0);
        expect(strategy!.keywords.length).toBeGreaterThan(0);
        expect(strategy!.filePatterns.length).toBeGreaterThan(0);
        expect(strategy!.contextTypes.length).toBeGreaterThan(0);
      }
    });

    it('should have appropriate keywords for each strategy', () => {
      const architectureStrategy = extractor.getExtractionStrategy(DelegationType.ARCHITECTURE_DESIGN)!;
      expect(architectureStrategy.keywords).toContain('architecture');
      expect(architectureStrategy.keywords).toContain('component');
      
      const apiStrategy = extractor.getExtractionStrategy(DelegationType.API_DESIGN)!;
      expect(apiStrategy.keywords).toContain('api');
      expect(apiStrategy.keywords).toContain('endpoint');
      
      const testStrategy = extractor.getExtractionStrategy(DelegationType.TEST_STRATEGY)!;
      expect(testStrategy.keywords).toContain('test');
      expect(testStrategy.keywords).toContain('spec');
    });

    it('should have appropriate file patterns for each strategy', () => {
      const architectureStrategy = extractor.getExtractionStrategy(DelegationType.ARCHITECTURE_DESIGN)!;
      expect(architectureStrategy.filePatterns.some(p => p.includes('architecture'))).toBe(true);
      
      const apiStrategy = extractor.getExtractionStrategy(DelegationType.API_DESIGN)!;
      expect(apiStrategy.filePatterns.some(p => p.includes('api'))).toBe(true);
      
      const testStrategy = extractor.getExtractionStrategy(DelegationType.TEST_STRATEGY)!;
      expect(testStrategy.filePatterns.some(p => p.includes('test'))).toBe(true);
    });
  });
});