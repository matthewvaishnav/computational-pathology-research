/**
 * Security tests for ContextExtractor - ReDoS Prevention
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ContextExtractor } from './ContextExtractor.js';
import { DelegationType } from '../types/core.js';

describe('ContextExtractor Security', () => {
  let extractor: ContextExtractor;

  beforeEach(() => {
    extractor = new ContextExtractor('/test/repo');
  });

  describe('ReDoS Prevention in Glob Matching', () => {
    it('should reject patterns with too many wildcards', () => {
      const maliciousPattern = '**/**/***/**/**/**/**/**/**/**/**/**';
      
      expect(() => {
        // @ts-ignore - accessing private method for testing
        extractor['matchesPattern']('test/file.ts', maliciousPattern);
      }).toThrow(/too complex/);
    });

    it('should handle normal patterns efficiently', () => {
      // @ts-ignore
      expect(extractor['matchesPattern']('test.ts', '*.ts')).toBe(true);
      
      // @ts-ignore - **/ patterns handled by early returns
      expect(extractor['matchesPattern']('test.ts', '**/test.ts')).toBe(true);
      
      // @ts-ignore
      expect(extractor['matchesPattern']('test.js', '*.ts')).toBe(false);
    });

    it('should complete glob matching in reasonable time', () => {
      const longPath = 'a/'.repeat(100) + 'test.ts';
      const pattern = '**/*.ts';
      
      const start = Date.now();
      // @ts-ignore - accessing private method for testing
      extractor['matchesPattern'](longPath, pattern);
      const duration = Date.now() - start;
      
      // Should complete in under 100ms even with long paths
      expect(duration).toBeLessThan(100);
    });

    it('should handle edge cases safely', () => {
      const edgeCases = [
        { path: 'test.ts', pattern: '*', expected: true },  // Single wildcard matches
        { path: 'test.ts', pattern: '', expected: false },  // Empty pattern
        { path: 'test.ts', pattern: 'test.ts', expected: true },  // Exact match
      ];

      for (const { path, pattern, expected } of edgeCases) {
        // @ts-ignore - accessing private method for testing
        const result = extractor['matchesPattern'](path, pattern);
        expect(result).toBe(expected);
      }
    });
  });
});
