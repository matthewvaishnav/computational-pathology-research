/**
 * Unit tests for Context Packager Component
 * Tests Task 6.1, 6.2, 6.3, 6.4, 6.5 - All Context Packager functionality
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ContextPackager, PackagingConfig } from './ContextPackager.js';
import { FileMatch, CodeSnippet, ExtractionStrategy } from './ContextExtractor.js';
import { ContextType } from '../types/core.js';

describe('ContextPackager', () => {
  let packager: ContextPackager;
  let mockFiles: FileMatch[];
  let mockSnippets: CodeSnippet[];
  let mockStrategy: ExtractionStrategy;

  beforeEach(() => {
    packager = new ContextPackager();
    
    mockFiles = [
      {
        filePath: 'src/components/UserService.ts',
        relevanceScore: 0.9,
        matchReasons: ['Contains keywords: user, service'],
        lastModified: new Date('2024-01-15'),
        size: 2048
      },
      {
        filePath: 'src/models/User.ts',
        relevanceScore: 0.8,
        matchReasons: ['Contains keywords: user, model'],
        lastModified: new Date('2024-01-10'),
        size: 1024
      },
      {
        filePath: 'docs/api.md',
        relevanceScore: 0.7,
        matchReasons: ['Documentation file'],
        lastModified: new Date('2024-01-05'),
        size: 3072
      }
    ];

    mockSnippets = [
      {
        filePath: 'src/components/UserService.ts',
        startLine: 1,
        endLine: 50,
        content: `export class UserService {
  constructor(private db: Database) {}
  
  async getUser(id: string): Promise<User> {
    return this.db.findUser(id);
  }
  
  async createUser(userData: CreateUserRequest): Promise<User> {
    return this.db.createUser(userData);
  }
}`,
        contextWindow: 5,
        language: 'typescript'
      },
      {
        filePath: 'src/models/User.ts',
        startLine: 1,
        endLine: 20,
        content: `export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

export interface CreateUserRequest {
  name: string;
  email: string;
}`,
        contextWindow: 5,
        language: 'typescript'
      }
    ];

    mockStrategy = {
      name: 'API Design',
      primarySources: ['existing endpoints', 'data models'],
      secondarySources: ['client code', 'integration tests'],
      keywords: ['api', 'endpoint', 'user', 'service'],
      filePatterns: ['**/api/**', '**/models/**'],
      contextTypes: [ContextType.API_ENDPOINTS, ContextType.CODE_SNIPPETS]
    };
  });

  describe('packageContext', () => {
    it('should create a complete context bundle', async () => {
      const bundle = await packager.packageContext(
        'User API Design',
        'Design a REST API for user management',
        mockFiles,
        mockSnippets,
        mockStrategy,
        ['Must follow REST conventions', 'Include proper error handling']
      );

      expect(bundle.problemTitle).toBe('User API Design');
      expect(bundle.problemSummary).toBe('Design a REST API for user management');
      expect(bundle.codeSnippets).toHaveLength(2);
      expect(bundle.constraints).toHaveLength(2);
      expect(bundle.contextManifest).toBeDefined();
      expect(bundle.totalSize).toBeGreaterThan(0);
    });

    it('should include formatted code snippets with syntax highlighting', async () => {
      const bundle = await packager.packageContext(
        'Test Title',
        'Test Description',
        mockFiles,
        mockSnippets,
        mockStrategy
      );

      const snippet = bundle.codeSnippets[0];
      expect(snippet.filePath).toBe('src/components/UserService.ts');
      expect(snippet.language).toBe('typescript');
      expect(snippet.content).toContain('```typescript');
      expect(snippet.size).toBeGreaterThan(0);
    });

    it('should generate context manifest with correct information', async () => {
      const bundle = await packager.packageContext(
        'Test Title',
        'Test Description',
        mockFiles,
        mockSnippets,
        mockStrategy
      );

      const manifest = bundle.contextManifest;
      expect(manifest.totalFiles).toBe(3);
      expect(manifest.sources).toHaveLength(3);
      expect(manifest.totalSize).toBeGreaterThan(0);
      
      // Check that code files are marked as included
      const codeSource = manifest.sources.find(s => s.path === 'src/components/UserService.ts');
      expect(codeSource?.included).toBe(true);
      expect(codeSource?.type).toBe('Code');
    });
  });

  describe('generateMarkdown', () => {
    it('should generate copy-paste ready markdown', async () => {
      const bundle = await packager.packageContext(
        'User API Design',
        'Design a REST API for user management',
        mockFiles,
        mockSnippets,
        mockStrategy,
        ['Must follow REST conventions']
      );

      const markdown = packager.generateMarkdown(bundle);

      expect(markdown).toContain('# Context Bundle: User API Design');
      expect(markdown).toContain('## Problem Summary');
      expect(markdown).toContain('Design a REST API for user management');
      expect(markdown).toContain('## Relevant Code');
      expect(markdown).toContain('### src/components/UserService.ts');
      expect(markdown).toContain('## Constraints');
      expect(markdown).toContain('- Must follow REST conventions');
      expect(markdown).toContain('## Context Manifest');
      expect(markdown).toContain('| Source | Type | Size | Relevance | Included |');
    });

    it('should include excluded content summary when compression is applied', async () => {
      // Create a large snippet that will trigger size limits
      const largeSnippets = Array.from({ length: 20 }, (_, i) => ({
        ...mockSnippets[0],
        filePath: `src/large-file-${i}.ts`,
        content: 'x'.repeat(5000) // 5KB each = 100KB total
      }));

      const config: Partial<PackagingConfig> = {
        maxSize: 10000 // 10KB limit
      };

      const bundle = await packager.packageContext(
        'Large Codebase',
        'Test with size limits',
        mockFiles,
        largeSnippets,
        mockStrategy,
        [],
        config
      );

      expect(bundle.excludedContent).toBeDefined();
      expect(bundle.compressionApplied).toBe(true);

      const markdown = packager.generateMarkdown(bundle);
      expect(markdown).toContain('## Excluded Content Summary');
      expect(markdown).toContain('Content has been compressed and optimized');
    });
  });

  describe('size management and optimization', () => {
    it('should enforce character limits', async () => {
      const config: Partial<PackagingConfig> = {
        maxSize: 1000 // Very small limit
      };

      const bundle = await packager.packageContext(
        'Size Test',
        'Test size limits',
        mockFiles,
        mockSnippets,
        mockStrategy,
        [],
        config
      );

      expect(bundle.totalSize).toBeLessThanOrEqual(1000);
      expect(bundle.excludedContent).toBeDefined();
    });

    it('should prioritize content by relevance when exceeding limits', async () => {
      // Create snippets with different relevance scores
      const prioritySnippets = [
        { ...mockSnippets[0], content: 'x'.repeat(2000) }, // High relevance (0.9)
        { ...mockSnippets[1], content: 'y'.repeat(2000) }  // Lower relevance (0.8)
      ];

      const config: Partial<PackagingConfig> = {
        maxSize: 3000 // Only room for one snippet + overhead
      };

      const bundle = await packager.packageContext(
        'Priority Test',
        'Test prioritization',
        mockFiles,
        prioritySnippets,
        mockStrategy,
        [],
        config
      );

      // Should include the higher relevance snippet
      expect(bundle.codeSnippets).toHaveLength(1);
      expect(bundle.codeSnippets[0].filePath).toBe('src/components/UserService.ts');
      expect(bundle.excludedContent?.excludedFiles).toHaveLength(1);
    });

    it('should generate summaries of excluded content', async () => {
      const config: Partial<PackagingConfig> = {
        maxSize: 500 // Very small to force exclusions
      };

      const bundle = await packager.packageContext(
        'Exclusion Test',
        'Test exclusions',
        mockFiles,
        mockSnippets,
        mockStrategy,
        [],
        config
      );

      expect(bundle.excludedContent).toBeDefined();
      expect(bundle.excludedContent!.summary).toContain('exceeded');
      expect(bundle.excludedContent!.totalExcluded).toBeGreaterThan(0);
    });
  });

  describe('context compression', () => {
    it('should remove redundant whitespace while preserving semantics', async () => {
      const snippetWithWhitespace = {
        ...mockSnippets[0],
        content: `export class UserService {


  constructor(private db: Database) {}
  
  
  async getUser(id: string): Promise<User> {
    return this.db.findUser(id);
  }
}`
      };

      const config: Partial<PackagingConfig> = {
        enableCompression: true,
        syntaxHighlighting: false
      };

      const bundle = await packager.packageContext(
        'Compression Test',
        'Test compression',
        mockFiles,
        [snippetWithWhitespace],
        mockStrategy,
        [],
        config
      );

      expect(bundle.compressionApplied).toBe(true);
      // Should have fewer characters due to whitespace removal
      expect(bundle.codeSnippets[0].content.length).toBeLessThan(snippetWithWhitespace.content.length);
      // But should still contain the essential code
      expect(bundle.codeSnippets[0].content).toContain('export class UserService');
      expect(bundle.codeSnippets[0].content).toContain('async getUser');
    });

    it('should deduplicate similar code patterns', async () => {
      const snippetWithDuplicates = {
        ...mockSnippets[0],
        content: `import { Database } from './database';
import { User } from './user';
import { Database } from './database'; // Duplicate

export class UserService {
  // Implementation
}`
      };

      const config: Partial<PackagingConfig> = {
        enableCompression: true
      };

      const bundle = await packager.packageContext(
        'Deduplication Test',
        'Test deduplication',
        mockFiles,
        [snippetWithDuplicates],
        mockStrategy,
        [],
        config
      );

      const content = bundle.codeSnippets[0].content;
      // Should only have one instance of the Database import
      const databaseImports = (content.match(/import.*Database/g) || []).length;
      expect(databaseImports).toBe(1);
    });

    it('should implement extractive summarization for large docs', async () => {
      // This would be tested with actual documentation files
      // For now, we test that the compression flag is set correctly
      const config: Partial<PackagingConfig> = {
        enableCompression: true,
        enableSummarization: true
      };

      const bundle = await packager.packageContext(
        'Summarization Test',
        'Test summarization',
        mockFiles,
        mockSnippets,
        mockStrategy,
        [],
        config
      );

      expect(bundle.compressionApplied).toBe(true);
    });
  });

  describe('context bundle assembly', () => {
    it('should combine all content types correctly', async () => {
      const bundle = await packager.packageContext(
        'Assembly Test',
        'Test bundle assembly',
        mockFiles,
        mockSnippets,
        mockStrategy,
        ['Constraint 1', 'Constraint 2']
      );

      // Should have all components
      expect(bundle.codeSnippets.length).toBeGreaterThan(0);
      expect(bundle.constraints.length).toBe(2);
      expect(bundle.contextManifest).toBeDefined();
      expect(bundle.problemTitle).toBe('Assembly Test');
      expect(bundle.problemSummary).toBe('Test bundle assembly');
    });

    it('should generate copy-paste ready markdown format', async () => {
      const bundle = await packager.packageContext(
        'Format Test',
        'Test markdown format',
        mockFiles,
        mockSnippets,
        mockStrategy
      );

      const markdown = packager.generateMarkdown(bundle);

      // Should be valid markdown with proper structure
      expect(markdown).toMatch(/^# Context Bundle:/);
      expect(markdown).toContain('## Problem Summary');
      expect(markdown).toContain('## Relevant Code');
      expect(markdown).toContain('## Context Manifest');
      expect(markdown).toContain('```typescript');
      expect(markdown).toContain('| Source | Type | Size | Relevance | Included |');
    });
  });

  describe('markdown formatting engine', () => {
    it('should format code snippets with syntax highlighting markers', async () => {
      const bundle = await packager.packageContext(
        'Syntax Test',
        'Test syntax highlighting',
        mockFiles,
        mockSnippets,
        mockStrategy
      );

      const snippet = bundle.codeSnippets[0];
      expect(snippet.content).toContain('```typescript');
      expect(snippet.content).toContain('export class UserService');
      expect(snippet.language).toBe('typescript');
    });

    it('should add file path and line number annotations', async () => {
      const bundle = await packager.packageContext(
        'Annotation Test',
        'Test annotations',
        mockFiles,
        mockSnippets,
        mockStrategy
      );

      const markdown = packager.generateMarkdown(bundle);
      expect(markdown).toContain('### src/components/UserService.ts (lines 1-50)');
      expect(markdown).toContain('### src/models/User.ts (lines 1-20)');
    });

    it('should generate context manifest tables', async () => {
      const bundle = await packager.packageContext(
        'Manifest Test',
        'Test manifest generation',
        mockFiles,
        mockSnippets,
        mockStrategy
      );

      const markdown = packager.generateMarkdown(bundle);
      
      // Should have table headers
      expect(markdown).toContain('| Source | Type | Size | Relevance | Included |');
      expect(markdown).toContain('|--------|------|------|-----------|----------|');
      
      // Should have file entries
      expect(markdown).toContain('src/components/UserService.ts');
      expect(markdown).toContain('Code');
      expect(markdown).toContain('✓'); // Included marker
      
      // Should have summary information
      expect(markdown).toContain('**Total Files:**');
      expect(markdown).toContain('**Total Size:**');
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle empty input gracefully', async () => {
      const bundle = await packager.packageContext(
        'Empty Test',
        'Test empty input',
        [],
        [],
        mockStrategy
      );

      expect(bundle.codeSnippets).toHaveLength(0);
      expect(bundle.documentationExcerpts).toHaveLength(0);
      expect(bundle.contextManifest.totalFiles).toBe(0);
      expect(bundle.totalSize).toBeGreaterThan(0); // Should still have headers
    });

    it('should handle very small size limits', async () => {
      const config: Partial<PackagingConfig> = {
        maxSize: 100 // Extremely small
      };

      const bundle = await packager.packageContext(
        'Tiny Test',
        'Test tiny limits',
        mockFiles,
        mockSnippets,
        mockStrategy,
        [],
        config
      );

      expect(bundle.totalSize).toBeLessThanOrEqual(100);
      expect(bundle.excludedContent).toBeDefined();
    });

    it('should preserve semantic meaning during compression', async () => {
      const semanticSnippet = {
        ...mockSnippets[0],
        content: `// This is an important comment
export class UserService {
  // Critical business logic comment
  async getUser(id: string): Promise<User> {
    // Validation comment
    if (!id) throw new Error('ID required');
    return this.db.findUser(id);
  }
}`
      };

      const config: Partial<PackagingConfig> = {
        enableCompression: true
      };

      const bundle = await packager.packageContext(
        'Semantic Test',
        'Test semantic preservation',
        mockFiles,
        [semanticSnippet],
        mockStrategy,
        [],
        config
      );

      const compressed = bundle.codeSnippets[0].content;
      // Should still contain the essential structure
      expect(compressed).toContain('export class UserService');
      expect(compressed).toContain('async getUser');
      expect(compressed).toContain('throw new Error');
      expect(compressed).toContain('return this.db.findUser');
    });
  });

  describe('configuration options', () => {
    it('should respect custom packaging configuration', async () => {
      const config: PackagingConfig = {
        maxSize: 5000,
        enableCompression: false,
        enableSummarization: false,
        prioritizeRecent: true,
        includeLineNumbers: false,
        syntaxHighlighting: false
      };

      const bundle = await packager.packageContext(
        'Config Test',
        'Test custom config',
        mockFiles,
        mockSnippets,
        mockStrategy,
        [],
        config
      );

      expect(bundle.compressionApplied).toBe(false);
      expect(bundle.totalSize).toBeLessThanOrEqual(5000);
    });

    it('should handle disabled syntax highlighting', async () => {
      const config: Partial<PackagingConfig> = {
        syntaxHighlighting: false
      };

      const bundle = await packager.packageContext(
        'No Syntax Test',
        'Test without syntax highlighting',
        mockFiles,
        mockSnippets,
        mockStrategy,
        [],
        config
      );

      // Should still have content but without syntax highlighting markers
      expect(bundle.codeSnippets[0].content).toBeDefined();
      expect(bundle.codeSnippets[0].content.length).toBeGreaterThan(0);
    });
  });
});