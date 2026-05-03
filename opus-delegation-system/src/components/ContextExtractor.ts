/**
 * Context Extractor Component
 * Implements Task 5.1, 5.2, 5.3 - Semantic search, problem-specific extraction, code snippet extraction
 * Requirements: 2.1, 2.2, 2.7, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
 */

import * as fs from 'fs';
import * as path from 'path';
import { DelegationType, ContextType } from '../types/core.js';

// File discovery and ranking
export interface FileMatch {
  filePath: string;
  relevanceScore: number;
  matchReasons: string[];
  lastModified: Date;
  size: number;
}

// Code snippet with context
export interface CodeSnippet {
  filePath: string;
  startLine: number;
  endLine: number;
  content: string;
  contextWindow: number;
  language?: string;
}

// Extraction configuration
export interface ExtractionConfig {
  maxFiles: number;
  maxTotalSize: number; // in characters
  contextWindow: number; // lines around functions/classes
  includePatterns: string[];
  excludePatterns: string[];
  recencyWeight: number; // 0-1, how much to weight recent files
}

// Problem-specific extraction strategy
export interface ExtractionStrategy {
  name: string;
  primarySources: string[];
  secondarySources: string[];
  keywords: string[];
  filePatterns: string[];
  contextTypes: ContextType[];
}

export class ContextExtractor {
  private defaultConfig: ExtractionConfig = {
    maxFiles: 50,
    maxTotalSize: 45000, // Leave room for other content in 50k limit
    contextWindow: 5,
    includePatterns: [
      '**/*.ts', '**/*.js', '**/*.py', '**/*.java', '**/*.cpp', '**/*.c',
      '**/*.md', '**/*.yaml', '**/*.yml', '**/*.json', '**/*.toml'
    ],
    excludePatterns: [
      '**/node_modules/**', '**/dist/**', '**/build/**', '**/.git/**',
      '**/coverage/**', '**/.venv/**'
    ],
    recencyWeight: 0.3
  };

  private extractionStrategies: Map<DelegationType, ExtractionStrategy> = new Map([
    [DelegationType.ARCHITECTURE_DESIGN, {
      name: 'Architecture Design',
      primarySources: ['architecture docs', 'component interfaces', 'system diagrams'],
      secondarySources: ['config files', 'deployment specs'],
      keywords: [
        'architecture', 'component', 'interface', 'service', 'module', 'system',
        'design', 'structure', 'pattern', 'layer', 'tier', 'boundary'
      ],
      filePatterns: [
        '**/architecture/**', '**/docs/architecture/**', '**/design/**',
        '**/interfaces/**', '**/services/**', '**/components/**',
        '**/*architecture*', '**/*design*', '**/*interface*'
      ],
      contextTypes: [ContextType.ARCHITECTURE_DOCS, ContextType.CODE_SNIPPETS, ContextType.EXISTING_DESIGNS]
    }],
    
    [DelegationType.API_DESIGN, {
      name: 'API Design',
      primarySources: ['existing endpoints', 'data models', 'OpenAPI specs'],
      secondarySources: ['client code', 'integration tests'],
      keywords: [
        'api', 'endpoint', 'route', 'controller', 'handler', 'model', 'schema',
        'request', 'response', 'rest', 'graphql', 'openapi', 'swagger'
      ],
      filePatterns: [
        '**/api/**', '**/routes/**', '**/controllers/**', '**/handlers/**',
        '**/models/**', '**/schemas/**', '**/*api*', '**/*route*',
        '**/*.openapi.*', '**/*swagger*'
      ],
      contextTypes: [ContextType.API_ENDPOINTS, ContextType.CODE_SNIPPETS, ContextType.EXISTING_DESIGNS]
    }],
    
    [DelegationType.TEST_STRATEGY, {
      name: 'Test Strategy',
      primarySources: ['test files', 'code under test', 'test utilities'],
      secondarySources: ['coverage reports', 'CI config'],
      keywords: [
        'test', 'spec', 'assert', 'expect', 'mock', 'stub', 'fixture',
        'property', 'generator', 'coverage', 'unit', 'integration'
      ],
      filePatterns: [
        '**/test/**', '**/tests/**', '**/spec/**', '**/__tests__/**',
        '**/*.test.*', '**/*.spec.*', '**/coverage/**', '**/*test*'
      ],
      contextTypes: [ContextType.TEST_FILES, ContextType.CODE_SNIPPETS, ContextType.REQUIREMENTS_DOCS]
    }],
    
    [DelegationType.INTEGRATION_DESIGN, {
      name: 'Integration Design',
      primarySources: ['external interfaces', 'protocol specs', 'adapters'],
      secondarySources: ['error logs', 'retry policies'],
      keywords: [
        'integration', 'external', 'adapter', 'connector', 'client', 'protocol',
        'dicom', 'hl7', 'fhir', 'webhook', 'queue', 'event', 'message'
      ],
      filePatterns: [
        '**/integration/**', '**/adapters/**', '**/connectors/**', '**/clients/**',
        '**/external/**', '**/*integration*', '**/*adapter*', '**/*client*'
      ],
      contextTypes: [ContextType.EXTERNAL_INTERFACES, ContextType.CODE_SNIPPETS, ContextType.ARCHITECTURE_DOCS]
    }],
    
    [DelegationType.REFACTORING_ANALYSIS, {
      name: 'Refactoring Analysis',
      primarySources: ['target modules', 'dependency graph', 'code metrics'],
      secondarySources: ['git history', 'code review comments'],
      keywords: [
        'refactor', 'cleanup', 'debt', 'smell', 'coupling', 'cohesion',
        'dependency', 'import', 'export', 'class', 'function', 'method'
      ],
      filePatterns: [
        '**/*.ts', '**/*.js', '**/*.py', '**/*.java',
        '**/src/**', '**/lib/**', '**/core/**'
      ],
      contextTypes: [ContextType.CODE_SNIPPETS, ContextType.DEPENDENCY_GRAPHS, ContextType.ARCHITECTURE_DOCS]
    }]
  ]);

  /**
   * Extract context for a specific problem type and description
   * Implements Requirements 2.1, 11.1-11.6
   */
  public async extractContext(
    problemType: DelegationType,
    problemDescription: string,
    repositoryPath: string,
    config?: Partial<ExtractionConfig>
  ): Promise<{
    files: FileMatch[];
    snippets: CodeSnippet[];
    totalSize: number;
    strategy: ExtractionStrategy;
  }> {
    const finalConfig = { ...this.defaultConfig, ...config };
    const strategy = this.extractionStrategies.get(problemType);
    
    if (!strategy) {
      throw new Error(`No extraction strategy found for problem type: ${problemType}`);
    }

    // Step 1: Semantic search for file discovery (Task 5.1)
    const discoveredFiles = await this.semanticFileSearch(
      problemDescription,
      strategy,
      repositoryPath,
      finalConfig
    );

    // Step 2: Apply problem-specific extraction strategy (Task 5.2)
    const relevantFiles = this.applyExtractionStrategy(
      discoveredFiles,
      strategy,
      finalConfig
    );

    // Step 3: Extract code snippets with context (Task 5.3)
    const snippets = await this.extractCodeSnippets(
      relevantFiles,
      finalConfig
    );

    // Calculate total size
    const totalSize = snippets.reduce((sum, snippet) => sum + snippet.content.length, 0);

    return {
      files: relevantFiles,
      snippets,
      totalSize,
      strategy
    };
  }

  /**
   * Semantic search for file discovery (Task 5.1)
   * Implements keyword-based search, dependency analysis, and recency weighting
   */
  private async semanticFileSearch(
    problemDescription: string,
    strategy: ExtractionStrategy,
    repositoryPath: string,
    config: ExtractionConfig
  ): Promise<FileMatch[]> {
    const allFiles = await this.findAllFiles(repositoryPath, config);
    const keywords = this.extractKeywords(problemDescription, strategy);
    
    const fileMatches: FileMatch[] = [];

    for (const filePath of allFiles) {
      try {
        const stats = await fs.promises.stat(filePath);
        const content = await fs.promises.readFile(filePath, 'utf-8');
        
        // Calculate relevance score
        const relevanceScore = this.calculateRelevanceScore(
          filePath,
          content,
          keywords,
          strategy,
          stats.mtime,
          config.recencyWeight
        );

        if (relevanceScore > 0) {
          const matchReasons = this.getMatchReasons(filePath, content, keywords, strategy);
          
          fileMatches.push({
            filePath,
            relevanceScore,
            matchReasons,
            lastModified: stats.mtime,
            size: stats.size
          });
        }
      } catch (error) {
        // Skip files that can't be read (binary, permissions, etc.)
        continue;
      }
    }

    // Sort by relevance score (descending)
    return fileMatches.sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  /**
   * Apply problem-specific extraction strategy (Task 5.2)
   */
  private applyExtractionStrategy(
    discoveredFiles: FileMatch[],
    strategy: ExtractionStrategy,
    config: ExtractionConfig
  ): FileMatch[] {
    // Prioritize files based on strategy patterns
    const prioritizedFiles = discoveredFiles.map(file => {
      let priorityBoost = 0;
      
      // Check if file matches primary source patterns
      for (const pattern of strategy.filePatterns) {
        if (this.matchesPattern(file.filePath, pattern)) {
          priorityBoost += 0.3;
          break;
        }
      }
      
      // Boost for files in primary source directories
      for (const source of strategy.primarySources) {
        if (file.filePath.toLowerCase().includes(source.toLowerCase().replace(' ', ''))) {
          priorityBoost += 0.2;
        }
      }
      
      return {
        ...file,
        relevanceScore: file.relevanceScore + priorityBoost
      };
    });

    // Re-sort and limit
    const sorted = prioritizedFiles.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return sorted.slice(0, config.maxFiles);
  }

  /**
   * Extract code snippets with configurable context window (Task 5.3)
   */
  private async extractCodeSnippets(
    files: FileMatch[],
    config: ExtractionConfig
  ): Promise<CodeSnippet[]> {
    const snippets: CodeSnippet[] = [];
    let totalSize = 0;

    for (const file of files) {
      if (totalSize >= config.maxTotalSize) {
        break;
      }

      try {
        const content = await fs.promises.readFile(file.filePath, 'utf-8');
        const lines = content.split('\n');
        
        // For now, extract full file content with annotations
        // In a more sophisticated implementation, we would parse AST to extract functions/classes
        const snippet: CodeSnippet = {
          filePath: file.filePath,
          startLine: 1,
          endLine: lines.length,
          content: this.addFilePathAnnotations(content, file.filePath),
          contextWindow: config.contextWindow,
          language: this.detectLanguage(file.filePath)
        };

        const snippetSize = snippet.content.length;
        if (totalSize + snippetSize <= config.maxTotalSize) {
          snippets.push(snippet);
          totalSize += snippetSize;
        } else {
          // Truncate the snippet to fit within size limit
          const remainingSize = config.maxTotalSize - totalSize;
          if (remainingSize > 500) { // Only include if we have reasonable space
            snippet.content = snippet.content.substring(0, remainingSize - 100) + '\n... [truncated]';
            snippets.push(snippet);
          }
          break;
        }
      } catch (error) {
        continue;
      }
    }

    return snippets;
  }

  /**
   * Find all files in repository matching include/exclude patterns
   */
  private async findAllFiles(
    repositoryPath: string,
    config: ExtractionConfig
  ): Promise<string[]> {
    const files: string[] = [];
    
    const walkDir = async (dir: string): Promise<void> => {
      try {
        const entries = await fs.promises.readdir(dir, { withFileTypes: true });
        
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          const relativePath = path.relative(repositoryPath, fullPath);
          
          if (entry.isDirectory()) {
            // Check if directory should be excluded
            if (!this.shouldExclude(relativePath, config.excludePatterns)) {
              await walkDir(fullPath);
            }
          } else if (entry.isFile()) {
            // Check if file should be included
            if (this.shouldInclude(relativePath, config.includePatterns) &&
                !this.shouldExclude(relativePath, config.excludePatterns)) {
              files.push(fullPath);
            }
          }
        }
      } catch (error) {
        // Skip directories we can't read
      }
    };

    try {
      await walkDir(repositoryPath);
    } catch (error) {
      // If we can't read the root directory, return empty array
    }
    
    return files;
  }

  /**
   * Extract keywords from problem description and strategy
   */
  private extractKeywords(problemDescription: string, strategy: ExtractionStrategy): string[] {
    const descriptionWords = problemDescription
      .toLowerCase()
      .split(/\s+/)
      .filter(word => word.length > 3)
      .slice(0, 20); // Limit to avoid noise

    return [...strategy.keywords, ...descriptionWords];
  }

  /**
   * Calculate relevance score for a file
   */
  private calculateRelevanceScore(
    filePath: string,
    content: string,
    keywords: string[],
    strategy: ExtractionStrategy,
    lastModified: Date,
    recencyWeight: number
  ): number {
    let score = 0;
    const contentLower = content.toLowerCase();
    const pathLower = filePath.toLowerCase();

    // Keyword matching in content (40% weight)
    const keywordMatches = keywords.filter(keyword => 
      contentLower.includes(keyword.toLowerCase())
    ).length;
    const keywordScore = Math.min(1, keywordMatches / keywords.length * 2);
    score += keywordScore * 0.4;

    // Path matching (30% weight)
    const pathMatches = strategy.filePatterns.filter(pattern =>
      this.matchesPattern(filePath, pattern)
    ).length;
    const pathScore = pathMatches > 0 ? 1 : 0;
    score += pathScore * 0.3;

    // File name relevance (20% weight)
    const nameMatches = strategy.keywords.filter(keyword =>
      pathLower.includes(keyword.toLowerCase())
    ).length;
    const nameScore = Math.min(1, nameMatches / strategy.keywords.length * 2);
    score += nameScore * 0.2;

    // Recency bonus (10% weight)
    const daysSinceModified = (Date.now() - lastModified.getTime()) / (1000 * 60 * 60 * 24);
    const recencyScore = Math.max(0, 1 - daysSinceModified / 30); // Decay over 30 days
    score += recencyScore * recencyWeight * 0.1;

    return score;
  }

  /**
   * Get human-readable match reasons
   */
  private getMatchReasons(
    filePath: string,
    content: string,
    keywords: string[],
    strategy: ExtractionStrategy
  ): string[] {
    const reasons: string[] = [];
    const contentLower = content.toLowerCase();
    const pathLower = filePath.toLowerCase();

    // Check keyword matches
    const matchedKeywords = keywords.filter(keyword =>
      contentLower.includes(keyword.toLowerCase())
    );
    if (matchedKeywords.length > 0) {
      reasons.push(`Contains keywords: ${matchedKeywords.slice(0, 3).join(', ')}`);
    }

    // Check path patterns
    for (const pattern of strategy.filePatterns) {
      if (this.matchesPattern(filePath, pattern)) {
        reasons.push(`Matches pattern: ${pattern}`);
        break;
      }
    }

    // Check strategy-specific matches
    for (const source of strategy.primarySources) {
      if (pathLower.includes(source.toLowerCase().replace(' ', ''))) {
        reasons.push(`Primary source: ${source}`);
      }
    }

    return reasons;
  }

  /**
   * Add file path and line number annotations to content
   */
  private addFilePathAnnotations(content: string, filePath: string): string {
    const lines = content.split('\n');
    const annotatedLines = lines.map((line, index) => 
      `${(index + 1).toString().padStart(3, ' ')}: ${line}`
    );
    
    return `### ${filePath}\n\n\`\`\`${this.detectLanguage(filePath)}\n${annotatedLines.join('\n')}\n\`\`\`\n`;
  }

  /**
   * Detect programming language from file extension
   */
  private detectLanguage(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    const languageMap: Record<string, string> = {
      '.ts': 'typescript',
      '.js': 'javascript',
      '.py': 'python',
      '.java': 'java',
      '.cpp': 'cpp',
      '.c': 'c',
      '.md': 'markdown',
      '.yaml': 'yaml',
      '.yml': 'yaml',
      '.json': 'json',
      '.toml': 'toml'
    };
    
    return languageMap[ext] || 'text';
  }

  /**
   * Check if file should be included based on patterns
   */
  private shouldInclude(filePath: string, patterns: string[]): boolean {
    if (patterns.length === 0) return true;
    return patterns.some(pattern => this.matchesPattern(filePath, pattern));
  }

  /**
   * Check if file should be excluded based on patterns
   */
  private shouldExclude(filePath: string, patterns: string[]): boolean {
    if (patterns.length === 0) return false;
    return patterns.some(pattern => this.matchesPattern(filePath, pattern));
  }

  /**
   * Simple glob pattern matching
   */
  private matchesPattern(filePath: string, pattern: string): boolean {
    // Normalize path separators
    const normalizedPath = filePath.replace(/\\/g, '/');
    const normalizedPattern = pattern.replace(/\\/g, '/');
    
    // Simple implementation for common patterns
    if (normalizedPattern === '**/*') {
      return true; // Match everything
    }
    
    if (normalizedPattern.startsWith('**/') && normalizedPattern.endsWith('/**')) {
      // Pattern like **/node_modules/**
      const middle = normalizedPattern.slice(3, -3);
      return normalizedPath.includes('/' + middle + '/') || 
             normalizedPath.startsWith(middle + '/') ||
             normalizedPath.endsWith('/' + middle);
    }
    
    if (normalizedPattern.startsWith('**/')) {
      // Pattern like **/test.ts
      const suffix = normalizedPattern.slice(3);
      return normalizedPath.endsWith('/' + suffix) || normalizedPath === suffix;
    }
    
    if (normalizedPattern.endsWith('/**')) {
      // Pattern like src/**
      const prefix = normalizedPattern.slice(0, -3);
      return normalizedPath.startsWith(prefix + '/') || normalizedPath === prefix;
    }
    
    if (normalizedPattern.includes('**/')) {
      // Pattern like src/**/test.ts
      const parts = normalizedPattern.split('**/');
      if (parts.length === 2) {
        const [prefix, suffix] = parts;
        return normalizedPath.startsWith(prefix) && normalizedPath.endsWith(suffix);
      }
    }
    
    // Simple wildcard matching
    const regexPattern = normalizedPattern
      .replace(/\./g, '\\.')
      .replace(/\*/g, '[^/]*')
      .replace(/\?/g, '[^/]');
    
    const regex = new RegExp('^' + regexPattern + '$', 'i');
    return regex.test(normalizedPath);
  }

  /**
   * Get extraction strategy for a delegation type
   */
  public getExtractionStrategy(delegationType: DelegationType): ExtractionStrategy | undefined {
    return this.extractionStrategies.get(delegationType);
  }

  /**
   * Get all supported delegation types for extraction
   */
  public getSupportedDelegationTypes(): DelegationType[] {
    return Array.from(this.extractionStrategies.keys());
  }
}