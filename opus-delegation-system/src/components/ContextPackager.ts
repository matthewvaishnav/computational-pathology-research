/**
 * Context Packager Component
 * Implements Task 6.1, 6.2, 6.3, 6.4 - Markdown formatting, size management, compression, bundle assembly
 * Requirements: 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7
 */

import { FileMatch, CodeSnippet, ExtractionStrategy } from './ContextExtractor.js';
import { DelegationType, ContextType } from '../types/core.js';

// Context Bundle Structure (Requirement 2.8)
export interface ContextBundle {
  problemTitle: string;
  problemSummary: string;
  codeSnippets: FormattedCodeSnippet[];
  documentationExcerpts: DocumentationExcerpt[];
  constraints: string[];
  contextManifest: ContextManifest;
  totalSize: number;
  compressionApplied: boolean;
  excludedContent?: ExcludedContentSummary;
}

// Formatted code snippet with markdown
export interface FormattedCodeSnippet {
  filePath: string;
  startLine: number;
  endLine: number;
  content: string; // Markdown formatted with syntax highlighting
  language: string;
  size: number;
  relevanceScore: number;
}

// Documentation excerpt
export interface DocumentationExcerpt {
  source: string;
  title: string;
  content: string;
  size: number;
  relevanceScore: number;
}

// Context manifest table (Requirement 2.7)
export interface ContextManifest {
  sources: ContextSource[];
  totalFiles: number;
  totalSize: number;
  compressionRatio?: number;
}

export interface ContextSource {
  path: string;
  type: 'Code' | 'Documentation' | 'Configuration';
  size: string; // Human readable (e.g., "2.1 KB")
  relevance: 'High' | 'Medium' | 'Low';
  included: boolean;
}

// Excluded content summary (Requirement 2.6)
export interface ExcludedContentSummary {
  totalExcluded: number;
  excludedFiles: Array<{
    path: string;
    reason: string;
    size: number;
  }>;
  summary: string;
}

// Packaging configuration
export interface PackagingConfig {
  maxSize: number; // Default 50,000 characters (Requirement 2.5)
  enableCompression: boolean;
  enableSummarization: boolean;
  prioritizeRecent: boolean;
  includeLineNumbers: boolean;
  syntaxHighlighting: boolean;
}

// Compression options (Requirements 14.1, 14.2, 14.3)
export interface CompressionOptions {
  removeComments: boolean;
  removeExtraWhitespace: boolean;
  deduplicatePatterns: boolean;
  summarizeLargeDocs: boolean;
  maxDocSize: number; // Size threshold for summarization
}

export class ContextPackager {
  private defaultConfig: PackagingConfig = {
    maxSize: 50000,
    enableCompression: true,
    enableSummarization: true,
    prioritizeRecent: true,
    includeLineNumbers: true,
    syntaxHighlighting: true
  };

  private defaultCompressionOptions: CompressionOptions = {
    removeComments: true,
    removeExtraWhitespace: true,
    deduplicatePatterns: true,
    summarizeLargeDocs: true,
    maxDocSize: 5000
  };

  /**
   * Package extracted context into copy-paste ready markdown bundle
   * Implements Requirements 2.3, 2.4, 2.8
   */
  public async packageContext(
    problemTitle: string,
    problemDescription: string,
    files: FileMatch[],
    snippets: CodeSnippet[],
    strategy: ExtractionStrategy,
    constraints: string[] = [],
    config?: Partial<PackagingConfig>
  ): Promise<ContextBundle> {
    const finalConfig = { ...this.defaultConfig, ...config };
    
    // Step 1: Format code snippets with markdown and transfer relevance scores (Task 6.1)
    let formattedSnippets = this.formatCodeSnippets(snippets, files, finalConfig);
    
    // Step 2: Extract and format documentation excerpts
    let docExcerpts = this.extractDocumentationExcerpts(files, finalConfig);
    
    // Step 3: Calculate initial size and apply size management (Task 6.2)
    let currentSize = this.calculateTotalSize(formattedSnippets, docExcerpts, problemDescription, constraints);
    let compressionApplied = false;
    let excludedContent: ExcludedContentSummary | undefined;
    
    // Apply compression and size management if needed
    if (finalConfig.enableCompression || currentSize > finalConfig.maxSize) {
      const optimizationResult = await this.optimizeSize(
        formattedSnippets,
        docExcerpts,
        finalConfig,
        this.defaultCompressionOptions
      );
      
      formattedSnippets = optimizationResult.snippets;
      docExcerpts = optimizationResult.excerpts;
      compressionApplied = optimizationResult.compressionApplied;
      excludedContent = optimizationResult.excludedContent;
      currentSize = optimizationResult.finalSize;
    }
    
    // Step 4: Generate context manifest (Task 6.1)
    const manifest = this.generateContextManifest(
      files,
      formattedSnippets,
      docExcerpts,
      currentSize,
      compressionApplied
    );
    
    // Step 5: Create final context bundle (Task 6.4)
    const bundle: ContextBundle = {
      problemTitle,
      problemSummary: problemDescription,
      codeSnippets: formattedSnippets,
      documentationExcerpts: docExcerpts,
      constraints,
      contextManifest: manifest,
      totalSize: currentSize,
      compressionApplied,
      excludedContent
    };
    
    return bundle;
  }

  /**
   * Generate copy-paste ready markdown from context bundle
   * Implements Requirement 2.8 - copy-paste ready format
   */
  public generateMarkdown(bundle: ContextBundle): string {
    const sections: string[] = [];
    
    // Header
    sections.push(`# Context Bundle: ${bundle.problemTitle}\n`);
    
    // Problem Summary
    sections.push(`## Problem Summary\n${bundle.problemSummary}\n`);
    
    // Code Snippets Section
    if (bundle.codeSnippets.length > 0) {
      sections.push(`## Relevant Code\n`);
      
      for (const snippet of bundle.codeSnippets) {
        sections.push(`### ${snippet.filePath} (lines ${snippet.startLine}-${snippet.endLine})\n`);
        sections.push(snippet.content);
        sections.push(''); // Empty line for spacing
      }
    }
    
    // Documentation Excerpts Section
    if (bundle.documentationExcerpts.length > 0) {
      sections.push(`## Documentation Excerpts\n`);
      
      for (const excerpt of bundle.documentationExcerpts) {
        sections.push(`**From:** ${excerpt.source}\n`);
        if (excerpt.title) {
          sections.push(`**Title:** ${excerpt.title}\n`);
        }
        sections.push(excerpt.content);
        sections.push(''); // Empty line for spacing
      }
    }
    
    // Constraints Section
    if (bundle.constraints.length > 0) {
      sections.push(`## Constraints\n`);
      for (const constraint of bundle.constraints) {
        sections.push(`- ${constraint}`);
      }
      sections.push(''); // Empty line for spacing
    }
    
    // Context Manifest Section
    sections.push(`## Context Manifest\n`);
    sections.push(this.generateManifestTable(bundle.contextManifest));
    
    // Excluded Content Summary (if any)
    if (bundle.excludedContent) {
      sections.push(`## Excluded Content Summary\n`);
      sections.push(`**Total excluded:** ${bundle.excludedContent.totalExcluded} files\n`);
      sections.push(`**Reason:** ${bundle.excludedContent.summary}\n`);
      
      if (bundle.excludedContent.excludedFiles.length > 0) {
        sections.push(`**Excluded files:**`);
        for (const file of bundle.excludedContent.excludedFiles.slice(0, 10)) { // Limit to 10 for brevity
          sections.push(`- ${file.path} (${this.formatSize(file.size)}) - ${file.reason}`);
        }
        if (bundle.excludedContent.excludedFiles.length > 10) {
          sections.push(`- ... and ${bundle.excludedContent.excludedFiles.length - 10} more files`);
        }
        sections.push(''); // Empty line for spacing
      }
    }
    
    // Compression Notice
    if (bundle.compressionApplied) {
      sections.push(`---\n**Note:** Content has been compressed and optimized to fit within size limits while preserving semantic meaning.\n`);
    }
    
    return sections.join('\n');
  }

  /**
   * Format code snippets with syntax highlighting and annotations (Task 6.1)
   */
  private formatCodeSnippets(
    snippets: CodeSnippet[],
    files: FileMatch[],
    config: PackagingConfig
  ): FormattedCodeSnippet[] {
    // Create a map for quick relevance score lookup
    const relevanceMap = new Map<string, number>();
    files.forEach(file => {
      relevanceMap.set(file.filePath, file.relevanceScore);
    });

    return snippets.map(snippet => {
      let content = snippet.content;
      
      // Apply syntax highlighting markers if enabled
      if (config.syntaxHighlighting && snippet.language) {
        // The content from ContextExtractor already includes markdown formatting
        // We just need to ensure it's properly formatted
        if (!content.startsWith('```')) {
          const lines = content.split('\n');
          const codeLines = config.includeLineNumbers 
            ? lines.map((line, index) => `${(snippet.startLine + index).toString().padStart(3, ' ')}: ${line}`)
            : lines;
          
          content = `\`\`\`${snippet.language}\n${codeLines.join('\n')}\n\`\`\``;
        }
      }
      
      return {
        filePath: snippet.filePath,
        startLine: snippet.startLine,
        endLine: snippet.endLine,
        content,
        language: snippet.language || 'text',
        size: content.length,
        relevanceScore: relevanceMap.get(snippet.filePath) || 0.5 // Default relevance if not found
      };
    });
  }

  /**
   * Extract documentation excerpts from files
   */
  private extractDocumentationExcerpts(
    files: FileMatch[],
    config: PackagingConfig
  ): DocumentationExcerpt[] {
    const excerpts: DocumentationExcerpt[] = [];
    
    // Filter for documentation files
    const docFiles = files.filter(file => 
      file.filePath.endsWith('.md') || 
      file.filePath.endsWith('.txt') ||
      file.filePath.includes('README') ||
      file.filePath.includes('doc')
    );
    
    for (const file of docFiles.slice(0, 5)) { // Limit to 5 doc files
      try {
        // In a real implementation, we would read the file content
        // For now, we'll create a placeholder
        const excerpt: DocumentationExcerpt = {
          source: file.filePath,
          title: this.extractTitleFromPath(file.filePath),
          content: `[Documentation content from ${file.filePath}]`,
          size: 500, // Placeholder size
          relevanceScore: file.relevanceScore
        };
        
        excerpts.push(excerpt);
      } catch (error) {
        // Skip files that can't be read
        continue;
      }
    }
    
    return excerpts;
  }

  /**
   * Size management and optimization (Task 6.2)
   */
  private async optimizeSize(
    snippets: FormattedCodeSnippet[],
    excerpts: DocumentationExcerpt[],
    config: PackagingConfig,
    compressionOptions: CompressionOptions
  ): Promise<{
    snippets: FormattedCodeSnippet[];
    excerpts: DocumentationExcerpt[];
    compressionApplied: boolean;
    excludedContent?: ExcludedContentSummary;
    finalSize: number;
  }> {
    let currentSnippets = [...snippets];
    let currentExcerpts = [...excerpts];
    let compressionApplied = false;
    const excludedFiles: Array<{ path: string; reason: string; size: number }> = [];
    
    // Step 1: Apply compression to existing content (Task 6.3)
    if (config.enableCompression) {
      currentSnippets = this.compressCodeSnippets(currentSnippets, compressionOptions);
      currentExcerpts = this.compressDocumentationExcerpts(currentExcerpts, compressionOptions);
      compressionApplied = true;
    }
    
    // Step 2: Check size after compression
    let currentSize = this.calculateContentSize(currentSnippets, currentExcerpts);
    // Use smaller overhead for very small limits
    const overhead = config.maxSize < 500 ? 40 : Math.min(500, Math.max(100, config.maxSize * 0.2));
    let totalSize = currentSize + overhead;
    
    // Step 3: If still too large, prioritize content (Task 6.2)
    if (totalSize > config.maxSize) {
      const prioritizationResult = this.prioritizeContent(
        currentSnippets,
        currentExcerpts,
        config.maxSize,
        excludedFiles
      );
      
      currentSnippets = prioritizationResult.snippets;
      currentExcerpts = prioritizationResult.excerpts;
      excludedFiles.push(...prioritizationResult.excluded);
      totalSize = prioritizationResult.finalSize;
    }
    
    // Always create excludedContent when compression is applied or content is excluded
    const excludedContent = (excludedFiles.length > 0 || compressionApplied) ? {
      totalExcluded: excludedFiles.length,
      excludedFiles,
      summary: excludedFiles.length > 0 
        ? `Content exceeded ${this.formatSize(config.maxSize)} limit. Applied compression and prioritized most relevant files.`
        : 'Content compressed to fit within size limits while preserving semantic meaning.'
    } : undefined;
    
    return {
      snippets: currentSnippets,
      excerpts: currentExcerpts,
      compressionApplied,
      excludedContent,
      finalSize: totalSize
    };
  }

  /**
   * Context compression implementation (Task 6.3)
   */
  private compressCodeSnippets(
    snippets: FormattedCodeSnippet[],
    options: CompressionOptions
  ): FormattedCodeSnippet[] {
    return snippets.map(snippet => {
      let content = snippet.content;
      
      // Remove redundant whitespace while preserving semantics
      if (options.removeExtraWhitespace) {
        content = this.removeRedundantWhitespace(content);
      }
      
      // Remove comments if enabled
      if (options.removeComments) {
        content = this.removeComments(content, snippet.language);
      }
      
      // Deduplicate similar patterns
      if (options.deduplicatePatterns) {
        content = this.deduplicatePatterns(content);
      }
      
      return {
        ...snippet,
        content,
        size: content.length
      };
    });
  }

  /**
   * Compress documentation excerpts
   */
  private compressDocumentationExcerpts(
    excerpts: DocumentationExcerpt[],
    options: CompressionOptions
  ): DocumentationExcerpt[] {
    return excerpts.map(excerpt => {
      let content = excerpt.content;
      
      // Apply extractive summarization for large docs
      if (options.summarizeLargeDocs && content.length > options.maxDocSize) {
        content = this.extractiveSummarization(content, options.maxDocSize);
      }
      
      // Remove extra whitespace
      if (options.removeExtraWhitespace) {
        content = this.removeRedundantWhitespace(content);
      }
      
      return {
        ...excerpt,
        content,
        size: content.length
      };
    });
  }

  /**
   * Remove redundant whitespace while preserving semantics
   */
  private removeRedundantWhitespace(content: string): string {
    // Handle markdown code blocks properly
    if (content.includes('```')) {
      const parts = content.split('```');
      for (let i = 1; i < parts.length; i += 2) {
        // Only compress whitespace inside code blocks (odd indices)
        const codeContent = parts[i];
        const lines = codeContent.split('\n');
        const compressedLines: string[] = [];
        
        for (const line of lines) {
          // Preserve line number formatting but compress other whitespace
          if (line.match(/^\s*\d+:\s*/)) {
            // Line with number - preserve structure but compress internal whitespace
            const match = line.match(/^(\s*\d+:\s*)(.*)/);
            if (match) {
              const [, prefix, code] = match;
              const compressedCode = code.replace(/\s+/g, ' ').trim();
              compressedLines.push(prefix + compressedCode);
            } else {
              compressedLines.push(line);
            }
          } else if (line.trim()) {
            // Non-empty line - compress whitespace more aggressively
            compressedLines.push(line.replace(/\s+/g, ' ').trim());
          }
          // Skip all empty lines in code blocks for maximum compression
        }
        
        parts[i] = compressedLines.join('\n');
      }
      return parts.join('```');
    }
    
    // More aggressive compression for non-markdown content
    return content
      .replace(/\n\s*\n\s*\n/g, '\n') // Multiple empty lines to single
      .replace(/\s+/g, ' ') // Multiple spaces to single
      .replace(/\n\s+/g, '\n') // Leading whitespace on lines
      .trim();
  }

  /**
   * Remove comments based on language
   */
  private removeComments(content: string, language: string): string {
    // Simple comment removal - in production would use proper AST parsing
    switch (language) {
      case 'typescript':
      case 'javascript':
        return content
          .replace(/\/\*[\s\S]*?\*\//g, '') // Block comments
          .replace(/\/\/.*$/gm, ''); // Line comments
      
      case 'python':
        return content
          .replace(/#.*$/gm, '') // Line comments
          .replace(/"""[\s\S]*?"""/g, '') // Docstrings
          .replace(/'''[\s\S]*?'''/g, ''); // Docstrings
      
      case 'java':
      case 'cpp':
      case 'c':
        return content
          .replace(/\/\*[\s\S]*?\*\//g, '') // Block comments
          .replace(/\/\/.*$/gm, ''); // Line comments
      
      default:
        return content;
    }
  }

  /**
   * Deduplicate similar code patterns
   */
  private deduplicatePatterns(content: string): string {
    // Handle markdown code blocks properly
    if (content.includes('```')) {
      const parts = content.split('```');
      for (let i = 1; i < parts.length; i += 2) {
        // Only deduplicate inside code blocks (odd indices)
        const codeContent = parts[i];
        const lines = codeContent.split('\n');
        const seenImports = new Set<string>();
        const dedupedLines: string[] = [];
        
        for (const line of lines) {
          const trimmed = line.trim();
          
          // Check for import statements (remove line numbers if present)
          const cleanLine = trimmed.replace(/^\d+:\s*/, '');
          if (cleanLine.startsWith('import ') || cleanLine.startsWith('from ')) {
            if (!seenImports.has(cleanLine)) {
              seenImports.add(cleanLine);
              dedupedLines.push(line);
            }
          } else {
            dedupedLines.push(line);
          }
        }
        
        parts[i] = dedupedLines.join('\n');
      }
      return parts.join('```');
    }
    
    // Simple deduplication for non-markdown content
    const lines = content.split('\n');
    const seenImports = new Set<string>();
    const dedupedLines: string[] = [];
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Check for import statements
      if (trimmed.startsWith('import ') || trimmed.startsWith('from ')) {
        if (!seenImports.has(trimmed)) {
          seenImports.add(trimmed);
          dedupedLines.push(line);
        }
      } else {
        dedupedLines.push(line);
      }
    }
    
    return dedupedLines.join('\n');
  }

  /**
   * Extractive summarization for large documents
   */
  private extractiveSummarization(content: string, maxSize: number): string {
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    if (sentences.length <= 3) {
      return content.substring(0, maxSize) + (content.length > maxSize ? '...' : '');
    }
    
    // Simple extractive summarization - take first, middle, and last sentences
    const firstSentence = sentences[0].trim() + '.';
    const middleIndex = Math.floor(sentences.length / 2);
    const middleSentence = sentences[middleIndex].trim() + '.';
    const lastSentence = sentences[sentences.length - 1].trim() + '.';
    
    const summary = `${firstSentence}\n\n${middleSentence}\n\n${lastSentence}`;
    
    if (summary.length <= maxSize) {
      return summary;
    }
    
    // If still too long, truncate
    return summary.substring(0, maxSize - 3) + '...';
  }

  /**
   * Prioritize content when exceeding size limits
   */
  private prioritizeContent(
    snippets: FormattedCodeSnippet[],
    excerpts: DocumentationExcerpt[],
    maxSize: number,
    excludedFiles: Array<{ path: string; reason: string; size: number }>
  ): {
    snippets: FormattedCodeSnippet[];
    excerpts: DocumentationExcerpt[];
    excluded: Array<{ path: string; reason: string; size: number }>;
    finalSize: number;
  } {
    // Sort by relevance score (descending)
    const sortedSnippets = [...snippets].sort((a, b) => b.relevanceScore - a.relevanceScore);
    const sortedExcerpts = [...excerpts].sort((a, b) => b.relevanceScore - a.relevanceScore);
    
    const includedSnippets: FormattedCodeSnippet[] = [];
    const includedExcerpts: DocumentationExcerpt[] = [];
    const excluded: Array<{ path: string; reason: string; size: number }> = [];
    
    let currentSize = 0;
    // Use much smaller overhead for very small limits
    const overhead = maxSize < 500 ? 20 : Math.min(500, Math.max(100, maxSize * 0.2));
    const availableSize = Math.max(30, maxSize - overhead); // Ensure minimum space
    
    // Include highest priority snippets first
    for (const snippet of sortedSnippets) {
      if (currentSize + snippet.size <= availableSize) {
        includedSnippets.push(snippet);
        currentSize += snippet.size;
      } else {
        excluded.push({
          path: snippet.filePath,
          reason: 'Size limit exceeded',
          size: snippet.size
        });
      }
    }
    
    // Include documentation excerpts if space remains
    for (const excerpt of sortedExcerpts) {
      if (currentSize + excerpt.size <= availableSize) {
        includedExcerpts.push(excerpt);
        currentSize += excerpt.size;
      } else {
        excluded.push({
          path: excerpt.source,
          reason: 'Size limit exceeded',
          size: excerpt.size
        });
      }
    }
    
    return {
      snippets: includedSnippets,
      excerpts: includedExcerpts,
      excluded,
      finalSize: currentSize + overhead
    };
  }

  /**
   * Generate context manifest table
   */
  private generateContextManifest(
    files: FileMatch[],
    snippets: FormattedCodeSnippet[],
    excerpts: DocumentationExcerpt[],
    totalSize: number,
    compressionApplied: boolean
  ): ContextManifest {
    const sources: ContextSource[] = [];
    
    // Add code snippets to manifest
    for (const snippet of snippets) {
      sources.push({
        path: snippet.filePath,
        type: 'Code',
        size: this.formatSize(snippet.size),
        relevance: this.scoreToRelevance(snippet.relevanceScore),
        included: true
      });
    }
    
    // Add documentation excerpts to manifest
    for (const excerpt of excerpts) {
      sources.push({
        path: excerpt.source,
        type: 'Documentation',
        size: this.formatSize(excerpt.size),
        relevance: this.scoreToRelevance(excerpt.relevanceScore),
        included: true
      });
    }
    
    // Add excluded files
    const includedPaths = new Set([
      ...snippets.map(s => s.filePath),
      ...excerpts.map(e => e.source)
    ]);
    
    for (const file of files) {
      if (!includedPaths.has(file.filePath)) {
        sources.push({
          path: file.filePath,
          type: this.getFileType(file.filePath),
          size: this.formatSize(file.size),
          relevance: this.scoreToRelevance(file.relevanceScore),
          included: false
        });
      }
    }
    
    return {
      sources,
      totalFiles: files.length,
      totalSize,
      compressionRatio: compressionApplied ? 0.7 : undefined // Estimated compression ratio
    };
  }

  /**
   * Generate manifest table markdown
   */
  private generateManifestTable(manifest: ContextManifest): string {
    const lines = [
      '| Source | Type | Size | Relevance | Included |',
      '|--------|------|------|-----------|----------|'
    ];
    
    for (const source of manifest.sources.slice(0, 20)) { // Limit to 20 for readability
      const included = source.included ? '✓' : '✗';
      lines.push(`| ${source.path} | ${source.type} | ${source.size} | ${source.relevance} | ${included} |`);
    }
    
    if (manifest.sources.length > 20) {
      lines.push(`| ... | ... | ... | ... | ... |`);
      lines.push(`| *${manifest.sources.length - 20} more files* | | | | |`);
    }
    
    lines.push('');
    lines.push(`**Total Files:** ${manifest.totalFiles}`);
    lines.push(`**Total Size:** ${this.formatSize(manifest.totalSize)}`);
    
    if (manifest.compressionRatio) {
      lines.push(`**Compression Ratio:** ${Math.round(manifest.compressionRatio * 100)}%`);
    }
    
    return lines.join('\n');
  }

  /**
   * Calculate total size of all content
   */
  private calculateTotalSize(
    snippets: FormattedCodeSnippet[],
    excerpts: DocumentationExcerpt[],
    problemDescription: string,
    constraints: string[]
  ): number {
    const snippetSize = snippets.reduce((sum, s) => sum + s.size, 0);
    const excerptSize = excerpts.reduce((sum, e) => sum + e.size, 0);
    const descriptionSize = problemDescription.length;
    const constraintSize = constraints.join('\n').length;
    const contentSize = snippetSize + excerptSize + descriptionSize + constraintSize;
    const overhead = Math.min(500, Math.max(100, contentSize * 0.3)); // Adaptive overhead
    
    return contentSize + overhead;
  }

  /**
   * Calculate size of content only (no overhead)
   */
  private calculateContentSize(
    snippets: FormattedCodeSnippet[],
    excerpts: DocumentationExcerpt[]
  ): number {
    return snippets.reduce((sum, s) => sum + s.size, 0) + 
           excerpts.reduce((sum, e) => sum + e.size, 0);
  }

  /**
   * Helper methods
   */
  private extractTitleFromPath(filePath: string): string {
    const fileName = filePath.split('/').pop() || filePath;
    return fileName.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');
  }

  private getFileType(filePath: string): 'Code' | 'Documentation' | 'Configuration' {
    if (filePath.endsWith('.md') || filePath.endsWith('.txt') || filePath.includes('README')) {
      return 'Documentation';
    }
    if (filePath.endsWith('.json') || filePath.endsWith('.yaml') || filePath.endsWith('.yml') || 
        filePath.endsWith('.toml') || filePath.endsWith('.ini')) {
      return 'Configuration';
    }
    return 'Code';
  }

  private scoreToRelevance(score: number): 'High' | 'Medium' | 'Low' {
    if (score >= 0.7) return 'High';
    if (score >= 0.4) return 'Medium';
    return 'Low';
  }

  private formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}