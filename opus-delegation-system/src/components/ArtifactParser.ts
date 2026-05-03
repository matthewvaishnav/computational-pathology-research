/**
 * Artifact Parser Component
 * Implements Task 11 - Artifact Reception and Parsing
 * Requirements: 5.1-5.8
 */

import { ArtifactType, ParsedArtifact, MermaidAST, OpenAPISpec, Step, ComplexityLevel } from '../types/core.js';
import { parse as parseYaml } from 'yaml';

/**
 * Error class for artifact parsing failures
 */
export class ArtifactParseError extends Error {
  constructor(
    message: string,
    public readonly lineNumber?: number,
    public readonly field?: string
  ) {
    super(message);
    this.name = 'ArtifactParseError';
  }
}

/**
 * Storage interface for parsed artifacts
 */
export interface ArtifactStorage {
  sessionId: string;
  roundNumber: number;
  artifacts: ParsedArtifact[];
  savedAt: Date;
}

/**
 * Artifact Parser Component
 * Extracts and validates artifacts from Opus-generated text responses
 */
export class ArtifactParser {
  private artifactStorage: Map<string, ArtifactStorage> = new Map();

  /**
   * Parse Opus response text and extract all artifacts
   * Requirement 5.1: Accept copy-pasted text from use.ai
   */
  public parseResponse(
    responseText: string,
    sessionId: string,
    roundNumber: number
  ): ParsedArtifact[] {
    const artifacts: ParsedArtifact[] = [];

    // Extract all code blocks
    const codeBlocks = this.extractCodeBlocks(responseText);

    for (const block of codeBlocks) {
      const artifact = this.parseCodeBlock(block);
      if (artifact) {
        artifacts.push(artifact);
      }
    }

    // Extract implementation guides from markdown sections
    const implementationGuides = this.extractImplementationGuides(responseText);
    artifacts.push(...implementationGuides);

    // Extract test strategies
    const testStrategies = this.extractTestStrategies(responseText);
    artifacts.push(...testStrategies);

    // Store artifacts
    this.storeArtifacts(sessionId, roundNumber, artifacts);

    return artifacts;
  }

  /**
   * Extract fenced code blocks with language identifiers
   * Requirement 5.1: Extract fenced code blocks with language identifiers
   * Task 11.1: Create markdown code block extractor
   */
  private extractCodeBlocks(text: string): Array<{
    language: string;
    content: string;
    startLine: number;
    endLine: number;
    startPos: number;
    endPos: number;
  }> {
    const blocks: Array<{
      language: string;
      content: string;
      startLine: number;
      endLine: number;
      startPos: number;
      endPos: number;
    }> = [];

    // Match fenced code blocks with language identifier
    const codeBlockRegex = /```(\w+)\n([\s\S]*?)```/g;
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      const language = match[1];
      const content = match[2];
      const startPos = match.index;
      const endPos = match.index + match[0].length;

      // Calculate line numbers
      const textBeforeBlock = text.substring(0, startPos);
      const startLine = textBeforeBlock.split('\n').length;
      const endLine = startLine + content.split('\n').length;

      blocks.push({
        language,
        content,
        startLine,
        endLine,
        startPos,
        endPos,
      });
    }

    return blocks;
  }

  /**
   * Parse a code block into a structured artifact
   */
  private parseCodeBlock(block: {
    language: string;
    content: string;
    startLine: number;
    endLine: number;
    startPos: number;
    endPos: number;
  }): ParsedArtifact | null {
    const { language, content, startLine, endLine, startPos, endPos } = block;

    // Determine artifact type based on language and content
    if (language === 'mermaid') {
      return this.parseMermaidDiagram(content, startPos, endPos, startLine);
    } else if (language === 'yaml' || language === 'yml') {
      // Check if it's an OpenAPI spec
      if (this.isOpenAPISpec(content)) {
        return this.parseOpenAPISpec(content, startPos, endPos, startLine);
      }
    }

    // Generic code snippet
    return {
      id: this.generateId(),
      type: ArtifactType.CODE_SNIPPET,
      content,
      metadata: {
        sourceLocation: { start: startPos, end: endPos },
        parseWarnings: [],
        extractedAt: new Date(),
      },
    };
  }

  /**
   * Parse Mermaid diagram
   * Requirement 5.2: Extract Mermaid diagrams and validate syntax
   * Task 11.2: Implement Mermaid diagram parser
   */
  private parseMermaidDiagram(
    content: string,
    startPos: number,
    endPos: number,
    startLine: number
  ): ParsedArtifact {
    const warnings: string[] = [];

    // Basic Mermaid syntax validation
    const validationResult = this.validateMermaidSyntax(content, startLine);
    warnings.push(...validationResult.warnings);

    if (validationResult.errors.length > 0) {
      throw new ArtifactParseError(
        `Mermaid syntax error: ${validationResult.errors[0].message}`,
        validationResult.errors[0].lineNumber
      );
    }

    // Parse Mermaid into AST
    const mermaidAST = this.parseMermaidAST(content);

    return {
      id: this.generateId(),
      type: ArtifactType.MERMAID_DIAGRAM,
      content,
      metadata: {
        sourceLocation: { start: startPos, end: endPos },
        parseWarnings: warnings,
        extractedAt: new Date(),
      },
      structured: {
        mermaid: mermaidAST,
      },
    };
  }

  /**
   * Validate Mermaid syntax
   * Reports errors with line numbers
   */
  private validateMermaidSyntax(
    content: string,
    startLine: number
  ): { errors: Array<{ message: string; lineNumber: number }>; warnings: string[] } {
    const errors: Array<{ message: string; lineNumber: number }> = [];
    const warnings: string[] = [];
    const lines = content.split('\n');

    // Check for diagram type declaration
    const firstLine = lines[0]?.trim();
    const validDiagramTypes = [
      'graph',
      'flowchart',
      'sequenceDiagram',
      'classDiagram',
      'stateDiagram',
      'erDiagram',
      'gantt',
      'pie',
      'gitGraph',
    ];

    const hasValidType = validDiagramTypes.some((type) => firstLine?.startsWith(type));
    if (!hasValidType && lines.length > 0) {
      errors.push({
        message: `Invalid or missing diagram type. Expected one of: ${validDiagramTypes.join(', ')}`,
        lineNumber: startLine,
      });
    }

    // Check for empty content
    if (content.trim().length === 0) {
      errors.push({
        message: 'Empty Mermaid diagram',
        lineNumber: startLine,
      });
    }

    // Basic syntax checks for graph/flowchart
    if (firstLine?.startsWith('graph') || firstLine?.startsWith('flowchart')) {
      // Check for node definitions
      const nodePattern = /[A-Za-z0-9_]+(\[.*?\]|\(.*?\)|\{.*?\})/;
      const hasNodes = lines.some((line) => nodePattern.test(line));
      if (!hasNodes) {
        warnings.push('No nodes found in graph diagram');
      }
    }

    return { errors, warnings };
  }

  /**
   * Parse Mermaid content into AST structure
   */
  private parseMermaidAST(content: string): MermaidAST {
    const lines = content.split('\n').map((l) => l.trim());
    const firstLine = lines[0] || '';

    // Extract diagram type
    const typeMatch = firstLine.match(/^(graph|flowchart|sequenceDiagram|classDiagram)\s*/);
    const type = typeMatch ? typeMatch[1] : 'unknown';

    const nodes: Array<{ id: string; label: string; type: string }> = [];
    const edges: Array<{ from: string; to: string; label?: string }> = [];

    // Parse nodes and edges for graph/flowchart
    if (type === 'graph' || type === 'flowchart') {
      for (const line of lines.slice(1)) {
        // Node definition: A[Label] or A(Label) or A{Label}
        const nodeMatch = line.match(/([A-Za-z0-9_]+)([\[\(\{])(.*?)([\]\)\}])/);
        if (nodeMatch) {
          const [, id, openBracket, label] = nodeMatch;
          const nodeType = this.getNodeType(openBracket);
          nodes.push({ id, label, type: nodeType });
        }

        // Edge definition: A --> B or A -->|label| B
        const edgeMatch = line.match(/([A-Za-z0-9_]+)\s*(-->|---)\s*(?:\|([^|]+)\|)?\s*([A-Za-z0-9_]+)/);
        if (edgeMatch) {
          const [, from, , label, to] = edgeMatch;
          edges.push({ from, to, label: label?.trim() });
        }
      }
    }

    return { type, nodes, edges };
  }

  /**
   * Determine node type from bracket style
   */
  private getNodeType(bracket: string): string {
    switch (bracket) {
      case '[':
        return 'rectangle';
      case '(':
        return 'rounded';
      case '{':
        return 'diamond';
      default:
        return 'default';
    }
  }

  /**
   * Check if YAML content is an OpenAPI specification
   */
  private isOpenAPISpec(content: string): boolean {
    try {
      const parsed = parseYaml(content) as any;
      return parsed && (parsed.openapi || parsed.swagger);
    } catch {
      return false;
    }
  }

  /**
   * Parse OpenAPI specification
   * Requirement 5.3: Extract and validate OpenAPI specs
   * Task 11.3: Implement OpenAPI specification parser
   */
  private parseOpenAPISpec(
    content: string,
    startPos: number,
    endPos: number,
    startLine: number
  ): ParsedArtifact {
    const warnings: string[] = [];

    try {
      const spec = parseYaml(content) as any;

      // Validate OpenAPI structure
      const validationResult = this.validateOpenAPISpec(spec, startLine);
      warnings.push(...validationResult.warnings);

      if (validationResult.errors.length > 0) {
        throw new ArtifactParseError(
          `OpenAPI validation error: ${validationResult.errors[0].message}`,
          validationResult.errors[0].lineNumber,
          validationResult.errors[0].field
        );
      }

      return {
        id: this.generateId(),
        type: ArtifactType.OPENAPI_SPEC,
        content,
        metadata: {
          sourceLocation: { start: startPos, end: endPos },
          parseWarnings: warnings,
          extractedAt: new Date(),
        },
        structured: {
          openapi: spec as OpenAPISpec,
        },
      };
    } catch (error) {
      if (error instanceof ArtifactParseError) {
        throw error;
      }
      throw new ArtifactParseError(
        `Failed to parse YAML: ${error instanceof Error ? error.message : 'Unknown error'}`,
        startLine
      );
    }
  }

  /**
   * Validate OpenAPI specification against OpenAPI 3.0 schema
   */
  private validateOpenAPISpec(
    spec: any,
    startLine: number
  ): { errors: Array<{ message: string; lineNumber: number; field?: string }>; warnings: string[] } {
    const errors: Array<{ message: string; lineNumber: number; field?: string }> = [];
    const warnings: string[] = [];

    // Check required fields
    if (!spec.openapi && !spec.swagger) {
      errors.push({
        message: 'Missing required field: openapi or swagger version',
        lineNumber: startLine,
        field: 'openapi',
      });
    }

    if (!spec.info) {
      errors.push({
        message: 'Missing required field: info',
        lineNumber: startLine,
        field: 'info',
      });
    } else {
      if (!spec.info.title) {
        errors.push({
          message: 'Missing required field: info.title',
          lineNumber: startLine,
          field: 'info.title',
        });
      }
      if (!spec.info.version) {
        errors.push({
          message: 'Missing required field: info.version',
          lineNumber: startLine,
          field: 'info.version',
        });
      }
    }

    if (!spec.paths) {
      errors.push({
        message: 'Missing required field: paths',
        lineNumber: startLine,
        field: 'paths',
      });
    } else if (Object.keys(spec.paths).length === 0) {
      warnings.push('No paths defined in OpenAPI specification');
    }

    // Validate OpenAPI version
    if (spec.openapi && !spec.openapi.startsWith('3.')) {
      warnings.push(`OpenAPI version ${spec.openapi} may not be fully supported. Expected 3.x`);
    }

    return { errors, warnings };
  }

  /**
   * Extract implementation guides from markdown sections
   * Requirement 5.4: Extract implementation guides as structured steps
   * Task 11.4: Implement implementation guide parser
   */
  private extractImplementationGuides(text: string): ParsedArtifact[] {
    const artifacts: ParsedArtifact[] = [];

    // Look for implementation guide sections
    const guidePattern = /#{1,3}\s*(Implementation|Guide|Steps|Plan)[\s\S]*?(?=\n#{1,3}\s|$)/gi;
    let match;

    while ((match = guidePattern.exec(text)) !== null) {
      const content = match[0];
      const startPos = match.index;
      const endPos = startPos + content.length;

      // Parse steps from the guide
      const steps = this.parseImplementationSteps(content);

      if (steps.length > 0) {
        artifacts.push({
          id: this.generateId(),
          type: ArtifactType.IMPLEMENTATION_GUIDE,
          content,
          metadata: {
            sourceLocation: { start: startPos, end: endPos },
            parseWarnings: [],
            extractedAt: new Date(),
          },
          structured: {
            implementationSteps: steps,
          },
        });
      }
    }

    return artifacts;
  }

  /**
   * Parse implementation steps from markdown content
   * Extracts step hierarchy and dependencies
   */
  private parseImplementationSteps(content: string): Step[] {
    const steps: Step[] = [];
    const lines = content.split('\n');

    let currentStep: Partial<Step> | null = null;
    let stepCounter = 0;

    for (const line of lines) {
      // Match numbered steps: 1. Step or 1) Step
      const stepMatch = line.match(/^\s*(\d+)[.)]\s+(.+)/);
      if (stepMatch) {
        // Save previous step
        if (currentStep && currentStep.action) {
          steps.push(this.finalizeStep(currentStep, stepCounter++));
        }

        // Start new step
        const fullAction = stepMatch[2].trim();
        
        // Check for inline dependencies in the action text
        const depMatch = fullAction.match(/\((?:depends on|requires|after):\s*([^)]+)\)/i);
        const action = depMatch ? fullAction.replace(depMatch[0], '').trim() : fullAction;
        const dependencies = depMatch ? depMatch[1].split(',').map((d) => d.trim()) : [];

        currentStep = {
          action,
          description: '',
          dependencies,
        };
        continue;
      }

      // Match sub-steps or bullet points
      const bulletMatch = line.match(/^\s*[-*]\s+(.+)/);
      if (bulletMatch && currentStep) {
        if (!currentStep.description) {
          currentStep.description = bulletMatch[1].trim();
        } else {
          currentStep.description += '\n' + bulletMatch[1].trim();
        }
        continue;
      }

      // Look for dependencies on separate lines
      const depMatch = line.match(/(?:depends on|requires|after):\s*(.+)/i);
      if (depMatch && currentStep) {
        const deps = depMatch[1].split(',').map((d) => d.trim());
        currentStep.dependencies = [...(currentStep.dependencies || []), ...deps];
      }
    }

    // Save last step
    if (currentStep && currentStep.action) {
      steps.push(this.finalizeStep(currentStep, stepCounter++));
    }

    return steps;
  }

  /**
   * Finalize a step with default values
   */
  private finalizeStep(step: Partial<Step>, index: number): Step {
    return {
      id: `step-${index + 1}`,
      action: step.action || '',
      description: step.description || '',
      dependencies: step.dependencies || [],
      complexity: step.complexity || ComplexityLevel.SIMPLE,
      estimatedTime: step.estimatedTime,
    };
  }

  /**
   * Extract test strategies from markdown
   * Requirement 5.5: Extract test strategies including property-based test designs
   * Task 11.5: Implement test strategy parser
   */
  private extractTestStrategies(text: string): ParsedArtifact[] {
    const artifacts: ParsedArtifact[] = [];

    // Look for test strategy sections - match the header and capture content until next header or end
    const lines = text.split('\n');
    let inTestSection = false;
    let sectionStart = -1;
    let sectionContent = '';
    let currentLinePos = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Check if this is a test-related header (only ## level, not ###)
      const headerMatch = line.match(/^##\s+(?:Test|Testing|Test Strategy|Test Plan|Test Approach|Testing Approach)/i);
      
      if (headerMatch) {
        // Save previous section if any
        if (inTestSection && sectionContent && this.containsTestContent(sectionContent)) {
          artifacts.push({
            id: this.generateId(),
            type: ArtifactType.TEST_STRATEGY,
            content: sectionContent.trim(),
            metadata: {
              sourceLocation: { start: sectionStart, end: sectionStart + sectionContent.length },
              parseWarnings: [],
              extractedAt: new Date(),
            },
          });
        }
        
        // Start new section
        inTestSection = true;
        sectionStart = currentLinePos;
        sectionContent = line + '\n';
      } else if (inTestSection) {
        // Check if we hit another ## header (end of test section)
        if (line.match(/^##\s+/)) {
          // Save current section
          if (this.containsTestContent(sectionContent)) {
            artifacts.push({
              id: this.generateId(),
              type: ArtifactType.TEST_STRATEGY,
              content: sectionContent.trim(),
              metadata: {
                sourceLocation: { start: sectionStart, end: sectionStart + sectionContent.length },
                parseWarnings: [],
                extractedAt: new Date(),
              },
            });
          }
          inTestSection = false;
          sectionContent = '';
        } else {
          // Continue accumulating content
          sectionContent += line + '\n';
        }
      }
      
      currentLinePos += line.length + 1; // +1 for newline
    }

    // Save last section if any
    if (inTestSection && sectionContent && this.containsTestContent(sectionContent)) {
      artifacts.push({
        id: this.generateId(),
        type: ArtifactType.TEST_STRATEGY,
        content: sectionContent.trim(),
        metadata: {
          sourceLocation: { start: sectionStart, end: sectionStart + sectionContent.length },
          parseWarnings: [],
          extractedAt: new Date(),
        },
      });
    }

    return artifacts;
  }

  /**
   * Check if content contains test-related information
   */
  private containsTestContent(content: string): boolean {
    const testKeywords = [
      'test case',
      'property',
      'invariant',
      'unit test',
      'integration test',
      'property-based',
      'coverage',
      'assertion',
      'expect',
      'should',
    ];

    const lowerContent = content.toLowerCase();
    return testKeywords.some((keyword) => lowerContent.includes(keyword));
  }

  /**
   * Store parsed artifacts
   * Requirement 5.7: Store parsed artifacts in JSON format
   * Task 11.6: Create artifact storage
   */
  private storeArtifacts(sessionId: string, roundNumber: number, artifacts: ParsedArtifact[]): void {
    const storageKey = `${sessionId}-${roundNumber}`;
    this.artifactStorage.set(storageKey, {
      sessionId,
      roundNumber,
      artifacts,
      savedAt: new Date(),
    });
  }

  /**
   * Retrieve stored artifacts for a session and round
   */
  public getArtifacts(sessionId: string, roundNumber: number): ParsedArtifact[] {
    const storageKey = `${sessionId}-${roundNumber}`;
    const storage = this.artifactStorage.get(storageKey);
    return storage?.artifacts || [];
  }

  /**
   * Get all artifacts for a session across all rounds
   */
  public getAllSessionArtifacts(sessionId: string): ParsedArtifact[] {
    const artifacts: ParsedArtifact[] = [];
    for (const [key, storage] of this.artifactStorage.entries()) {
      if (storage.sessionId === sessionId) {
        artifacts.push(...storage.artifacts);
      }
    }
    return artifacts;
  }

  /**
   * Export artifacts as JSON
   */
  public exportArtifactsAsJSON(sessionId: string, roundNumber?: number): string {
    if (roundNumber !== undefined) {
      const artifacts = this.getArtifacts(sessionId, roundNumber);
      return JSON.stringify(artifacts, null, 2);
    } else {
      const artifacts = this.getAllSessionArtifacts(sessionId);
      return JSON.stringify(artifacts, null, 2);
    }
  }

  /**
   * Clear all stored artifacts (for testing)
   */
  public clearStorage(): void {
    this.artifactStorage.clear();
  }

  /**
   * Generate unique artifact ID
   */
  private generateId(): string {
    return `artifact-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
