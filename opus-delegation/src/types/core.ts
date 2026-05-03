/**
 * Core type definitions for the Opus Delegation System
 */

/**
 * Delegation types categorizing different problem domains
 */
export enum DelegationType {
  ARCHITECTURE_DESIGN = 'architecture_design',
  API_DESIGN = 'api_design',
  TEST_STRATEGY = 'test_strategy',
  INTEGRATION_DESIGN = 'integration_design',
  REFACTORING_ANALYSIS = 'refactoring_analysis',
  FORMAL_VERIFICATION = 'formal_verification',
}

/**
 * Complexity levels for problems and implementation steps
 */
export enum ComplexityLevel {
  SIMPLE = 'simple',
  MODERATE = 'moderate',
  COMPLEX = 'complex',
}

/**
 * Types of context that can be extracted from the codebase
 */
export enum ContextType {
  CODE_SNIPPETS = 'code_snippets',
  REQUIREMENTS_DOCS = 'requirements_docs',
  EXISTING_DESIGNS = 'existing_designs',
  CONSTRAINTS = 'constraints',
  ARCHITECTURE_DOCS = 'architecture_docs',
  API_DEFINITIONS = 'api_definitions',
  TEST_FILES = 'test_files',
  CONFIG_FILES = 'config_files',
  PROTOCOL_SPECS = 'protocol_specs',
  DEPENDENCY_GRAPHS = 'dependency_graphs',
}

/**
 * Types of artifacts that Opus can generate
 */
export enum ArtifactType {
  MERMAID_DIAGRAM = 'mermaid_diagram',
  OPENAPI_SPEC = 'openapi_spec',
  IMPLEMENTATION_GUIDE = 'implementation_guide',
  TEST_STRATEGY = 'test_strategy',
  CODE_SNIPPET = 'code_snippet',
  REQUIREMENTS = 'requirements',
  DESIGN_DOC = 'design_doc',
}

/**
 * Problem classification result
 */
export interface ProblemClassification {
  /** Whether the problem is suitable for Opus delegation */
  suitable: boolean;
  /** Primary delegation type */
  delegationType: DelegationType;
  /** Alternative delegation types if applicable */
  alternativeTypes?: DelegationType[];
  /** Estimated complexity of the problem */
  complexity: ComplexityLevel;
  /** Required context types for this problem */
  requiredContextTypes: ContextType[];
  /** Expected artifact types from Opus */
  expectedArtifacts: ArtifactType[];
  /** Suitability score (0-100) */
  suitabilityScore: number;
  /** Reasoning for the classification */
  reasoning: string;
}

/**
 * Context extraction strategy
 */
export interface ExtractionStrategy {
  /** Primary file patterns to search */
  primaryPatterns: string[];
  /** Secondary file patterns */
  secondaryPatterns: string[];
  /** Keywords for semantic search */
  keywords: string[];
  /** Maximum number of files to extract */
  maxFiles: number;
  /** Whether to include dependency analysis */
  includeDependencies: boolean;
}

/**
 * Code snippet with metadata
 */
export interface CodeSnippet {
  /** File path relative to repository root */
  filePath: string;
  /** Starting line number (1-indexed) */
  startLine: number;
  /** Ending line number (1-indexed) */
  endLine: number;
  /** Code content */
  content: string;
  /** Programming language */
  language: string;
  /** Relevance score (0-1) */
  relevance: number;
}

/**
 * Documentation excerpt
 */
export interface DocumentationExcerpt {
  /** Source file path */
  sourcePath: string;
  /** Section title or heading */
  title: string;
  /** Content */
  content: string;
  /** Relevance score (0-1) */
  relevance: number;
}

/**
 * Context bundle containing all extracted context
 */
export interface ContextBundle {
  /** Unique identifier */
  id: string;
  /** Problem description */
  problemDescription: string;
  /** Code snippets */
  codeSnippets: CodeSnippet[];
  /** Documentation excerpts */
  documentation: DocumentationExcerpt[];
  /** Constraints */
  constraints: string[];
  /** Total character count */
  totalSize: number;
  /** Whether content was truncated */
  truncated: boolean;
  /** Summary of excluded content if truncated */
  excludedSummary?: string;
  /** Context manifest */
  manifest: ContextManifestEntry[];
}

/**
 * Entry in the context manifest
 */
export interface ContextManifestEntry {
  /** Source file or document */
  source: string;
  /** Type of content */
  type: 'code' | 'documentation' | 'config';
  /** Size in characters */
  size: number;
  /** Relevance level */
  relevance: 'high' | 'medium' | 'low';
}

/**
 * Template parameter definition
 */
export interface TemplateParameter {
  /** Parameter name */
  name: string;
  /** Whether the parameter is required */
  required: boolean;
  /** Parameter type */
  type: 'string' | 'number' | 'boolean' | 'list';
  /** Default value if not required */
  default?: string | number | boolean | string[];
  /** Description of the parameter */
  description?: string;
}

/**
 * Delegation template
 */
export interface DelegationTemplate {
  /** Unique template identifier */
  templateId: string;
  /** Human-readable name */
  name: string;
  /** Template category */
  category: DelegationType;
  /** Template version */
  version: string;
  /** Template parameters */
  parameters: TemplateParameter[];
  /** Required context types */
  contextRequirements: ContextType[];
  /** Expected artifact types */
  expectedArtifacts: Array<{
    type: ArtifactType;
    subtype?: string;
    format?: string;
    granularity?: string;
  }>;
  /** Prompt template with placeholders */
  promptTemplate: string;
  /** Usage count for statistics */
  usageCount?: number;
}

/**
 * Delegation request ready for copy-paste to use.ai
 */
export interface DelegationRequest {
  /** Unique request identifier */
  id: string;
  /** Session identifier */
  sessionId: string;
  /** Round number in multi-round session */
  roundNumber: number;
  /** Problem title */
  title: string;
  /** Problem description */
  description: string;
  /** Objectives */
  objectives: string[];
  /** Constraints */
  constraints: string[];
  /** Expected artifacts */
  expectedArtifacts: ArtifactType[];
  /** Output format requirements */
  formatRequirements: string;
  /** Context bundle */
  contextBundle: ContextBundle;
  /** Formatted text ready for copy-paste */
  formattedText: string;
  /** Timestamp */
  createdAt: Date;
}

/**
 * Parsed artifact from Opus response
 */
export interface ParsedArtifact {
  /** Unique artifact identifier */
  id: string;
  /** Artifact type */
  type: ArtifactType;
  /** Raw content */
  content: string;
  /** Source location in response */
  metadata: {
    sourceLocation: { start: number; end: number };
    parseWarnings: string[];
    extractedAt: Date;
  };
  /** Structured representation (type-specific) */
  structured?: MermaidDiagram | OpenAPISpec | ImplementationGuide | TestStrategy;
}

/**
 * Mermaid diagram artifact
 */
export interface MermaidDiagram {
  /** Diagram type (graph, sequence, class, etc.) */
  diagramType: string;
  /** Raw Mermaid syntax */
  syntax: string;
  /** Validation result */
  valid: boolean;
  /** Validation errors if any */
  errors?: string[];
}

/**
 * OpenAPI specification artifact
 */
export interface OpenAPISpec {
  /** OpenAPI version */
  version: string;
  /** Parsed YAML/JSON content */
  spec: Record<string, unknown>;
  /** Validation result */
  valid: boolean;
  /** Validation errors if any */
  errors?: string[];
}

/**
 * Implementation step
 */
export interface ImplementationStep {
  /** Step identifier */
  id: string;
  /** Step number */
  number: string;
  /** Action description */
  action: string;
  /** File path or location */
  filePath?: string;
  /** Dependencies (step IDs) */
  dependencies: string[];
  /** Complexity estimate */
  complexity: ComplexityLevel;
  /** Expected outcome */
  expectedOutcome?: string;
  /** Code template or boilerplate */
  codeTemplate?: string;
  /** Artifact references */
  artifactReferences?: string[];
}

/**
 * Implementation guide artifact
 */
export interface ImplementationGuide {
  /** Guide title */
  title: string;
  /** Prerequisites */
  prerequisites: string[];
  /** Implementation phases */
  phases: Array<{
    name: string;
    complexity: ComplexityLevel;
    steps: ImplementationStep[];
  }>;
  /** Risk register */
  risks?: Array<{
    risk: string;
    mitigation: string;
    owner?: string;
  }>;
}

/**
 * Test strategy artifact
 */
export interface TestStrategy {
  /** Coverage targets */
  coverageTargets: {
    unit?: number;
    integration?: number;
    e2e?: number;
  };
  /** Property-based tests */
  propertyTests: Array<{
    name: string;
    property: string;
    generator?: string;
  }>;
  /** Edge cases */
  edgeCases: string[];
  /** Test data requirements */
  testDataRequirements: string[];
}

/**
 * Validation result for artifacts
 */
export interface ValidationResult {
  /** Whether the artifact is valid */
  valid: boolean;
  /** Completeness score (0-100) */
  completenessScore: number;
  /** Quality scores by dimension */
  qualityScores: {
    completeness: number;
    clarity: number;
    implementability: number;
  };
  /** Missing required elements */
  missingElements: string[];
  /** Validation errors */
  errors: string[];
  /** Warnings */
  warnings: string[];
  /** Follow-up questions for incomplete artifacts */
  followUpQuestions?: string[];
}

/**
 * Delegation round in a multi-round session
 */
export interface DelegationRound {
  /** Round number */
  roundNumber: number;
  /** Delegation request */
  request: DelegationRequest;
  /** Opus response (raw text) */
  response?: string;
  /** Parsed artifacts */
  artifacts: ParsedArtifact[];
  /** Validation results */
  validation?: ValidationResult;
  /** Timestamp */
  timestamp: Date;
}

/**
 * Session metrics
 */
export interface SessionMetrics {
  /** Total time spent (milliseconds) */
  totalTime: number;
  /** Context size (characters) */
  contextSize: number;
  /** Number of rounds */
  roundCount: number;
  /** Final completeness score */
  finalCompleteness: number;
  /** Estimated cost */
  estimatedCost?: number;
}

/**
 * Delegation session
 */
export interface DelegationSession {
  /** Unique session identifier */
  id: string;
  /** Creation timestamp */
  createdAt: Date;
  /** Last update timestamp */
  updatedAt: Date;
  /** Problem information */
  problem: {
    title: string;
    description: string;
    type: DelegationType;
    complexity: ComplexityLevel;
  };
  /** Delegation rounds */
  rounds: DelegationRound[];
  /** Final artifacts */
  finalArtifacts: ParsedArtifact[];
  /** Implementation guide */
  implementationGuide?: ImplementationGuide;
  /** Session metrics */
  metrics: SessionMetrics;
  /** Session status */
  status: 'active' | 'completed' | 'abandoned';
}
