// Core type definitions for Opus Delegation System

export type DelegationType =
  | 'architecture_design'
  | 'api_design'
  | 'test_strategy'
  | 'integration_design'
  | 'refactoring_analysis'
  | 'formal_verification';

export type ComplexityLevel = 'simple' | 'moderate' | 'complex';

export type ContextType =
  | 'code_snippets'
  | 'requirements_docs'
  | 'existing_designs'
  | 'constraints'
  | 'architecture_docs'
  | 'api_endpoints'
  | 'test_files'
  | 'external_interfaces'
  | 'dependency_graphs';

export type ArtifactType =
  | 'mermaid_diagram'
  | 'openapi_spec'
  | 'implementation_guide'
  | 'test_strategy'
  | 'code_snippet';

// Enum-like objects for easier usage
export const DelegationType = {
  ARCHITECTURE_DESIGN: 'architecture_design' as const,
  API_DESIGN: 'api_design' as const,
  TEST_STRATEGY: 'test_strategy' as const,
  INTEGRATION_DESIGN: 'integration_design' as const,
  REFACTORING_ANALYSIS: 'refactoring_analysis' as const,
  FORMAL_VERIFICATION: 'formal_verification' as const
} as const;

export const ComplexityLevel = {
  SIMPLE: 'simple' as const,
  MODERATE: 'moderate' as const,
  COMPLEX: 'complex' as const
} as const;

export const ContextType = {
  CODE_SNIPPETS: 'code_snippets' as const,
  REQUIREMENTS_DOCS: 'requirements_docs' as const,
  EXISTING_DESIGNS: 'existing_designs' as const,
  CONSTRAINTS: 'constraints' as const,
  ARCHITECTURE_DOCS: 'architecture_docs' as const,
  API_ENDPOINTS: 'api_endpoints' as const,
  TEST_FILES: 'test_files' as const,
  EXTERNAL_INTERFACES: 'external_interfaces' as const,
  DEPENDENCY_GRAPHS: 'dependency_graphs' as const
} as const;

export const ArtifactType = {
  MERMAID_DIAGRAM: 'mermaid_diagram' as const,
  OPENAPI_SPEC: 'openapi_spec' as const,
  IMPLEMENTATION_GUIDE: 'implementation_guide' as const,
  TEST_STRATEGY: 'test_strategy' as const,
  CODE_SNIPPET: 'code_snippet' as const
} as const;

// Type guard functions
export function isDelegationType(value: any): value is DelegationType {
  return typeof value === 'string' && Object.values(DelegationType).includes(value as DelegationType);
}

export function isComplexityLevel(value: any): value is ComplexityLevel {
  return typeof value === 'string' && Object.values(ComplexityLevel).includes(value as ComplexityLevel);
}

export function isContextType(value: any): value is ContextType {
  return typeof value === 'string' && Object.values(ContextType).includes(value as ContextType);
}

export function isArtifactType(value: any): value is ArtifactType {
  return typeof value === 'string' && Object.values(ArtifactType).includes(value as ArtifactType);
}

export interface ProblemClassification {
  delegationType: DelegationType;
  suitabilityScore: number;
  complexity: ComplexityLevel;
  requiredContextTypes: ContextType[];
  expectedArtifactTypes: ArtifactType[];
  estimatedRounds: number;
  confidence: number;
}

export interface DelegationRecommendation {
  suitable: boolean;
  shouldDelegate: boolean;
  classification: ProblemClassification;
  recommendation: string;
  reasoning: string;
  contextEstimate: {
    estimatedSize: number;
    extractionComplexity: ComplexityLevel;
  };
}

export interface ClassificationResult {
  shouldDelegate: boolean;
  classification: ProblemClassification;
  recommendation: string;
}

export interface ExtractedFile {
  path: string;
  content: string;
  type: 'code' | 'doc' | 'config';
  relevance: number;
  size: number;
  lastModified?: Date;
}

export interface ContextBundle {
  title: string;
  markdown: string;
  files: ExtractedFile[];
  totalSize: number;
  compressionApplied: boolean;
}

export interface ParsedArtifact {
  id: string;
  type: ArtifactType;
  content: string;
  metadata: {
    sourceLocation: { start: number; end: number };
    parseWarnings: string[];
    extractedAt: Date;
    usage?: {
      status: 'not_implemented' | 'in_progress' | 'implemented' | 'modified';
      updatedAt: Date;
    };
  };
  structured?: {
    mermaid?: string;
    openapi?: Record<string, unknown>;
    implementationSteps?: ImplementationStep[];
  };
}

export interface ImplementationStep {
  id: string;
  phase: string;
  title: string;
  description: string;
  action: string;
  file?: string;
  dependencies: string[];
  complexity: ComplexityLevel;
  estimate?: string;
}

export interface ValidationIssue {
  severity: 'error' | 'warning' | 'info';
  message: string;
  location?: string;
  suggestion?: string;
}

export interface ValidationResult {
  artifactId: string;
  artifactType: ArtifactType;
  isValid: boolean;
  completenessScore: number;
  issues: ValidationIssue[];
  suggestions: string[];
}

export interface DelegationRound {
  roundNumber: number;
  request: string;
  response: string;
  artifacts: ParsedArtifact[];
  validation: ExtendedValidationResult[];
  timestamp: Date;
  contextSize: number;
}

export interface DelegationSession {
  id: string;
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
  status: 'active' | 'completed' | 'abandoned';
  problem: {
    title: string;
    description: string;
    type: DelegationType;
    complexity: ComplexityLevel;
  };
  rounds: DelegationRound[];
  finalArtifacts: ParsedArtifact[];
  implementationGuide?: ImplementationGuide;
  metrics: {
    totalTime: number;
    contextSize: number;
    roundCount: number;
    finalCompleteness: number;
  };
}

export interface ImplementationGuide {
  title: string;
  overview: string;
  projectName: string;
  prerequisites: string[];
  phases: Array<{
    name: string;
    description: string;
    steps: ImplementationStep[];
  }>;
  steps: ImplementationStep[];
  codeMappings: Record<string, string>;
  dependencies: string[];
  testImplementation?: string;
  riskRegister?: Array<{
    risk: string;
    mitigation: string;
    owner: string;
  }>;
}

export interface Template {
  template_id: string;
  name: string;
  category: DelegationType;
  version: string;
  parameters: Array<{
    name: string;
    required: boolean;
    type?: string;
    default?: unknown;
  }>;
  context_requirements: string[];
  expected_artifacts: Array<{
    type: string;
    subtype?: string;
    format?: string;
  }>;
  prompt_template: string;
}

export interface ExtractionOptions {
  maxFiles?: number;
  maxFileSize?: number;
  includeTests?: boolean;
  depth?: number;
}

export interface PackagingOptions {
  maxSize?: number;
  compressionEnabled?: boolean;
  summarizationEnabled?: boolean;
}

export interface SearchCriteria {
  problemType?: DelegationType;
  keywords?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  status?: 'active' | 'completed' | 'abandoned';
}

export interface VersionedArtifact extends ParsedArtifact {
  version: number;
  sessionId: string;
  roundNumber: number;
  previousVersion?: number;
}

export interface ArtifactDiff {
  artifactId: string;
  fromVersion: number;
  toVersion: number;
  additions: string[];
  deletions: string[];
  modifications: string[];
  summary: string;
}

// Additional missing types
export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}

export interface MermaidAST {
  nodes: Array<{
    id: string;
    label?: string;
    type?: string;
  }>;
  edges: Array<{
    from: string;
    to: string;
    label?: string;
    type?: string;
  }>;
}

export interface OpenAPISpec {
  openapi: string;
  info: {
    title: string;
    version: string;
    description?: string;
  };
  paths: Record<string, any>;
  components?: {
    schemas?: Record<string, any>;
  };
}

export interface Step {
  id: string;
  title: string;
  description: string;
  action: string;
  dependencies: string[];
  complexity: ComplexityLevel;
}

export type SessionComplexity = 'simple' | 'moderate' | 'complex';

// Update ValidationResult to include missing properties
export interface QualityScores {
  completeness: number;
  clarity: number;
  implementability: number;
}

// Extend ValidationResult with missing properties
export interface ExtendedValidationResult extends ValidationResult {
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  qualityScores: QualityScores;
  followUpQuestions: string[];
}

// Update DelegationRequest interface
export interface DelegationRequest {
  sessionId: string;
  problemDescription: string;
  contextBundle: ContextBundle;
  expectedArtifacts: ArtifactType[];
  template: Template;
  roundNumber: number;
  previousArtifacts?: ParsedArtifact[];
}
