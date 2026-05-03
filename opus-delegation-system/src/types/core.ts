// Core type definitions for Opus Delegation System

export type DelegationType =
  | 'architecture_design'
  | 'api_design'
  | 'test_strategy'
  | 'integration_design'
  | 'refactoring_analysis'
  | 'formal_verification';

export type ComplexityLevel = 'simple' | 'moderate' | 'complex';

export type ArtifactType =
  | 'mermaid_diagram'
  | 'openapi_specification'
  | 'implementation_plan'
  | 'test_strategy'
  | 'code_snippet';

export interface ProblemClassification {
  delegationType: DelegationType;
  complexity: ComplexityLevel;
  requiredContext: string[];
  expectedArtifacts: ArtifactType[];
  confidence: number;
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
  validation: ValidationResult[];
  timestamp: Date;
  contextSize: number;
}

export interface DelegationSession {
  id: string;
  createdAt: Date;
  updatedAt: Date;
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
  projectName: string;
  prerequisites: string[];
  steps: ImplementationStep[];
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
