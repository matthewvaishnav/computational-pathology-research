/**
 * Core type definitions for the Opus Delegation System
 * Implements Task 2.1, 2.2, 2.3 - Core data structures
 */

// Delegation Types (Requirement 1.2)
export enum DelegationType {
  ARCHITECTURE_DESIGN = 'architecture_design',
  API_DESIGN = 'api_design', 
  TEST_STRATEGY = 'test_strategy',
  INTEGRATION_DESIGN = 'integration_design',
  REFACTORING_ANALYSIS = 'refactoring_analysis',
  FORMAL_VERIFICATION = 'formal_verification'
}

// Complexity Levels (Requirement 1.4)
export enum ComplexityLevel {
  SIMPLE = 'simple',
  MODERATE = 'moderate', 
  COMPLEX = 'complex'
}

// Context Types (Requirement 1.3)
export enum ContextType {
  CODE_SNIPPETS = 'code_snippets',
  REQUIREMENTS_DOCS = 'requirements_docs',
  EXISTING_DESIGNS = 'existing_designs',
  CONSTRAINTS = 'constraints',
  ARCHITECTURE_DOCS = 'architecture_docs',
  API_ENDPOINTS = 'api_endpoints',
  TEST_FILES = 'test_files',
  EXTERNAL_INTERFACES = 'external_interfaces',
  DEPENDENCY_GRAPHS = 'dependency_graphs'
}

// Artifact Types (Requirement 5.1-5.5)
export enum ArtifactType {
  MERMAID_DIAGRAM = 'mermaid_diagram',
  OPENAPI_SPEC = 'openapi_spec',
  IMPLEMENTATION_GUIDE = 'implementation_guide',
  TEST_STRATEGY = 'test_strategy',
  CODE_SNIPPET = 'code_snippet'
}

// Problem Classification (Requirements 1.1-1.6)
export interface ProblemClassification {
  delegationType: DelegationType;
  suitabilityScore: number; // 0-100
  complexity: ComplexityLevel;
  requiredContextTypes: ContextType[];
  expectedArtifactTypes: ArtifactType[];
  estimatedRounds: number;
  confidence: number; // 0-100
}

// Delegation Recommendation (Requirement 1.5)
export interface DelegationRecommendation {
  suitable: boolean;
  classification: ProblemClassification;
  reasoning: string;
  contextEstimate: {
    estimatedSize: number; // characters
    extractionComplexity: ComplexityLevel;
  };
}

// Parsed Artifact Structure (Requirement 5.7)
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
    mermaid?: MermaidAST;
    openapi?: OpenAPISpec;
    implementationSteps?: Step[];
  };
}

// Mermaid AST placeholder
export interface MermaidAST {
  type: string;
  nodes: Array<{ id: string; label: string; type: string }>;
  edges: Array<{ from: string; to: string; label?: string }>;
}

// OpenAPI Spec placeholder  
export interface OpenAPISpec {
  openapi: string;
  info: { 
    title: string; 
    version: string;
    description?: string;
  };
  paths: Record<string, any>;
  components?: Record<string, any>;
  security?: any[];
}

// Implementation Step
export interface Step {
  id: string;
  action: string;
  description: string;
  dependencies: string[];
  complexity: ComplexityLevel;
  estimatedTime?: number;
}

// Session Management (Requirements 8.1, 8.2, 9.1)
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
  
  rounds: Array<{
    roundNumber: number;
    request: string;
    response: string;
    artifacts: ParsedArtifact[];
    validation: ValidationResult;
    timestamp: Date;
  }>;
  
  finalArtifacts: ParsedArtifact[];
  implementationGuide?: ImplementationGuide;
  
  metrics: {
    totalTime: number;
    contextSize: number;
    roundCount: number;
    finalCompleteness: number;
  };
}

export interface DelegationRound {
  roundNumber: number;
  request: string;
  response: string;
  artifacts: ParsedArtifact[];
  validation: ValidationResult;
  timestamp: Date;
}

export interface SessionMetrics {
  totalTime: number;
  contextSize: number;
  roundCount: number;
  finalCompleteness: number;
}

// Validation Results (Requirement 6.6, 6.7)
export interface ValidationResult {
  completenessScore: number; // 0-100
  qualityScores: {
    completeness: number;
    clarity: number;
    implementability: number;
  };
  isValid: boolean;
  errors: ValidationError[];
  warnings: string[];
  followUpQuestions: string[];
}

export interface ValidationError {
  type: string;
  message: string;
  location?: string;
  severity: 'error' | 'warning';
}

// Implementation Guide (Requirement 7.1-7.7)
export interface ImplementationGuide {
  id: string;
  title: string;
  phases: ImplementationPhase[];
  prerequisites: string[];
  riskRegister: Risk[];
  generatedAt: Date;
}

export interface ImplementationPhase {
  name: string;
  complexity: ComplexityLevel;
  steps: ImplementationStep[];
}

export interface ImplementationStep {
  id: string;
  action: string;
  description: string;
  filePath?: string;
  dependencies: string[];
  complexity: ComplexityLevel;
  estimatedTime?: number;
  codeTemplate?: string;
  artifactReferences: string[];
}

export interface Risk {
  description: string;
  mitigation: string;
  owner?: string;
}

// Type Guards
export function isDelegationType(value: string): value is DelegationType {
  return Object.values(DelegationType).includes(value as DelegationType);
}

export function isComplexityLevel(value: string): value is ComplexityLevel {
  return Object.values(ComplexityLevel).includes(value as ComplexityLevel);
}

export function isContextType(value: string): value is ContextType {
  return Object.values(ContextType).includes(value as ContextType);
}

export function isArtifactType(value: string): value is ArtifactType {
  return Object.values(ArtifactType).includes(value as ArtifactType);
}