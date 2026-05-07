/**
 * Zod schemas for runtime validation of core types
 */

import { z } from 'zod';
import { DelegationType, ComplexityLevel, ContextType, ArtifactType } from './core';

/**
 * Schema for delegation types
 */
export const DelegationTypeSchema = z.nativeEnum(DelegationType);

/**
 * Schema for complexity levels
 */
export const ComplexityLevelSchema = z.nativeEnum(ComplexityLevel);

/**
 * Schema for context types
 */
export const ContextTypeSchema = z.nativeEnum(ContextType);

/**
 * Schema for artifact types
 */
export const ArtifactTypeSchema = z.nativeEnum(ArtifactType);

/**
 * Schema for problem classification
 */
export const ProblemClassificationSchema = z.object({
  suitable: z.boolean(),
  delegationType: DelegationTypeSchema,
  alternativeTypes: z.array(DelegationTypeSchema).optional(),
  complexity: ComplexityLevelSchema,
  requiredContextTypes: z.array(ContextTypeSchema),
  expectedArtifacts: z.array(ArtifactTypeSchema),
  suitabilityScore: z.number().min(0).max(100),
  reasoning: z.string().min(1),
});

/**
 * Schema for code snippets
 */
export const CodeSnippetSchema = z.object({
  filePath: z.string().min(1),
  startLine: z.number().positive(),
  endLine: z.number().positive(),
  content: z.string(),
  language: z.string().min(1),
  relevance: z.number().min(0).max(1),
});

/**
 * Schema for documentation excerpts
 */
export const DocumentationExcerptSchema = z.object({
  sourcePath: z.string().min(1),
  title: z.string().min(1),
  content: z.string(),
  relevance: z.number().min(0).max(1),
});

/**
 * Schema for context manifest entries
 */
export const ContextManifestEntrySchema = z.object({
  source: z.string().min(1),
  type: z.enum(['code', 'documentation', 'config']),
  size: z.number().nonnegative(),
  relevance: z.enum(['high', 'medium', 'low']),
});

/**
 * Schema for context bundles
 */
export const ContextBundleSchema = z.object({
  id: z.string().min(1),
  problemDescription: z.string().min(1),
  codeSnippets: z.array(CodeSnippetSchema),
  documentation: z.array(DocumentationExcerptSchema),
  constraints: z.array(z.string()),
  totalSize: z.number().nonnegative(),
  truncated: z.boolean(),
  excludedSummary: z.string().optional(),
  manifest: z.array(ContextManifestEntrySchema),
});

/**
 * Schema for template parameters
 */
export const TemplateParameterSchema = z.object({
  name: z.string().min(1),
  required: z.boolean(),
  type: z.enum(['string', 'number', 'boolean', 'list']),
  default: z.union([z.string(), z.number(), z.boolean(), z.array(z.string())]).optional(),
  description: z.string().optional(),
});

/**
 * Schema for delegation templates
 */
export const DelegationTemplateSchema = z.object({
  templateId: z.string().min(1),
  name: z.string().min(1),
  category: DelegationTypeSchema,
  version: z.string().min(1),
  parameters: z.array(TemplateParameterSchema),
  contextRequirements: z.array(ContextTypeSchema),
  expectedArtifacts: z.array(
    z.object({
      type: ArtifactTypeSchema,
      subtype: z.string().optional(),
      format: z.string().optional(),
      granularity: z.string().optional(),
    })
  ),
  promptTemplate: z.string().min(1),
  usageCount: z.number().nonnegative().optional(),
});

/**
 * Schema for validation results
 */
export const ValidationResultSchema = z.object({
  valid: z.boolean(),
  completenessScore: z.number().min(0).max(100),
  qualityScores: z.object({
    completeness: z.number().min(0).max(100),
    clarity: z.number().min(0).max(100),
    implementability: z.number().min(0).max(100),
  }),
  missingElements: z.array(z.string()),
  errors: z.array(z.string()),
  warnings: z.array(z.string()),
  followUpQuestions: z.array(z.string()).optional(),
});

/**
 * Schema for Mermaid diagrams
 */
export const MermaidDiagramSchema = z.object({
  diagramType: z.string().min(1),
  syntax: z.string().min(1),
  valid: z.boolean(),
  errors: z.array(z.string()).optional(),
});

/**
 * Schema for OpenAPI specifications
 */
export const OpenAPISpecSchema = z.object({
  version: z.string().min(1),
  spec: z.record(z.unknown()),
  valid: z.boolean(),
  errors: z.array(z.string()).optional(),
});

/**
 * Schema for implementation steps
 */
export const ImplementationStepSchema = z.object({
  id: z.string().min(1),
  number: z.string().min(1),
  action: z.string().min(1),
  filePath: z.string().optional(),
  dependencies: z.array(z.string()),
  complexity: ComplexityLevelSchema,
  expectedOutcome: z.string().optional(),
  codeTemplate: z.string().optional(),
  artifactReferences: z.array(z.string()).optional(),
});

/**
 * Schema for implementation guides
 */
export const ImplementationGuideSchema = z.object({
  title: z.string().min(1),
  prerequisites: z.array(z.string()),
  phases: z.array(
    z.object({
      name: z.string().min(1),
      complexity: ComplexityLevelSchema,
      steps: z.array(ImplementationStepSchema),
    })
  ),
  risks: z
    .array(
      z.object({
        risk: z.string().min(1),
        mitigation: z.string().min(1),
        owner: z.string().optional(),
      })
    )
    .optional(),
});

/**
 * Schema for test strategies
 */
export const TestStrategySchema = z.object({
  coverageTargets: z.object({
    unit: z.number().min(0).max(100).optional(),
    integration: z.number().min(0).max(100).optional(),
    e2e: z.number().min(0).max(100).optional(),
  }),
  propertyTests: z.array(
    z.object({
      name: z.string().min(1),
      property: z.string().min(1),
      generator: z.string().optional(),
    })
  ),
  edgeCases: z.array(z.string()),
  testDataRequirements: z.array(z.string()),
});

/**
 * Schema for parsed artifacts
 */
export const ParsedArtifactSchema = z.object({
  id: z.string().min(1),
  type: ArtifactTypeSchema,
  content: z.string(),
  metadata: z.object({
    sourceLocation: z.object({
      start: z.number().nonnegative(),
      end: z.number().nonnegative(),
    }),
    parseWarnings: z.array(z.string()),
    extractedAt: z.date(),
  }),
  structured: z
    .union([MermaidDiagramSchema, OpenAPISpecSchema, ImplementationGuideSchema, TestStrategySchema])
    .optional(),
});

/**
 * Schema for delegation requests
 */
export const DelegationRequestSchema = z.object({
  id: z.string().min(1),
  sessionId: z.string().min(1),
  roundNumber: z.number().positive(),
  title: z.string().min(1),
  description: z.string().min(1),
  objectives: z.array(z.string()),
  constraints: z.array(z.string()),
  expectedArtifacts: z.array(ArtifactTypeSchema),
  formatRequirements: z.string(),
  contextBundle: ContextBundleSchema,
  formattedText: z.string().min(1),
  createdAt: z.date(),
});

/**
 * Schema for delegation rounds
 */
export const DelegationRoundSchema = z.object({
  roundNumber: z.number().positive(),
  request: DelegationRequestSchema,
  response: z.string().optional(),
  artifacts: z.array(ParsedArtifactSchema),
  validation: ValidationResultSchema.optional(),
  timestamp: z.date(),
});

/**
 * Schema for session metrics
 */
export const SessionMetricsSchema = z.object({
  totalTime: z.number().nonnegative(),
  contextSize: z.number().nonnegative(),
  roundCount: z.number().nonnegative(),
  finalCompleteness: z.number().min(0).max(100),
  estimatedCost: z.number().nonnegative().optional(),
});

/**
 * Schema for delegation sessions
 */
export const DelegationSessionSchema = z.object({
  id: z.string().min(1),
  createdAt: z.date(),
  updatedAt: z.date(),
  problem: z.object({
    title: z.string().min(1),
    description: z.string().min(1),
    type: DelegationTypeSchema,
    complexity: ComplexityLevelSchema,
  }),
  rounds: z.array(DelegationRoundSchema),
  finalArtifacts: z.array(ParsedArtifactSchema),
  implementationGuide: ImplementationGuideSchema.optional(),
  metrics: SessionMetricsSchema,
  status: z.enum(['active', 'completed', 'abandoned']),
});

/**
 * Type guards using Zod schemas
 */
export const isProblemClassification = (
  obj: unknown
): obj is z.infer<typeof ProblemClassificationSchema> => {
  return ProblemClassificationSchema.safeParse(obj).success;
};

export const isContextBundle = (obj: unknown): obj is z.infer<typeof ContextBundleSchema> => {
  return ContextBundleSchema.safeParse(obj).success;
};

export const isDelegationTemplate = (
  obj: unknown
): obj is z.infer<typeof DelegationTemplateSchema> => {
  return DelegationTemplateSchema.safeParse(obj).success;
};

export const isValidationResult = (obj: unknown): obj is z.infer<typeof ValidationResultSchema> => {
  return ValidationResultSchema.safeParse(obj).success;
};

export const isDelegationSession = (
  obj: unknown
): obj is z.infer<typeof DelegationSessionSchema> => {
  return DelegationSessionSchema.safeParse(obj).success;
};

export const isParsedArtifact = (obj: unknown): obj is z.infer<typeof ParsedArtifactSchema> => {
  return ParsedArtifactSchema.safeParse(obj).success;
};

export const isDelegationRequest = (
  obj: unknown
): obj is z.infer<typeof DelegationRequestSchema> => {
  return DelegationRequestSchema.safeParse(obj).success;
};

export const isDelegationRound = (obj: unknown): obj is z.infer<typeof DelegationRoundSchema> => {
  return DelegationRoundSchema.safeParse(obj).success;
};

export const isImplementationGuide = (
  obj: unknown
): obj is z.infer<typeof ImplementationGuideSchema> => {
  return ImplementationGuideSchema.safeParse(obj).success;
};

export const isMermaidDiagram = (obj: unknown): obj is z.infer<typeof MermaidDiagramSchema> => {
  return MermaidDiagramSchema.safeParse(obj).success;
};

export const isOpenAPISpec = (obj: unknown): obj is z.infer<typeof OpenAPISpecSchema> => {
  return OpenAPISpecSchema.safeParse(obj).success;
};

export const isTestStrategy = (obj: unknown): obj is z.infer<typeof TestStrategySchema> => {
  return TestStrategySchema.safeParse(obj).success;
};
