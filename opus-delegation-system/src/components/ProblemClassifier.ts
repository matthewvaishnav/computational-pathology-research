/**
 * Problem Classifier Component
 * Implements Task 3.1, 3.2 - Problem classification engine and context recommendation logic
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
 */

import {
  DelegationType,
  ComplexityLevel,
  ContextType,
  ArtifactType,
  ProblemClassification,
  DelegationRecommendation
} from '../types/core.js';

// Complexity indicators for classification (Requirement 1.1)
interface ComplexityIndicator {
  name: string;
  weight: number;
  keywords: string[];
  description: string;
}

// Problem type patterns for delegation categorization (Requirement 1.2)
interface DelegationTypePattern {
  type: DelegationType;
  keywords: string[];
  requiredContextTypes: ContextType[];
  expectedArtifacts: ArtifactType[];
  complexityMultiplier: number;
}

export class ProblemClassifier {
  private complexityIndicators: ComplexityIndicator[] = [
    {
      name: 'architectural_scope',
      weight: 0.3,
      keywords: [
        'architecture', 'system design', 'component', 'microservice', 'distributed',
        'scalability', 'performance', 'multi-tier', 'service mesh', 'event-driven',
        'data flow', 'system boundaries', 'integration patterns'
      ],
      description: 'Multi-component design, system boundaries'
    },
    {
      name: 'formal_reasoning',
      weight: 0.3,
      keywords: [
        'verification', 'proof', 'invariant', 'correctness', 'formal method',
        'property-based', 'theorem', 'specification', 'contract', 'assertion',
        'safety', 'liveness', 'consistency'
      ],
      description: 'Verification, correctness proofs, invariants'
    },
    {
      name: 'novel_patterns',
      weight: 0.2,
      keywords: [
        'novel', 'research', 'experimental', 'cutting-edge', 'innovative',
        'machine learning', 'AI', 'blockchain', 'quantum', 'federated',
        'zero-knowledge', 'differential privacy', 'homomorphic'
      ],
      description: 'Unfamiliar domain, research-adjacent'
    },
    {
      name: 'integration_complexity',
      weight: 0.15,
      keywords: [
        'integration', 'external system', 'API', 'protocol', 'DICOM', 'HL7',
        'REST', 'GraphQL', 'gRPC', 'message queue', 'event bus', 'webhook',
        'third-party', 'legacy system'
      ],
      description: 'Multiple external systems, protocols'
    },
    {
      name: 'strategic_decisions',
      weight: 0.05,
      keywords: [
        'technology selection', 'trade-off', 'decision', 'comparison',
        'evaluation', 'framework choice', 'database selection', 'cloud provider',
        'architecture pattern', 'design pattern'
      ],
      description: 'Technology selection, trade-off analysis'
    }
  ];

  private delegationTypePatterns: DelegationTypePattern[] = [
    {
      type: DelegationType.ARCHITECTURE_DESIGN,
      keywords: [
        'architecture', 'system design', 'component design', 'service design',
        'data flow', 'system structure', 'component relationships', 'scalability',
        'performance', 'distributed system', 'microservices'
      ],
      requiredContextTypes: [
        ContextType.ARCHITECTURE_DOCS,
        ContextType.EXISTING_DESIGNS,
        ContextType.CONSTRAINTS,
        ContextType.REQUIREMENTS_DOCS
      ],
      expectedArtifacts: [
        ArtifactType.MERMAID_DIAGRAM,
        ArtifactType.IMPLEMENTATION_GUIDE
      ],
      complexityMultiplier: 1.2
    },
    {
      type: DelegationType.API_DESIGN,
      keywords: [
        'API', 'endpoint', 'REST', 'GraphQL', 'gRPC', 'schema', 'data model',
        'request', 'response', 'authentication', 'authorization', 'versioning',
        'OpenAPI', 'swagger'
      ],
      requiredContextTypes: [
        ContextType.API_ENDPOINTS,
        ContextType.CODE_SNIPPETS,
        ContextType.EXISTING_DESIGNS,
        ContextType.CONSTRAINTS
      ],
      expectedArtifacts: [
        ArtifactType.OPENAPI_SPEC,
        ArtifactType.IMPLEMENTATION_GUIDE
      ],
      complexityMultiplier: 1.0
    },
    {
      type: DelegationType.TEST_STRATEGY,
      keywords: [
        'test', 'testing', 'property-based', 'unit test', 'integration test',
        'coverage', 'test case', 'test suite', 'verification', 'validation',
        'edge case', 'invariant', 'generator'
      ],
      requiredContextTypes: [
        ContextType.TEST_FILES,
        ContextType.CODE_SNIPPETS,
        ContextType.REQUIREMENTS_DOCS
      ],
      expectedArtifacts: [
        ArtifactType.TEST_STRATEGY,
        ArtifactType.IMPLEMENTATION_GUIDE
      ],
      complexityMultiplier: 0.9
    },
    {
      type: DelegationType.INTEGRATION_DESIGN,
      keywords: [
        'integration', 'external system', 'protocol', 'DICOM', 'HL7', 'FHIR',
        'message queue', 'event bus', 'webhook', 'adapter', 'connector',
        'third-party', 'legacy system'
      ],
      requiredContextTypes: [
        ContextType.EXTERNAL_INTERFACES,
        ContextType.ARCHITECTURE_DOCS,
        ContextType.CONSTRAINTS,
        ContextType.CODE_SNIPPETS
      ],
      expectedArtifacts: [
        ArtifactType.MERMAID_DIAGRAM,
        ArtifactType.OPENAPI_SPEC,
        ArtifactType.IMPLEMENTATION_GUIDE
      ],
      complexityMultiplier: 1.1
    },
    {
      type: DelegationType.REFACTORING_ANALYSIS,
      keywords: [
        'refactor', 'refactoring', 'code smell', 'technical debt', 'cleanup',
        'restructure', 'dependency', 'coupling', 'cohesion', 'maintainability',
        'code quality', 'design pattern'
      ],
      requiredContextTypes: [
        ContextType.CODE_SNIPPETS,
        ContextType.DEPENDENCY_GRAPHS,
        ContextType.ARCHITECTURE_DOCS
      ],
      expectedArtifacts: [
        ArtifactType.IMPLEMENTATION_GUIDE,
        ArtifactType.MERMAID_DIAGRAM
      ],
      complexityMultiplier: 0.8
    },
    {
      type: DelegationType.FORMAL_VERIFICATION,
      keywords: [
        'verification', 'proof', 'formal method', 'invariant', 'correctness',
        'specification', 'contract', 'assertion', 'theorem', 'property',
        'safety', 'liveness', 'consistency'
      ],
      requiredContextTypes: [
        ContextType.CODE_SNIPPETS,
        ContextType.REQUIREMENTS_DOCS,
        ContextType.CONSTRAINTS
      ],
      expectedArtifacts: [
        ArtifactType.TEST_STRATEGY,
        ArtifactType.IMPLEMENTATION_GUIDE
      ],
      complexityMultiplier: 1.5
    }
  ];

  /**
   * Classifies a problem for Opus delegation suitability
   * Implements Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
   */
  public classifyProblem(
    problemDescription: string,
    problemTitle?: string
  ): DelegationRecommendation {
    const fullText = `${problemTitle || ''} ${problemDescription}`.toLowerCase();
    
    // Calculate complexity indicators (Requirement 1.1)
    const complexityScore = this.calculateComplexityScore(fullText);
    
    // Determine delegation type (Requirement 1.2)
    const delegationType = this.determineDelegationType(fullText);
    
    // Get pattern for the delegation type
    const pattern = this.delegationTypePatterns.find(p => p.type === delegationType);
    if (!pattern) {
      throw new Error(`No pattern found for delegation type: ${delegationType}`);
    }
    
    // Calculate suitability score
    const baseSuitability = Math.min(100, complexityScore * 100);
    const suitabilityScore = Math.min(100, baseSuitability * pattern.complexityMultiplier);
    
    // Determine complexity level (Requirement 1.4)
    const complexity = this.determineComplexityLevel(complexityScore, pattern.complexityMultiplier);
    
    // Estimate rounds based on complexity and type
    const estimatedRounds = this.estimateRounds(complexity, delegationType);
    
    // Calculate confidence based on keyword matches
    const confidence = this.calculateConfidence(fullText, pattern);
    
    // Create classification (Requirements 1.3, 1.5)
    const classification: ProblemClassification = {
      delegationType,
      suitabilityScore,
      complexity,
      requiredContextTypes: pattern.requiredContextTypes,
      expectedArtifactTypes: pattern.expectedArtifacts,
      estimatedRounds,
      confidence
    };
    
    // Determine if suitable for delegation (minimum threshold: 60)
    const suitable = suitabilityScore >= 60 && confidence >= 30; // Lower confidence threshold
    
    // Generate reasoning
    const reasoning = this.generateReasoning(
      complexityScore,
      delegationType,
      suitabilityScore,
      confidence,
      suitable
    );
    
    // Estimate context requirements
    const contextEstimate = this.estimateContextRequirements(
      complexity,
      pattern.requiredContextTypes.length
    );
    
    return {
      suitable,
      classification,
      reasoning,
      contextEstimate
    };
  }

  /**
   * Calculate complexity score based on indicators (Requirement 1.1)
   */
  private calculateComplexityScore(text: string): number {
    let totalScore = 0;
    let totalWeight = 0;
    
    for (const indicator of this.complexityIndicators) {
      const matches = indicator.keywords.filter(keyword => 
        text.includes(keyword.toLowerCase())
      ).length;
      
      // Score based on keyword density with better scaling
      const keywordDensity = matches / indicator.keywords.length;
      // Use exponential scaling to reward multiple matches, more generous scoring
      const indicatorScore = Math.min(1, keywordDensity * 4 + (matches > 0 ? 0.3 : 0));
      
      totalScore += indicatorScore * indicator.weight;
      totalWeight += indicator.weight;
    }
    
    return totalWeight > 0 ? totalScore / totalWeight : 0;
  }

  /**
   * Determine delegation type based on keyword matching (Requirement 1.2)
   */
  private determineDelegationType(text: string): DelegationType {
    let bestMatch = this.delegationTypePatterns[0];
    let bestScore = 0;
    
    for (const pattern of this.delegationTypePatterns) {
      const matches = pattern.keywords.filter(keyword =>
        text.includes(keyword.toLowerCase())
      ).length;
      
      const score = matches / pattern.keywords.length;
      
      if (score > bestScore) {
        bestScore = score;
        bestMatch = pattern;
      }
    }
    
    return bestMatch.type;
  }

  /**
   * Determine complexity level (Requirement 1.4)
   */
  private determineComplexityLevel(
    complexityScore: number,
    multiplier: number
  ): ComplexityLevel {
    const adjustedScore = complexityScore * multiplier;
    
    if (adjustedScore >= 0.7) {
      return ComplexityLevel.COMPLEX;
    } else if (adjustedScore >= 0.4) {
      return ComplexityLevel.MODERATE;
    } else {
      return ComplexityLevel.SIMPLE;
    }
  }

  /**
   * Estimate number of rounds needed
   */
  private estimateRounds(complexity: ComplexityLevel, type: DelegationType): number {
    const baseRounds = {
      [ComplexityLevel.SIMPLE]: 1,
      [ComplexityLevel.MODERATE]: 2,
      [ComplexityLevel.COMPLEX]: 3
    };
    
    const typeMultiplier = {
      [DelegationType.ARCHITECTURE_DESIGN]: 1.2,
      [DelegationType.API_DESIGN]: 1.0,
      [DelegationType.TEST_STRATEGY]: 0.9,
      [DelegationType.INTEGRATION_DESIGN]: 1.1,
      [DelegationType.REFACTORING_ANALYSIS]: 0.8,
      [DelegationType.FORMAL_VERIFICATION]: 1.5
    };
    
    return Math.ceil(baseRounds[complexity] * typeMultiplier[type]);
  }

  /**
   * Calculate confidence based on keyword matches
   */
  private calculateConfidence(text: string, pattern: DelegationTypePattern): number {
    const matches = pattern.keywords.filter(keyword =>
      text.includes(keyword.toLowerCase())
    ).length;
    
    const keywordCoverage = matches / pattern.keywords.length;
    
    // Base confidence on keyword coverage and text length
    const textLength = text.length;
    const lengthFactor = Math.min(1, textLength / 100); // Normalize around 100 chars
    
    // Boost confidence for multiple keyword matches
    const matchBonus = matches > 1 ? Math.min(20, matches * 5) : 0;
    
    return Math.min(100, (keywordCoverage * 60 + lengthFactor * 20 + matchBonus));
  }

  /**
   * Generate human-readable reasoning for the classification
   */
  private generateReasoning(
    complexityScore: number,
    delegationType: DelegationType,
    suitabilityScore: number,
    confidence: number,
    suitable: boolean
  ): string {
    const complexityLevel = complexityScore >= 0.7 ? 'high' : 
                           complexityScore >= 0.4 ? 'moderate' : 'low';
    
    let reasoning = `Problem classified as ${delegationType.replace('_', ' ')} `;
    reasoning += `with ${complexityLevel} complexity (score: ${Math.round(complexityScore * 100)}%). `;
    reasoning += `Suitability score: ${Math.round(suitabilityScore)}%, `;
    reasoning += `confidence: ${Math.round(confidence)}%. `;
    
    if (suitable) {
      reasoning += 'Recommended for Opus delegation due to sufficient complexity and clear problem type match.';
    } else {
      if (confidence < 30) {
        reasoning += 'Not recommended: problem description is ambiguous or doesn\'t clearly match delegation patterns.';
      } else if (suitabilityScore < 60) {
        reasoning += 'Not recommended: problem may be too simple or not well-suited for architectural delegation.';
      }
    }
    
    return reasoning;
  }

  /**
   * Estimate context requirements (Requirement 1.3, 1.4)
   */
  private estimateContextRequirements(
    complexity: ComplexityLevel,
    contextTypeCount: number
  ): { estimatedSize: number; extractionComplexity: ComplexityLevel } {
    const baseSizes = {
      [ComplexityLevel.SIMPLE]: 15000,
      [ComplexityLevel.MODERATE]: 35000,
      [ComplexityLevel.COMPLEX]: 50000
    };
    
    const estimatedSize = Math.min(50000, baseSizes[complexity] * (1 + contextTypeCount * 0.1));
    
    return {
      estimatedSize,
      extractionComplexity: complexity
    };
  }

  /**
   * Get all supported delegation types
   */
  public getSupportedDelegationTypes(): DelegationType[] {
    return this.delegationTypePatterns.map(p => p.type);
  }

  /**
   * Get required context types for a delegation type
   */
  public getRequiredContextTypes(delegationType: DelegationType): ContextType[] {
    const pattern = this.delegationTypePatterns.find(p => p.type === delegationType);
    return pattern ? pattern.requiredContextTypes : [];
  }

  /**
   * Get expected artifact types for a delegation type
   */
  public getExpectedArtifactTypes(delegationType: DelegationType): ArtifactType[] {
    const pattern = this.delegationTypePatterns.find(p => p.type === delegationType);
    return pattern ? pattern.expectedArtifacts : [];
  }
}