/**
 * Problem Classifier Component
 * 
 * Determines whether a problem is suitable for Opus delegation and categorizes it by type.
 * Implements complexity indicator detection and context recommendation logic.
 */

import {
  DelegationType,
  ComplexityLevel,
  ContextType,
  ArtifactType,
  ProblemClassification,
  ExtractionStrategy,
} from '../types';

/**
 * Complexity indicators with their weights for delegation suitability scoring
 */
interface ComplexityIndicator {
  name: string;
  weight: number;
  keywords: string[];
  description: string;
}

/**
 * Problem classification configuration
 */
interface ClassificationConfig {
  /** Minimum suitability score for delegation (0-100) */
  minSuitabilityScore: number;
  /** Complexity indicators with weights */
  complexityIndicators: ComplexityIndicator[];
  /** Delegation type mappings */
  delegationTypeMappings: Record<string, DelegationType>;
}

/**
 * Default classification configuration
 */
const DEFAULT_CONFIG: ClassificationConfig = {
  minSuitabilityScore: 60,
  complexityIndicators: [
    {
      name: 'architectural_scope',
      weight: 0.3,
      keywords: [
        'architecture', 'system design', 'component', 'microservice', 'distributed',
        'scalability', 'performance', 'infrastructure', 'deployment', 'integration',
        'multi-tier', 'service mesh', 'api gateway', 'load balancer', 'database design'
      ],
      description: 'Multi-component design, system boundaries'
    },
    {
      name: 'formal_reasoning',
      weight: 0.3,
      keywords: [
        'verification', 'correctness', 'proof', 'invariant', 'property', 'formal',
        'mathematical', 'algorithm', 'complexity analysis', 'optimization',
        'constraint', 'specification', 'model checking', 'theorem proving'
      ],
      description: 'Verification, correctness proofs, invariants'
    },
    {
      name: 'novel_patterns',
      weight: 0.2,
      keywords: [
        'research', 'novel', 'innovative', 'experimental', 'cutting-edge',
        'state-of-the-art', 'machine learning', 'ai', 'blockchain', 'quantum',
        'federated learning', 'differential privacy', 'zero-knowledge', 'consensus'
      ],
      description: 'Unfamiliar domain, research-adjacent'
    },
    {
      name: 'integration_complexity',
      weight: 0.15,
      keywords: [
        'integration', 'external api', 'third-party', 'protocol', 'interface',
        'webhook', 'messaging', 'event-driven', 'pub-sub', 'queue', 'streaming',
        'real-time', 'synchronization', 'data pipeline', 'etl'
      ],
      description: 'Multiple external systems, protocols'
    },
    {
      name: 'strategic_decisions',
      weight: 0.05,
      keywords: [
        'technology selection', 'trade-off', 'decision', 'comparison', 'evaluation',
        'framework choice', 'database selection', 'cloud provider', 'architecture pattern',
        'design pattern', 'best practices', 'standards', 'compliance'
      ],
      description: 'Technology selection, trade-off analysis'
    }
  ],
  delegationTypeMappings: {
    'architecture': DelegationType.ARCHITECTURE_DESIGN,
    'system design': DelegationType.ARCHITECTURE_DESIGN,
    'api': DelegationType.API_DESIGN,
    'endpoint': DelegationType.API_DESIGN,
    'rest': DelegationType.API_DESIGN,
    'graphql': DelegationType.API_DESIGN,
    'test': DelegationType.TEST_STRATEGY,
    'testing': DelegationType.TEST_STRATEGY,
    'property-based': DelegationType.TEST_STRATEGY,
    'unit test': DelegationType.TEST_STRATEGY,
    'integration test': DelegationType.TEST_STRATEGY,
    'integration': DelegationType.INTEGRATION_DESIGN,
    'external system': DelegationType.INTEGRATION_DESIGN,
    'protocol': DelegationType.INTEGRATION_DESIGN,
    'refactor': DelegationType.REFACTORING_ANALYSIS,
    'refactoring': DelegationType.REFACTORING_ANALYSIS,
    'code smell': DelegationType.REFACTORING_ANALYSIS,
    'technical debt': DelegationType.REFACTORING_ANALYSIS,
    'verification': DelegationType.FORMAL_VERIFICATION,
    'proof': DelegationType.FORMAL_VERIFICATION,
    'correctness': DelegationType.FORMAL_VERIFICATION,
    'invariant': DelegationType.FORMAL_VERIFICATION,
  }
};

/**
 * Context requirements by delegation type
 */
const CONTEXT_REQUIREMENTS: Record<DelegationType, ContextType[]> = {
  [DelegationType.ARCHITECTURE_DESIGN]: [
    ContextType.ARCHITECTURE_DOCS,
    ContextType.CODE_SNIPPETS,
    ContextType.EXISTING_DESIGNS,
    ContextType.CONSTRAINTS,
    ContextType.CONFIG_FILES
  ],
  [DelegationType.API_DESIGN]: [
    ContextType.API_DEFINITIONS,
    ContextType.CODE_SNIPPETS,
    ContextType.EXISTING_DESIGNS,
    ContextType.CONSTRAINTS
  ],
  [DelegationType.TEST_STRATEGY]: [
    ContextType.TEST_FILES,
    ContextType.CODE_SNIPPETS,
    ContextType.REQUIREMENTS_DOCS,
    ContextType.EXISTING_DESIGNS
  ],
  [DelegationType.INTEGRATION_DESIGN]: [
    ContextType.PROTOCOL_SPECS,
    ContextType.API_DEFINITIONS,
    ContextType.CODE_SNIPPETS,
    ContextType.CONSTRAINTS,
    ContextType.CONFIG_FILES
  ],
  [DelegationType.REFACTORING_ANALYSIS]: [
    ContextType.CODE_SNIPPETS,
    ContextType.DEPENDENCY_GRAPHS,
    ContextType.ARCHITECTURE_DOCS,
    ContextType.CONSTRAINTS
  ],
  [DelegationType.FORMAL_VERIFICATION]: [
    ContextType.CODE_SNIPPETS,
    ContextType.REQUIREMENTS_DOCS,
    ContextType.EXISTING_DESIGNS,
    ContextType.CONSTRAINTS
  ]
};

/**
 * Expected artifacts by delegation type
 */
const EXPECTED_ARTIFACTS: Record<DelegationType, ArtifactType[]> = {
  [DelegationType.ARCHITECTURE_DESIGN]: [
    ArtifactType.MERMAID_DIAGRAM,
    ArtifactType.IMPLEMENTATION_GUIDE,
    ArtifactType.DESIGN_DOC
  ],
  [DelegationType.API_DESIGN]: [
    ArtifactType.OPENAPI_SPEC,
    ArtifactType.IMPLEMENTATION_GUIDE,
    ArtifactType.CODE_SNIPPET
  ],
  [DelegationType.TEST_STRATEGY]: [
    ArtifactType.TEST_STRATEGY,
    ArtifactType.CODE_SNIPPET,
    ArtifactType.IMPLEMENTATION_GUIDE
  ],
  [DelegationType.INTEGRATION_DESIGN]: [
    ArtifactType.MERMAID_DIAGRAM,
    ArtifactType.IMPLEMENTATION_GUIDE,
    ArtifactType.CODE_SNIPPET
  ],
  [DelegationType.REFACTORING_ANALYSIS]: [
    ArtifactType.IMPLEMENTATION_GUIDE,
    ArtifactType.CODE_SNIPPET,
    ArtifactType.MERMAID_DIAGRAM
  ],
  [DelegationType.FORMAL_VERIFICATION]: [
    ArtifactType.REQUIREMENTS,
    ArtifactType.IMPLEMENTATION_GUIDE,
    ArtifactType.CODE_SNIPPET
  ]
};

/**
 * Problem Classifier class
 */
export class ProblemClassifier {
  private config: ClassificationConfig;

  constructor(config?: Partial<ClassificationConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Classifies a problem for delegation suitability
   */
  classifyProblem(problemDescription: string, title?: string): ProblemClassification {
    const fullText = title ? `${title} ${problemDescription}` : problemDescription;
    const normalizedText = fullText.toLowerCase();

    // Calculate complexity scores
    const complexityScores = this.calculateComplexityScores(normalizedText);
    
    // Determine primary delegation type
    const delegationType = this.determineDelegationType(normalizedText, complexityScores);
    
    // Calculate overall suitability score
    const suitabilityScore = this.calculateSuitabilityScore(complexityScores);
    
    // Determine if suitable for delegation
    const suitable = suitabilityScore >= this.config.minSuitabilityScore;
    
    // Estimate complexity level
    const complexity = this.estimateComplexity(suitabilityScore, complexityScores);
    
    // Get required context types and expected artifacts
    const requiredContextTypes = CONTEXT_REQUIREMENTS[delegationType];
    const expectedArtifacts = EXPECTED_ARTIFACTS[delegationType];
    
    // Generate reasoning
    const reasoning = this.generateReasoning(
      complexityScores,
      delegationType,
      suitabilityScore,
      suitable
    );

    // Find alternative types
    const alternativeTypes = this.findAlternativeTypes(normalizedText, delegationType);

    return {
      suitable,
      delegationType,
      alternativeTypes: alternativeTypes.length > 0 ? alternativeTypes : undefined,
      complexity,
      requiredContextTypes,
      expectedArtifacts,
      suitabilityScore,
      reasoning
    };
  }

  /**
   * Generates extraction strategy recommendations for a delegation type
   */
  generateExtractionStrategy(delegationType: DelegationType, complexity: ComplexityLevel): ExtractionStrategy {
    const baseStrategy = this.getBaseExtractionStrategy(delegationType);
    
    // Adjust strategy based on complexity
    const maxFiles = complexity === ComplexityLevel.SIMPLE ? 10 : 
                    complexity === ComplexityLevel.MODERATE ? 20 : 30;
    
    const includeDependencies = complexity !== ComplexityLevel.SIMPLE;

    return {
      ...baseStrategy,
      maxFiles,
      includeDependencies
    };
  }

  /**
   * Calculates complexity scores for each indicator
   */
  private calculateComplexityScores(text: string): Record<string, number> {
    const scores: Record<string, number> = {};

    for (const indicator of this.config.complexityIndicators) {
      let matchCount = 0;
      for (const keyword of indicator.keywords) {
        if (text.includes(keyword)) {
          matchCount++;
        }
      }
      
      // Normalize score (0-1) based on keyword matches
      const maxPossibleMatches = Math.min(indicator.keywords.length, 5); // Cap at 5 for normalization
      scores[indicator.name] = Math.min(matchCount / maxPossibleMatches, 1.0);
    }

    return scores;
  }

  /**
   * Determines the primary delegation type based on text analysis
   */
  private determineDelegationType(text: string, complexityScores: Record<string, number>): DelegationType {
    // Check for explicit type keywords first
    for (const [keyword, type] of Object.entries(this.config.delegationTypeMappings)) {
      if (text.includes(keyword)) {
        return type;
      }
    }

    // Fall back to complexity score analysis
    const maxScore = Math.max(...Object.values(complexityScores));
    const dominantIndicator = Object.entries(complexityScores)
      .find(([_, score]) => score === maxScore)?.[0];

    switch (dominantIndicator) {
      case 'architectural_scope':
        return DelegationType.ARCHITECTURE_DESIGN;
      case 'formal_reasoning':
        return DelegationType.FORMAL_VERIFICATION;
      case 'novel_patterns':
        return DelegationType.ARCHITECTURE_DESIGN; // Novel patterns often need architecture
      case 'integration_complexity':
        return DelegationType.INTEGRATION_DESIGN;
      case 'strategic_decisions':
        return DelegationType.ARCHITECTURE_DESIGN;
      default:
        return DelegationType.ARCHITECTURE_DESIGN; // Default fallback
    }
  }

  /**
   * Calculates overall suitability score (0-100)
   */
  private calculateSuitabilityScore(complexityScores: Record<string, number>): number {
    let weightedSum = 0;
    let totalWeight = 0;

    for (const indicator of this.config.complexityIndicators) {
      const score = complexityScores[indicator.name] || 0;
      weightedSum += score * indicator.weight;
      totalWeight += indicator.weight;
    }

    return Math.round((weightedSum / totalWeight) * 100);
  }

  /**
   * Estimates complexity level based on scores
   */
  private estimateComplexity(suitabilityScore: number, complexityScores: Record<string, number>): ComplexityLevel {
    const maxComplexityScore = Math.max(...Object.values(complexityScores));
    
    if (suitabilityScore >= 80 || maxComplexityScore >= 0.8) {
      return ComplexityLevel.COMPLEX;
    } else if (suitabilityScore >= 60 || maxComplexityScore >= 0.5) {
      return ComplexityLevel.MODERATE;
    } else {
      return ComplexityLevel.SIMPLE;
    }
  }

  /**
   * Generates human-readable reasoning for the classification
   */
  private generateReasoning(
    complexityScores: Record<string, number>,
    delegationType: DelegationType,
    suitabilityScore: number,
    suitable: boolean
  ): string {
    const dominantIndicators = Object.entries(complexityScores)
      .filter(([_, score]) => score > 0.3)
      .sort(([_, a], [__, b]) => b - a)
      .slice(0, 2)
      .map(([name, _]) => name.replace('_', ' '));

    const typeDescription = this.getDelegationTypeDescription(delegationType);
    
    if (!suitable) {
      return `Problem has low complexity indicators (score: ${suitabilityScore}). ` +
             `Consider manual implementation or breaking into smaller, more focused problems.`;
    }

    let reasoning = `Problem shows ${dominantIndicators.length > 0 ? dominantIndicators.join(' and ') : 'complexity'} ` +
                   `indicators (score: ${suitabilityScore}), making it suitable for ${typeDescription} delegation.`;

    if (dominantIndicators.length > 1) {
      reasoning += ` Multiple complexity dimensions suggest this requires Opus's comprehensive reasoning capabilities.`;
    }

    return reasoning;
  }

  /**
   * Finds alternative delegation types that might also be suitable
   */
  private findAlternativeTypes(text: string, primaryType: DelegationType): DelegationType[] {
    const alternatives: DelegationType[] = [];

    // Check for secondary type indicators
    for (const [keyword, type] of Object.entries(this.config.delegationTypeMappings)) {
      if (type !== primaryType && text.includes(keyword)) {
        if (!alternatives.includes(type)) {
          alternatives.push(type);
        }
      }
    }

    return alternatives.slice(0, 2); // Limit to 2 alternatives
  }

  /**
   * Gets base extraction strategy for a delegation type
   */
  private getBaseExtractionStrategy(delegationType: DelegationType): Omit<ExtractionStrategy, 'maxFiles' | 'includeDependencies'> {
    switch (delegationType) {
      case DelegationType.ARCHITECTURE_DESIGN:
        return {
          primaryPatterns: ['**/*.md', '**/architecture/**', '**/docs/**', '**/design/**'],
          secondaryPatterns: ['**/config/**', '**/deploy/**', '**/infrastructure/**'],
          keywords: ['architecture', 'component', 'service', 'system', 'design', 'structure']
        };
      
      case DelegationType.API_DESIGN:
        return {
          primaryPatterns: ['**/api/**', '**/routes/**', '**/controllers/**', '**/handlers/**'],
          secondaryPatterns: ['**/models/**', '**/schemas/**', '**/types/**'],
          keywords: ['api', 'endpoint', 'route', 'controller', 'handler', 'request', 'response']
        };
      
      case DelegationType.TEST_STRATEGY:
        return {
          primaryPatterns: ['**/test/**', '**/tests/**', '**/*.test.*', '**/*.spec.*'],
          secondaryPatterns: ['**/src/**', '**/lib/**'],
          keywords: ['test', 'spec', 'assert', 'expect', 'mock', 'fixture', 'property']
        };
      
      case DelegationType.INTEGRATION_DESIGN:
        return {
          primaryPatterns: ['**/integration/**', '**/external/**', '**/adapters/**', '**/connectors/**'],
          secondaryPatterns: ['**/config/**', '**/protocols/**'],
          keywords: ['integration', 'external', 'adapter', 'connector', 'protocol', 'interface']
        };
      
      case DelegationType.REFACTORING_ANALYSIS:
        return {
          primaryPatterns: ['**/src/**', '**/lib/**'],
          secondaryPatterns: ['**/docs/**', '**/README.md'],
          keywords: ['class', 'function', 'method', 'module', 'component', 'service']
        };
      
      case DelegationType.FORMAL_VERIFICATION:
        return {
          primaryPatterns: ['**/src/**', '**/lib/**', '**/specs/**'],
          secondaryPatterns: ['**/docs/**', '**/requirements/**'],
          keywords: ['algorithm', 'invariant', 'property', 'constraint', 'specification', 'proof']
        };
      
      default:
        return {
          primaryPatterns: ['**/src/**', '**/lib/**'],
          secondaryPatterns: ['**/docs/**'],
          keywords: ['function', 'class', 'component', 'service']
        };
    }
  }

  /**
   * Gets human-readable description for delegation type
   */
  private getDelegationTypeDescription(type: DelegationType): string {
    switch (type) {
      case DelegationType.ARCHITECTURE_DESIGN:
        return 'architecture design';
      case DelegationType.API_DESIGN:
        return 'API design';
      case DelegationType.TEST_STRATEGY:
        return 'test strategy';
      case DelegationType.INTEGRATION_DESIGN:
        return 'integration design';
      case DelegationType.REFACTORING_ANALYSIS:
        return 'refactoring analysis';
      case DelegationType.FORMAL_VERIFICATION:
        return 'formal verification';
      default:
        return 'general design';
    }
  }
}

/**
 * Default problem classifier instance
 */
export const problemClassifier = new ProblemClassifier();

/**
 * Convenience function for problem classification
 */
export function classifyProblem(problemDescription: string, title?: string): ProblemClassification {
  return problemClassifier.classifyProblem(problemDescription, title);
}

/**
 * Convenience function for extraction strategy generation
 */
export function generateExtractionStrategy(delegationType: DelegationType, complexity: ComplexityLevel): ExtractionStrategy {
  return problemClassifier.generateExtractionStrategy(delegationType, complexity);
}