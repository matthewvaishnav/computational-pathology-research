/**
 * Spec Workflow Adapter Component
 * 
 * Integrates Opus artifacts with the existing spec workflow by generating
 * requirements.md, design.md, and tasks.md from Opus artifacts.
 * 
 * Requirements: 16.1, 16.4
 */

import { ParsedArtifact, ArtifactType, ImplementationGuide, ImplementationStep, ComplexityLevel } from '../types/core.js';

export interface EARSRequirement {
  id: string;
  title: string;
  userStory: string;
  acceptanceCriteria: string[];
  propertyBasedTestingGuidance?: string[];
}

export interface SpecDocument {
  title: string;
  content: string;
  metadata: {
    generatedAt: Date;
    sourceArtifacts: string[];
    earsCompliant: boolean;
    validationErrors: string[];
  };
}

export interface ConfigKiro {
  specId: string;
  workflowType: 'requirements-first' | 'design-first' | 'bugfix';
  specType: 'feature' | 'bugfix';
  metadata?: {
    generatedAt: Date;
    sourceArtifacts: string[];
    projectName?: string;
  };
}

export interface ConfigGenerationOptions {
  specId?: string;
  workflowType?: 'requirements-first' | 'design-first' | 'bugfix';
  specType?: 'feature' | 'bugfix';
  projectName?: string;
}

export interface HybridWorkflowOptions {
  projectName?: string;
  opusSource: 'design' | 'requirements' | 'tasks';
  localSource: 'requirements' | 'tasks' | 'design';
  includeSourceTracking?: boolean;
  validateConsistency?: boolean;
}

export interface HybridSpecPackage {
  requirements?: SpecDocument;
  design?: SpecDocument;
  tasks?: TasksDocument;
  config: ConfigKiro;
  metadata: {
    generatedAt: Date;
    opusArtifacts: string[];
    localArtifacts: string[];
    hybridType: string;
    consistencyValidation?: string[];
  };
}

export interface RequirementsGenerationOptions {
  projectName?: string;
  includePropertyBasedTesting?: boolean;
  earsValidation?: boolean;
  requirementIdPrefix?: string;
}

export interface DesignGenerationOptions {
  projectName?: string;
  includeArchitectureDiagrams?: boolean;
  includeSequenceDiagrams?: boolean;
  includeComponentDetails?: boolean;
  includeDataFlow?: boolean;
  includeTechnologyStack?: boolean;
  includeImplementationNotes?: boolean;
}

export interface TasksGenerationOptions {
  projectName?: string;
  includeRequirementsReferences?: boolean;
  includeDependencies?: boolean;
  includeComplexityEstimates?: boolean;
  taskIdPrefix?: string;
  groupByPhase?: boolean;
}

export interface TaskHierarchy {
  id: string;
  title: string;
  description?: string;
  status: 'not_started' | 'in_progress' | 'completed';
  subtasks?: TaskHierarchy[];
  dependencies?: string[];
  requirementReferences?: string[];
  complexity?: ComplexityLevel;
  estimate?: string;
  phase?: string;
}

export interface TasksDocument {
  title: string;
  content: string;
  metadata: {
    generatedAt: Date;
    sourceArtifacts: string[];
    taskCount: number;
    validationErrors: string[];
  };
}

/**
 * Design element interfaces for structured design document generation
 */
export interface DesignElements {
  overview: {
    problemStatement?: string;
    solutionApproach?: string;
    keyPrinciples?: string[];
    keyInnovations?: string[];
  };
  architecture: {
    diagrams: Array<{
      type: 'architecture' | 'sequence' | 'component' | 'data_flow';
      title: string;
      content: string;
      description?: string;
    }>;
    components: Array<{
      name: string;
      purpose: string;
      subcomponents?: Array<{
        name: string;
        description: string;
      }>;
      technologyStack?: string[];
      interfaces?: string[];
    }>;
  };
  dataFlow: {
    phases?: Array<{
      name: string;
      description: string;
      steps: string[];
    }>;
    sequences?: Array<{
      name: string;
      description: string;
      actors: string[];
    }>;
  };
  apiDesign: {
    endpoints?: Array<{
      path: string;
      method: string;
      description: string;
      requestSchema?: any;
      responseSchema?: any;
    }>;
    dataModels?: Array<{
      name: string;
      properties: Array<{
        name: string;
        type: string;
        description?: string;
      }>;
    }>;
  };
  technologyStack: {
    languages?: string[];
    frameworks?: string[];
    databases?: string[];
    infrastructure?: string[];
    tools?: string[];
  };
  implementationNotes: {
    designDecisions?: Array<{
      decision: string;
      rationale: string;
      alternatives?: string[];
    }>;
    constraints?: string[];
    assumptions?: string[];
    risks?: Array<{
      risk: string;
      mitigation: string;
    }>;
  };
}

/**
 * Spec Workflow Adapter - converts Opus artifacts to spec workflow documents
 */
export class SpecWorkflowAdapter {
  
  /**
   * Generate design.md from Opus architecture and API designs following project design template
   * 
   * Requirements: 16.2, 16.4
   */
  async generateDesign(
    artifacts: ParsedArtifact[],
    options: DesignGenerationOptions = {}
  ): Promise<SpecDocument> {
    const {
      projectName = 'Generated Project',
      includeArchitectureDiagrams = true,
      includeSequenceDiagrams = true,
      includeComponentDetails = true,
      includeDataFlow = true,
      includeTechnologyStack = true,
      includeImplementationNotes = true
    } = options;

    // Create options object with defaults applied
    const optionsWithDefaults: DesignGenerationOptions = {
      projectName,
      includeArchitectureDiagrams,
      includeSequenceDiagrams,
      includeComponentDetails,
      includeDataFlow,
      includeTechnologyStack,
      includeImplementationNotes
    };

    // Extract design elements from various artifact types
    const designElements = this.extractDesignElementsFromArtifacts(artifacts);
    
    // Generate design document content
    const content = this.generateDesignDocument(projectName, designElements, optionsWithDefaults);
    
    // Validate design completeness
    const validationErrors = this.validateDesignCompleteness(designElements, optionsWithDefaults);
    
    return {
      title: `${projectName} Design Document`,
      content,
      metadata: {
        generatedAt: new Date(),
        sourceArtifacts: artifacts.map(a => a.id),
        earsCompliant: true, // Design documents don't need EARS compliance
        validationErrors
      }
    };
  }
  /**
   * Generate requirements.md from Opus-provided requirements using EARS patterns
   * 
   * Requirements: 16.1, 16.4
   */
  async generateRequirements(
    artifacts: ParsedArtifact[],
    options: RequirementsGenerationOptions = {}
  ): Promise<SpecDocument> {
    const {
      projectName = 'Generated Project',
      includePropertyBasedTesting = true,
      earsValidation = true,
      requirementIdPrefix = 'REQ'
    } = options;

    // Extract requirements from various artifact types
    const requirements = this.extractRequirementsFromArtifacts(artifacts);
    
    // Convert to EARS patterns
    const earsRequirements = requirements.map((req, index) => 
      this.convertToEARSPattern(req, `${requirementIdPrefix}-${(index + 1).toString().padStart(2, '0')}`)
    );

    // Generate requirements document content
    const content = this.generateRequirementsDocument(projectName, earsRequirements, includePropertyBasedTesting);
    
    // Validate EARS compliance if requested
    const validationErrors = earsValidation ? this.validateEARSCompliance(earsRequirements) : [];
    
    return {
      title: `${projectName} Requirements`,
      content,
      metadata: {
        generatedAt: new Date(),
        sourceArtifacts: artifacts.map(a => a.id),
        earsCompliant: validationErrors.length === 0,
        validationErrors
      }
    };
  }

  /**
   * Extract requirements from various Opus artifact types
   */
  private extractRequirementsFromArtifacts(artifacts: ParsedArtifact[]): Partial<EARSRequirement>[] {
    const requirements: Partial<EARSRequirement>[] = [];

    for (const artifact of artifacts) {
      switch (artifact.type) {
        case 'implementation_guide':
          requirements.push(...this.extractRequirementsFromImplementationGuide(artifact));
          break;
        case 'openapi_spec':
          requirements.push(...this.extractRequirementsFromAPISpec(artifact));
          break;
        case 'mermaid_diagram':
          requirements.push(...this.extractRequirementsFromArchitecture(artifact));
          break;
        case 'test_strategy':
          requirements.push(...this.extractRequirementsFromTestStrategy(artifact));
          break;
        default:
          // Try to extract from general content
          requirements.push(...this.extractRequirementsFromContent(artifact.content));
      }
    }

    return requirements;
  }

  /**
   * Extract requirements from implementation guide artifacts
   */
  private extractRequirementsFromImplementationGuide(artifact: ParsedArtifact): Partial<EARSRequirement>[] {
    const requirements: Partial<EARSRequirement>[] = [];
    
    if (artifact.structured?.implementationSteps) {
      // Group steps by phase to create requirements
      const phaseGroups = this.groupStepsByPhase(artifact.structured.implementationSteps);
      
      for (const [phase, steps] of Object.entries(phaseGroups)) {
        const requirement: Partial<EARSRequirement> = {
          title: `${phase} Implementation`,
          userStory: `As a developer, I want to implement ${phase.toLowerCase()}, so that the system provides the required functionality.`,
          acceptanceCriteria: steps.map(step => this.convertStepToEARSCriteria(step))
        };
        requirements.push(requirement);
      }
    }

    return requirements;
  }

  /**
   * Extract requirements from OpenAPI specification artifacts
   */
  private extractRequirementsFromAPISpec(artifact: ParsedArtifact): Partial<EARSRequirement>[] {
    const requirements: Partial<EARSRequirement>[] = [];
    
    try {
      const apiSpec = artifact.structured?.openapi || JSON.parse(artifact.content);
      
      if (apiSpec.paths) {
        const endpoints = Object.keys(apiSpec.paths);
        const endpointGroups = this.groupEndpointsByResource(endpoints);
        
        for (const [resource, resourceEndpoints] of Object.entries(endpointGroups)) {
          const acceptanceCriteria: string[] = [];
          
          // Process each endpoint for this resource
          for (const endpoint of resourceEndpoints) {
            const endpointSpec = apiSpec.paths[endpoint];
            const methods = Object.keys(endpointSpec);
            
            // Generate criteria for each HTTP method
            for (const method of methods) {
              const methodUpper = method.toUpperCase();
              const resourceName = endpoint.split('/').filter(p => p && !p.startsWith('{')).pop() || 'resource';
              
              if (method === 'get') {
                acceptanceCriteria.push(`THE API SHALL provide GET ${endpoint} endpoint to retrieve ${resourceName} data`);
              } else if (method === 'post') {
                acceptanceCriteria.push(`THE API SHALL provide POST ${endpoint} endpoint to create new ${resourceName}`);
              } else if (method === 'put') {
                acceptanceCriteria.push(`THE API SHALL provide PUT ${endpoint} endpoint to update ${resourceName}`);
              } else if (method === 'delete') {
                acceptanceCriteria.push(`THE API SHALL provide DELETE ${endpoint} endpoint to remove ${resourceName}`);
              } else {
                acceptanceCriteria.push(`THE API SHALL provide ${methodUpper} ${endpoint} endpoint for ${resourceName} operations`);
              }
            }
          }
          
          const requirement: Partial<EARSRequirement> = {
            title: `${resource} API Operations`,
            userStory: `As a client application, I want to interact with ${resource} resources via API, so that I can perform required operations.`,
            acceptanceCriteria
          };
          requirements.push(requirement);
        }
      }
    } catch (error) {
      // If parsing fails, extract from content text
      requirements.push(...this.extractRequirementsFromContent(artifact.content));
    }

    return requirements;
  }

  /**
   * Extract requirements from Mermaid architecture diagrams
   */
  private extractRequirementsFromArchitecture(artifact: ParsedArtifact): Partial<EARSRequirement>[] {
    const requirements: Partial<EARSRequirement>[] = [];
    
    // Parse Mermaid content to identify components and relationships
    const components = this.extractComponentsFromMermaid(artifact.content);
    const relationships = this.extractRelationshipsFromMermaid(artifact.content);
    
    if (components.length > 0) {
      const acceptanceCriteria: string[] = [];
      
      // Add component implementation requirements
      const componentNames = this.extractComponentNamesFromMermaid(artifact.content);
      for (const comp of componentNames) {
        acceptanceCriteria.push(`THE System SHALL implement ${comp} component with defined interfaces`);
      }
      
      // Add relationship requirements
      for (const rel of relationships) {
        const fromName = this.getComponentNameFromId(artifact.content, rel.from) || rel.from;
        const toName = this.getComponentNameFromId(artifact.content, rel.to) || rel.to;
        acceptanceCriteria.push(`THE ${fromName} component SHALL ${rel.relationship} ${toName} component`);
      }
      
      const requirement: Partial<EARSRequirement> = {
        title: 'System Architecture Components',
        userStory: 'As a system architect, I want to implement the defined system components, so that the system architecture meets the design requirements.',
        acceptanceCriteria
      };
      requirements.push(requirement);
    }

    return requirements;
  }

  /**
   * Extract requirements from test strategy artifacts
   */
  private extractRequirementsFromTestStrategy(artifact: ParsedArtifact): Partial<EARSRequirement>[] {
    const requirements: Partial<EARSRequirement>[] = [];
    
    // Extract test categories and coverage requirements
    const testCategories = this.extractTestCategoriesFromContent(artifact.content);
    
    if (testCategories.length > 0) {
      const requirement: Partial<EARSRequirement> = {
        title: 'Testing and Quality Assurance',
        userStory: 'As a quality assurance engineer, I want comprehensive test coverage, so that the system meets quality and reliability standards.',
        acceptanceCriteria: testCategories.map(category => 
          `THE System SHALL provide ${category.type} tests with ${category.coverage}% coverage for ${category.scope}`
        ),
        propertyBasedTestingGuidance: this.extractPropertyBasedTestGuidance(artifact.content)
      };
      requirements.push(requirement);
    }

    return requirements;
  }

  /**
   * Extract requirements from general content using pattern matching
   */
  private extractRequirementsFromContent(content: string): Partial<EARSRequirement>[] {
    const requirements: Partial<EARSRequirement>[] = [];
    
    // Look for requirement-like patterns in the content
    const requirementPatterns = [
      /(?:must|should|shall|will)\s+([^.]+)/gi,
      /(?:requirement|req):\s*([^.]+)/gi,
      /the\s+system\s+(?:must|should|shall|will)\s+([^.]+)/gi
    ];

    const foundRequirements = new Set<string>();
    
    for (const pattern of requirementPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const reqText = match[1]?.trim();
        if (reqText && reqText.length > 10 && !foundRequirements.has(reqText)) {
          foundRequirements.add(reqText);
        }
      }
    }

    if (foundRequirements.size > 0) {
      const requirement: Partial<EARSRequirement> = {
        title: 'General System Requirements',
        userStory: 'As a user, I want the system to meet specified requirements, so that it provides the expected functionality.',
        acceptanceCriteria: Array.from(foundRequirements).map(req => 
          this.convertToEARSStatement(req)
        )
      };
      requirements.push(requirement);
    }

    return requirements;
  }

  /**
   * Convert a requirement to EARS pattern format
   */
  private convertToEARSPattern(requirement: Partial<EARSRequirement>, id: string): EARSRequirement {
    return {
      id,
      title: requirement.title || 'Untitled Requirement',
      userStory: requirement.userStory || 'As a user, I want this functionality, so that I can achieve my goals.',
      acceptanceCriteria: requirement.acceptanceCriteria || [],
      propertyBasedTestingGuidance: requirement.propertyBasedTestingGuidance
    };
  }

  /**
   * Convert implementation step to EARS acceptance criteria
   */
  private convertStepToEARSCriteria(step: ImplementationStep): string {
    const action = step.action.toLowerCase();
    const component = step.file ? `${step.file} component` : 'System';
    
    if (action.includes('create') || action.includes('implement')) {
      return `THE ${component} SHALL implement ${step.title.toLowerCase()}`;
    } else if (action.includes('configure') || action.includes('setup')) {
      return `THE ${component} SHALL configure ${step.title.toLowerCase()}`;
    } else if (action.includes('validate') || action.includes('test')) {
      return `THE ${component} SHALL validate ${step.title.toLowerCase()}`;
    } else {
      return `THE ${component} SHALL ${action} ${step.title.toLowerCase()}`;
    }
  }

  /**
   * Convert API endpoint to EARS acceptance criteria
   */
  private convertEndpointToEARSCriteria(endpoint: string, endpointSpec: any): string {
    const methods = Object.keys(endpointSpec);
    const resource = endpoint.split('/').filter(p => p && !p.startsWith('{')).pop() || 'resource';
    
    if (methods.includes('get')) {
      return `THE API SHALL provide GET ${endpoint} endpoint to retrieve ${resource} data`;
    } else if (methods.includes('post')) {
      return `THE API SHALL provide POST ${endpoint} endpoint to create new ${resource}`;
    } else if (methods.includes('put')) {
      return `THE API SHALL provide PUT ${endpoint} endpoint to update ${resource}`;
    } else if (methods.includes('delete')) {
      return `THE API SHALL provide DELETE ${endpoint} endpoint to remove ${resource}`;
    } else {
      return `THE API SHALL provide ${endpoint} endpoint with ${methods.join(', ')} operations`;
    }
  }

  /**
   * Convert general requirement text to EARS statement
   */
  private convertToEARSStatement(requirement: string): string {
    // Clean up the requirement text
    let cleaned = requirement.trim();
    
    // Remove common prefixes
    cleaned = cleaned.replace(/^(?:the\s+system\s+|system\s+|must\s+|should\s+|shall\s+|will\s+)/i, '');
    
    // Ensure it starts with proper EARS format
    if (!cleaned.toLowerCase().startsWith('the ')) {
      cleaned = `THE System SHALL ${cleaned}`;
    }
    
    return cleaned;
  }

  /**
   * Group implementation steps by phase
   */
  private groupStepsByPhase(steps: ImplementationStep[]): Record<string, ImplementationStep[]> {
    const groups: Record<string, ImplementationStep[]> = {};
    
    for (const step of steps) {
      const phase = step.phase || 'Implementation';
      if (!groups[phase]) {
        groups[phase] = [];
      }
      groups[phase].push(step);
    }
    
    return groups;
  }

  /**
   * Group API endpoints by resource
   */
  private groupEndpointsByResource(endpoints: string[]): Record<string, string[]> {
    const groups: Record<string, string[]> = {};
    
    for (const endpoint of endpoints) {
      const parts = endpoint.split('/').filter(p => p && !p.startsWith('{'));
      const resource = parts[parts.length - 1] || 'api';
      
      if (!groups[resource]) {
        groups[resource] = [];
      }
      groups[resource].push(endpoint);
    }
    
    return groups;
  }

  /**
   * Extract components from Mermaid diagram content
   */
  private extractComponentsFromMermaid(content: string): string[] {
    const components: string[] = [];
    
    // Look for node definitions in various Mermaid formats
    const nodePatterns = [
      /(\w+)\[([^\]]+)\]/g,  // [label] format
      /(\w+)\(([^)]+)\)/g,   // (label) format  
      /(\w+)\{([^}]+)\}/g,   // {label} format
      /(\w+)\s*-->\s*(\w+)/g // arrow connections
    ];

    for (const pattern of nodePatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        if (match[1] && !components.includes(match[1])) {
          components.push(match[1]);
        }
        if (match[2] && !components.includes(match[2])) {
          components.push(match[2]);
        }
      }
    }

    return components;
  }

  /**
   * Extract component names (labels) from Mermaid diagram content
   */
  private extractComponentNamesFromMermaid(content: string): string[] {
    const componentNames: string[] = [];
    
    // Look for node definitions with labels
    const labelPatterns = [
      /\w+\[([^\]]+)\]/g,  // [label] format
      /\w+\(([^)]+)\)/g,   // (label) format  
      /\w+\{([^}]+)\}/g    // {label} format
    ];

    for (const pattern of labelPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const label = match[1]?.trim();
        if (label && !componentNames.includes(label)) {
          componentNames.push(label);
        }
      }
    }

    return componentNames;
  }

  /**
   * Get component name from ID in Mermaid content
   */
  private getComponentNameFromId(content: string, id: string): string | null {
    // Look for the ID with a label
    const patterns = [
      new RegExp(`${id}\\[([^\\]]+)\\]`, 'g'),  // [label] format
      new RegExp(`${id}\\(([^)]+)\\)`, 'g'),    // (label) format  
      new RegExp(`${id}\\{([^}]+)\\}`, 'g')     // {label} format
    ];

    for (const pattern of patterns) {
      const match = pattern.exec(content);
      if (match && match[1]) {
        return match[1].trim();
      }
    }

    return null;
  }

  /**
   * Extract relationships from Mermaid diagram content
   */
  private extractRelationshipsFromMermaid(content: string): Array<{from: string, to: string, relationship: string}> {
    const relationships: Array<{from: string, to: string, relationship: string}> = [];
    
    // Look for arrow patterns with node labels
    const arrowPattern = /(\w+)(?:\[[^\]]+\])?\s*(-->|->|<->|<--)\s*(\w+)(?:\[[^\]]+\])?(?:\s*:\s*([^|\n]+))?/g;
    const matches = content.matchAll(arrowPattern);
    
    // Also look for simple node to node patterns
    const simplePattern = /(\w+)\s*(-->|->|<->|<--)\s*(\w+)/g;
    const simpleMatches = content.matchAll(simplePattern);
    
    // Process arrow patterns first
    for (const match of matches) {
      const [, from, arrow, to, label] = match;
      let relationship = 'communicate with';
      
      if (arrow === '-->') relationship = 'send data to';
      else if (arrow === '<--') relationship = 'receive data from';
      else if (arrow === '<->') relationship = 'exchange data with';
      
      if (label) {
        relationship = label.trim().toLowerCase();
      }
      
      relationships.push({ from, to, relationship });
    }
    
    // If no relationships found with labels, try simple patterns
    if (relationships.length === 0) {
      for (const match of simpleMatches) {
        const [, from, arrow, to] = match;
        let relationship = 'communicate with';
        
        if (arrow === '-->') relationship = 'send data to';
        else if (arrow === '<--') relationship = 'receive data from';
        else if (arrow === '<->') relationship = 'exchange data with';
        
        relationships.push({ from, to, relationship });
      }
    }
    
    return relationships;
  }

  /**
   * Extract test categories from test strategy content
   */
  private extractTestCategoriesFromContent(content: string): Array<{type: string, coverage: number, scope: string}> {
    const categories: Array<{type: string, coverage: number, scope: string}> = [];
    
    // Look for test coverage patterns
    const coveragePattern = /(\w+)\s+test[s]?\s*:?\s*(\d+)%?\s*coverage\s*(?:for\s+([^.\n]+))?/gi;
    const matches = content.matchAll(coveragePattern);
    
    for (const match of matches) {
      const [, type, coverage, scope] = match;
      categories.push({
        type: type.toLowerCase(),
        coverage: parseInt(coverage) || 80,
        scope: scope?.trim() || 'core functionality'
      });
    }
    
    // Default categories if none found
    if (categories.length === 0) {
      categories.push(
        { type: 'unit', coverage: 90, scope: 'individual components' },
        { type: 'integration', coverage: 80, scope: 'component interactions' },
        { type: 'property-based', coverage: 70, scope: 'system invariants' }
      );
    }
    
    return categories;
  }

  /**
   * Extract property-based testing guidance from content
   */
  private extractPropertyBasedTestGuidance(content: string): string[] {
    const guidance: string[] = [];
    
    // Look for property patterns
    const propertyPatterns = [
      /invariant:\s*([^.\n]+)/gi,
      /property:\s*([^.\n]+)/gi,
      /round-trip:\s*([^.\n]+)/gi,
      /metamorphic:\s*([^.\n]+)/gi
    ];

    for (const pattern of propertyPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const property = match[1]?.trim();
        if (property) {
          guidance.push(`**${match[0].split(':')[0].trim()}**: ${property}`);
        }
      }
    }

    return guidance;
  }

  /**
   * Generate the complete requirements document
   */
  private generateRequirementsDocument(
    projectName: string, 
    requirements: EARSRequirement[], 
    includePropertyBasedTesting: boolean
  ): string {
    const sections = [
      `# Requirements Document: ${projectName}`,
      '',
      '## Introduction',
      '',
      `This document specifies requirements for ${projectName}. The requirements are written using EARS (Easy Approach to Requirements Syntax) patterns to ensure clarity and testability.`,
      '',
      '## Requirements',
      ''
    ];

    for (let i = 0; i < requirements.length; i++) {
      const req = requirements[i];
      
      sections.push(`### Requirement ${i + 1}: ${req.title}`);
      sections.push('');
      sections.push(`**User Story:** ${req.userStory}`);
      sections.push('');
      sections.push('#### Acceptance Criteria');
      sections.push('');
      
      req.acceptanceCriteria.forEach((criteria, index) => {
        sections.push(`${index + 1}. ${criteria}`);
      });
      
      if (includePropertyBasedTesting && req.propertyBasedTestingGuidance && req.propertyBasedTestingGuidance.length > 0) {
        sections.push('');
        sections.push('**Property-Based Testing Guidance:**');
        req.propertyBasedTestingGuidance.forEach(guidance => {
          sections.push(`- ${guidance}`);
        });
      }
      
      sections.push('');
    }

    return sections.join('\n');
  }

  /**
   * Validate EARS compliance of generated requirements
   */
  private validateEARSCompliance(requirements: EARSRequirement[]): string[] {
    const errors: string[] = [];

    for (const req of requirements) {
      // Check that acceptance criteria follow EARS patterns
      for (const criteria of req.acceptanceCriteria) {
        if (!this.isValidEARSStatement(criteria)) {
          errors.push(`Requirement ${req.id}: "${criteria}" does not follow EARS pattern`);
        }
      }

      // Check for required elements
      if (!req.userStory || req.userStory.length < 10) {
        errors.push(`Requirement ${req.id}: User story is missing or too short`);
      }

      if (!req.acceptanceCriteria || req.acceptanceCriteria.length === 0) {
        errors.push(`Requirement ${req.id}: No acceptance criteria defined`);
      }
    }

    return errors;
  }

  /**
   * Check if a statement follows EARS pattern
   */
  private isValidEARSStatement(statement: string): boolean {
    // EARS patterns typically start with:
    // - THE <system> SHALL <action>
    // - WHEN <condition>, THE <system> SHALL <action>
    // - IF <condition>, THEN THE <system> SHALL <action>
    // - WHERE <condition>, THE <system> SHALL <action>
    
    const earsPatterns = [
      /^THE\s+\w+\s+SHALL\s+/i,
      /^WHEN\s+.+,\s*THE\s+\w+\s+SHALL\s+/i,
      /^IF\s+.+,\s*THEN\s+THE\s+\w+\s+SHALL\s+/i,
      /^WHERE\s+.+,\s*THE\s+\w+\s+SHALL\s+/i
    ];

    return earsPatterns.some(pattern => pattern.test(statement.trim()));
  }

  // ===== DESIGN GENERATION METHODS =====

  /**
   * Extract design elements from various Opus artifact types
   */
  private extractDesignElementsFromArtifacts(artifacts: ParsedArtifact[]): DesignElements {
    const designElements: DesignElements = {
      overview: {},
      architecture: { diagrams: [], components: [] },
      dataFlow: {},
      apiDesign: {},
      technologyStack: {},
      implementationNotes: {}
    };

    for (const artifact of artifacts) {
      switch (artifact.type) {
        case 'mermaid_diagram':
          this.extractArchitectureFromMermaid(artifact, designElements);
          break;
        case 'openapi_spec':
          this.extractAPIDesignFromSpec(artifact, designElements);
          break;
        case 'implementation_guide':
          this.extractImplementationElements(artifact, designElements);
          break;
        case 'test_strategy':
          this.extractTestingElements(artifact, designElements);
          break;
        default:
          // Extract from general content
          this.extractGeneralDesignElements(artifact, designElements);
      }
    }

    return designElements;
  }

  /**
   * Extract architecture elements from Mermaid diagrams
   */
  private extractArchitectureFromMermaid(artifact: ParsedArtifact, designElements: DesignElements): void {
    const content = artifact.content;
    
    // Determine diagram type
    let diagramType: 'architecture' | 'sequence' | 'component' | 'data_flow' = 'architecture';
    if (content.includes('sequenceDiagram')) {
      diagramType = 'sequence';
    } else if (content.includes('classDiagram')) {
      diagramType = 'component';
    } else if (content.includes('flowchart') || content.includes('graph')) {
      diagramType = 'data_flow';
    }

    // Add diagram - always add the diagram even if we can't extract other elements
    designElements.architecture.diagrams.push({
      type: diagramType,
      title: this.extractDiagramTitle(content) || `${diagramType.charAt(0).toUpperCase() + diagramType.slice(1)} Diagram`,
      content: content,
      description: this.extractDiagramDescription(content) || undefined
    });

    // Extract components from diagram
    const components = this.extractComponentsFromMermaidForDesign(content);
    designElements.architecture.components.push(...components);

    // Extract data flow if it's a flowchart
    if (diagramType === 'data_flow') {
      const phases = this.extractDataFlowPhases(content);
      if (phases.length > 0) {
        designElements.dataFlow.phases = phases;
      }
    }

    // Extract sequence information if it's a sequence diagram
    if (diagramType === 'sequence') {
      const sequences = this.extractSequenceFlows(content);
      if (sequences.length > 0) {
        designElements.dataFlow.sequences = sequences;
      }
    }
  }

  /**
   * Extract API design elements from OpenAPI specification
   */
  private extractAPIDesignFromSpec(artifact: ParsedArtifact, designElements: DesignElements): void {
    try {
      const apiSpec = artifact.structured?.openapi || JSON.parse(artifact.content);
      
      // Extract endpoints
      if (apiSpec.paths) {
        const endpoints: any[] = [];
        for (const [path, pathSpec] of Object.entries(apiSpec.paths as Record<string, any>)) {
          for (const [method, methodSpec] of Object.entries(pathSpec as Record<string, any>)) {
            endpoints.push({
              path,
              method: method.toUpperCase(),
              description: methodSpec.description || methodSpec.summary || `${method.toUpperCase()} ${path}`,
              requestSchema: methodSpec.requestBody?.content?.['application/json']?.schema,
              responseSchema: methodSpec.responses?.['200']?.content?.['application/json']?.schema
            });
          }
        }
        designElements.apiDesign.endpoints = endpoints;
      }

      // Extract data models
      if (apiSpec.components?.schemas) {
        const dataModels: any[] = [];
        for (const [modelName, schema] of Object.entries(apiSpec.components.schemas as Record<string, any>)) {
          const properties: any[] = [];
          if (schema.properties) {
            for (const [propName, propSpec] of Object.entries(schema.properties as Record<string, any>)) {
              properties.push({
                name: propName,
                type: (propSpec as any).type || 'unknown',
                description: (propSpec as any).description
              });
            }
          }
          dataModels.push({
            name: modelName,
            properties
          });
        }
        designElements.apiDesign.dataModels = dataModels;
      }

      // Extract technology stack from API spec
      if (apiSpec.info?.description) {
        const techStack = this.extractTechnologyFromText(apiSpec.info.description);
        Object.assign(designElements.technologyStack, techStack);
      }

    } catch (error) {
      // If parsing fails, extract from content text
      this.extractGeneralDesignElements(artifact, designElements);
    }
  }

  /**
   * Extract implementation elements from implementation guide
   */
  private extractImplementationElements(artifact: ParsedArtifact, designElements: DesignElements): void {
    const content = artifact.content;
    
    // Extract technology stack
    const techStack = this.extractTechnologyFromText(content);
    Object.assign(designElements.technologyStack, techStack);

    // Extract design decisions
    const decisions = this.extractDesignDecisions(content);
    if (decisions.length > 0) {
      designElements.implementationNotes.designDecisions = decisions;
    }

    // Extract constraints and assumptions
    const constraints = this.extractConstraints(content);
    const assumptions = this.extractAssumptions(content);
    
    if (constraints.length > 0) {
      designElements.implementationNotes.constraints = constraints;
    }
    if (assumptions.length > 0) {
      designElements.implementationNotes.assumptions = assumptions;
    }

    // Extract implementation phases
    if (artifact.structured?.implementationSteps) {
      const phases = this.groupImplementationPhases(artifact.structured.implementationSteps);
      if (phases.length > 0) {
        designElements.dataFlow.phases = phases;
      }
    }
  }

  /**
   * Extract testing elements that inform design
   */
  private extractTestingElements(artifact: ParsedArtifact, designElements: DesignElements): void {
    const content = artifact.content;
    
    // Extract quality attributes and constraints
    const qualityConstraints = this.extractQualityConstraints(content);
    if (qualityConstraints.length > 0) {
      if (!designElements.implementationNotes.constraints) {
        designElements.implementationNotes.constraints = [];
      }
      designElements.implementationNotes.constraints.push(...qualityConstraints);
    }

    // Extract testing technology stack
    const testTechStack = this.extractTestingTechnology(content);
    if (testTechStack.length > 0) {
      if (!designElements.technologyStack.tools) {
        designElements.technologyStack.tools = [];
      }
      designElements.technologyStack.tools.push(...testTechStack);
    }
  }

  /**
   * Extract general design elements from any content
   */
  private extractGeneralDesignElements(artifact: ParsedArtifact, designElements: DesignElements): void {
    const content = artifact.content;
    
    // Extract overview elements
    const problemStatement = this.extractProblemStatement(content);
    const solutionApproach = this.extractSolutionApproach(content);
    const keyPrinciples = this.extractKeyPrinciples(content);
    const keyInnovations = this.extractKeyInnovations(content);

    if (problemStatement) designElements.overview.problemStatement = problemStatement;
    if (solutionApproach) designElements.overview.solutionApproach = solutionApproach;
    if (keyPrinciples.length > 0) designElements.overview.keyPrinciples = keyPrinciples;
    if (keyInnovations.length > 0) designElements.overview.keyInnovations = keyInnovations;

    // Extract technology mentions
    const techStack = this.extractTechnologyFromText(content);
    Object.assign(designElements.technologyStack, techStack);
  }

  // ===== DESIGN ELEMENT EXTRACTION HELPER METHODS =====

  /**
   * Extract diagram title from Mermaid content
   */
  private extractDiagramTitle(content: string): string | null {
    // Look for title in various formats
    const titlePatterns = [
      /title\s+([^\n]+)/i,
      /^#\s*([^\n]+)/m,
      /%%\s*title:\s*([^\n]+)/i
    ];

    for (const pattern of titlePatterns) {
      const match = pattern.exec(content);
      if (match && match[1]) {
        return match[1].trim();
      }
    }

    return null;
  }

  /**
   * Extract diagram description from Mermaid content
   */
  private extractDiagramDescription(content: string): string | null {
    // Look for description comments
    const descPatterns = [
      /%%\s*description:\s*([^\n]+)/i,
      /%%\s*([^%\n]+)/
    ];

    for (const pattern of descPatterns) {
      const match = pattern.exec(content);
      if (match && match[1] && !match[1].includes('title')) {
        return match[1].trim();
      }
    }

    return null;
  }

  /**
   * Extract components from Mermaid for design document
   */
  private extractComponentsFromMermaidForDesign(content: string): Array<{
    name: string;
    purpose: string;
    subcomponents?: Array<{ name: string; description: string }>;
  }> {
    const components: Array<{
      name: string;
      purpose: string;
      subcomponents?: Array<{ name: string; description: string }>;
    }> = [];

    // Extract subgraph components (represent major system components)
    const subgraphPattern = /subgraph\s+([^"\n]+)(?:\s*\[([^\]]+)\])?\s*\n([\s\S]*?)(?=\n\s*end|\n\s*subgraph|\n\s*```|$)/gi;
    const subgraphMatches = content.matchAll(subgraphPattern);

    for (const match of subgraphMatches) {
      const [, name, label, subgraphContent] = match;
      const componentName = (label || name).replace(/["\[\]]/g, '').trim();
      
      // Extract nodes within this subgraph as subcomponents
      const subcomponents: Array<{ name: string; description: string }> = [];
      const nodePattern = /(\w+)\[([^\]]+)\]/g;
      const nodeMatches = subgraphContent.matchAll(nodePattern);
      
      for (const nodeMatch of nodeMatches) {
        const [, nodeId, nodeLabel] = nodeMatch;
        subcomponents.push({
          name: nodeLabel.trim(),
          description: `${nodeLabel.trim()} component`
        });
      }

      components.push({
        name: componentName,
        purpose: `${componentName} system component`,
        subcomponents: subcomponents.length > 0 ? subcomponents : undefined
      });
    }

    // If no subgraphs, extract individual nodes as components
    if (components.length === 0) {
      const nodePattern = /(\w+)\[([^\]]+)\]/g;
      const nodeMatches = content.matchAll(nodePattern);
      
      for (const match of nodeMatches) {
        const [, nodeId, nodeLabel] = match;
        components.push({
          name: nodeLabel.trim(),
          purpose: `${nodeLabel.trim()} component`
        });
      }
    }

    return components;
  }

  /**
   * Extract data flow phases from flowchart content
   */
  private extractDataFlowPhases(content: string): Array<{
    name: string;
    description: string;
    steps: string[];
  }> {
    const phases: Array<{
      name: string;
      description: string;
      steps: string[];
    }> = [];

    // Look for numbered or sequential flow patterns
    const flowPattern = /(\w+)\s*-->\s*(\w+)(?:\s*:\s*([^|\n]+))?/g;
    const flows = Array.from(content.matchAll(flowPattern));

    if (flows.length > 0) {
      // Group flows into phases
      const stepMap = new Map<string, string[]>();
      
      for (const [, from, to, label] of flows) {
        const fromLabel = this.getNodeLabel(content, from) || from;
        const toLabel = this.getNodeLabel(content, to) || to;
        const stepDescription = label || `${fromLabel} → ${toLabel}`;
        
        if (!stepMap.has(fromLabel)) {
          stepMap.set(fromLabel, []);
        }
        stepMap.get(fromLabel)!.push(stepDescription);
      }

      // Convert to phases
      let phaseIndex = 1;
      for (const [phaseName, steps] of stepMap) {
        phases.push({
          name: `Phase ${phaseIndex}: ${phaseName}`,
          description: `${phaseName} processing phase`,
          steps
        });
        phaseIndex++;
      }
    }

    return phases;
  }

  /**
   * Extract sequence flows from sequence diagrams
   */
  private extractSequenceFlows(content: string): Array<{
    name: string;
    description: string;
    actors: string[];
  }> {
    const sequences: Array<{
      name: string;
      description: string;
      actors: string[];
    }> = [];

    // Extract participants
    const participantPattern = /participant\s+(\w+)(?:\s+as\s+([^\n]+))?/g;
    const participants = Array.from(content.matchAll(participantPattern));
    
    if (participants.length > 0) {
      const actors = participants.map(([, id, label]) => label || id);
      
      sequences.push({
        name: 'Main Sequence Flow',
        description: 'Primary interaction sequence between system components',
        actors
      });
    }

    return sequences;
  }

  /**
   * Get node label from Mermaid content
   */
  private getNodeLabel(content: string, nodeId: string): string | null {
    const labelPattern = new RegExp(`${nodeId}\\[([^\\]]+)\\]`, 'g');
    const match = labelPattern.exec(content);
    return match ? match[1].trim() : null;
  }

  /**
   * Extract technology stack from text content
   */
  private extractTechnologyFromText(content: string): Partial<{
    languages: string[];
    frameworks: string[];
    databases: string[];
    infrastructure: string[];
    tools: string[];
  }> {
    const techStack: any = {};

    // Language patterns
    const languages = this.extractByPatterns(content, [
      /\b(Python|JavaScript|TypeScript|Java|C\+\+|C#|Go|Rust|Ruby|PHP)\b/gi,
      /\b(Python\s+[\d.]+|Node\.js|\.NET)\b/gi
    ]);
    if (languages.length > 0) techStack.languages = [...new Set(languages)];

    // Framework patterns
    const frameworks = this.extractByPatterns(content, [
      /\b(React|Vue|Angular|Django|Flask|FastAPI|Express|Spring|Laravel)\b/gi,
      /\b(PyTorch|TensorFlow|Keras|Scikit-learn)\b/gi,
      /\b(gRPC|REST|GraphQL)\b/gi
    ]);
    if (frameworks.length > 0) techStack.frameworks = [...new Set(frameworks)];

    // Database patterns
    const databases = this.extractByPatterns(content, [
      /\b(PostgreSQL|MySQL|MongoDB|Redis|SQLite|Elasticsearch)\b/gi,
      /\b(SQL|NoSQL)\b/gi
    ]);
    if (databases.length > 0) techStack.databases = [...new Set(databases)];

    // Infrastructure patterns
    const infrastructure = this.extractByPatterns(content, [
      /\b(Docker|Kubernetes|AWS|Azure|GCP|Terraform)\b/gi,
      /\b(Nginx|Apache|Load\s+Balancer)\b/gi
    ]);
    if (infrastructure.length > 0) techStack.infrastructure = [...new Set(infrastructure)];

    // Tool patterns
    const tools = this.extractByPatterns(content, [
      /\b(Git|Jenkins|GitHub\s+Actions|GitLab\s+CI|CircleCI)\b/gi,
      /\b(Jest|Pytest|Vitest|Mocha|JUnit)\b/gi,
      /\b(ESLint|Prettier|Black|Flake8)\b/gi
    ]);
    if (tools.length > 0) techStack.tools = [...new Set(tools)];

    return techStack;
  }

  /**
   * Extract matches using multiple patterns
   */
  private extractByPatterns(content: string, patterns: RegExp[]): string[] {
    const matches: string[] = [];
    
    for (const pattern of patterns) {
      const patternMatches = Array.from(content.matchAll(pattern));
      for (const match of patternMatches) {
        matches.push(match[1] || match[0]);
      }
    }
    
    return matches;
  }

  /**
   * Extract design decisions from content
   */
  private extractDesignDecisions(content: string): Array<{
    decision: string;
    rationale: string;
    alternatives?: string[];
  }> {
    const decisions: Array<{
      decision: string;
      rationale: string;
      alternatives?: string[];
    }> = [];

    // Look for decision patterns - more specific patterns first
    const decisionPatterns = [
      // Pattern: "Decision: X Rationale: Y"
      /(?:decision):\s*([^.\n]+?)[\s\S]*?(?:rationale|because|reason):\s*([^.\n]+)/gi,
      // Pattern: "We chose X because Y"
      /we\s+(?:chose|selected|decided)\s+([^.\n]+?)\s+because\s+([^.\n]+)/gi,
      // Pattern: "Selected X for Y"
      /(?:selected|chose)\s+([^.\n]+?)\s+for\s+([^.\n]+)/gi,
      // Pattern: "Design Decision: X because Y"
      /design\s+decision:\s*([^.\n]+?)\s+because\s+([^.\n]+)/gi
    ];

    for (const pattern of decisionPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const [, decision, rationale] = match;
        if (decision && rationale && decision.trim().length > 1 && rationale.trim().length > 1) {
          decisions.push({
            decision: decision.trim(),
            rationale: rationale.trim()
          });
        }
      }
    }

    return decisions;
  }

  /**
   * Extract constraints from content
   */
  private extractConstraints(content: string): string[] {
    const constraints: string[] = [];
    
    const constraintPatterns = [
      /constraint:\s*([^.\n]+)/gi,
      /limitation:\s*([^.\n]+)/gi,
      /must\s+(?:not\s+)?([^.\n]+)/gi,
      /cannot\s+([^.\n]+)/gi
    ];

    for (const pattern of constraintPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const constraint = match[1]?.trim();
        if (constraint && constraint.length > 5) {
          constraints.push(constraint);
        }
      }
    }

    return [...new Set(constraints)];
  }

  /**
   * Extract assumptions from content
   */
  private extractAssumptions(content: string): string[] {
    const assumptions: string[] = [];
    
    const assumptionPatterns = [
      /assumption:\s*([^.\n]+)/gi,
      /assume\s+(?:that\s+)?([^.\n]+)/gi,
      /assuming\s+([^.\n]+)/gi
    ];

    for (const pattern of assumptionPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const assumption = match[1]?.trim();
        if (assumption && assumption.length > 5) {
          assumptions.push(assumption);
        }
      }
    }

    return [...new Set(assumptions)];
  }

  /**
   * Group implementation steps into phases
   */
  private groupImplementationPhases(steps: ImplementationStep[]): Array<{
    name: string;
    description: string;
    steps: string[];
  }> {
    const phaseGroups = this.groupStepsByPhase(steps);
    const phases: Array<{
      name: string;
      description: string;
      steps: string[];
    }> = [];

    for (const [phaseName, phaseSteps] of Object.entries(phaseGroups)) {
      phases.push({
        name: phaseName,
        description: `${phaseName} implementation phase`,
        steps: phaseSteps.map(step => step.title)
      });
    }

    return phases;
  }

  /**
   * Extract quality constraints from test strategy
   */
  private extractQualityConstraints(content: string): string[] {
    const constraints: string[] = [];
    
    const qualityPatterns = [
      /performance:\s*([^.\n]+)/gi,
      /latency:\s*([^.\n]+)/gi,
      /throughput:\s*([^.\n]+)/gi,
      /memory:\s*([^.\n]+)/gi,
      /accuracy:\s*([^.\n]+)/gi,
      /coverage:\s*([^.\n]+)/gi
    ];

    for (const pattern of qualityPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const constraint = match[1]?.trim();
        if (constraint) {
          constraints.push(`Quality constraint: ${constraint}`);
        }
      }
    }

    return constraints;
  }

  /**
   * Extract testing technology from test strategy
   */
  private extractTestingTechnology(content: string): string[] {
    const testTech: string[] = [];
    
    const testPatterns = [
      /\b(Jest|Pytest|Vitest|Mocha|JUnit|TestNG)\b/gi,
      /\b(Hypothesis|QuickCheck|fast-check|Property-based)\b/gi,
      /\b(Selenium|Cypress|Playwright|Puppeteer)\b/gi,
      /\b(JMeter|Artillery|k6|LoadRunner)\b/gi
    ];

    for (const pattern of testPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        testTech.push(match[0]);
      }
    }

    return [...new Set(testTech)];
  }

  /**
   * Extract problem statement from content
   */
  private extractProblemStatement(content: string): string | null {
    const problemPatterns = [
      /problem\s+statement:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi,
      /problem:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi,
      /challenge:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi,
      /issue:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi
    ];

    for (const pattern of problemPatterns) {
      const match = pattern.exec(content);
      if (match && match[1]) {
        return match[1].trim();
      }
    }

    return null;
  }

  /**
   * Extract solution approach from content
   */
  private extractSolutionApproach(content: string): string | null {
    const solutionPatterns = [
      /solution\s+approach:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi,
      /solution:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi,
      /approach:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi,
      /methodology:\s*([^.\n]+(?:\.[^.\n]+)*\.?)/gi
    ];

    for (const pattern of solutionPatterns) {
      const match = pattern.exec(content);
      if (match && match[1]) {
        return match[1].trim();
      }
    }

    return null;
  }

  /**
   * Extract key principles from content
   */
  private extractKeyPrinciples(content: string): string[] {
    const principles: string[] = [];
    
    const principlePatterns = [
      /(?:key\s+)?principles?:\s*([^.\n]+)/gi,
      /design\s+principles?:\s*([^.\n]+)/gi,
      /guiding\s+principles?:\s*([^.\n]+)/gi
    ];

    for (const pattern of principlePatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const principle = match[1]?.trim();
        if (principle) {
          // Split on common delimiters
          const splitPrinciples = principle.split(/[,;]/).map(p => p.trim()).filter(p => p.length > 0);
          principles.push(...splitPrinciples);
        }
      }
    }

    return [...new Set(principles)];
  }

  /**
   * Extract key innovations from content
   */
  private extractKeyInnovations(content: string): string[] {
    const innovations: string[] = [];
    
    const innovationPatterns = [
      /(?:key\s+)?innovations?:\s*([^.\n]+)/gi,
      /breakthrough:\s*([^.\n]+)/gi,
      /novel\s+approach:\s*([^.\n]+)/gi,
      /innovative:\s*([^.\n]+)/gi
    ];

    for (const pattern of innovationPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const innovation = match[1]?.trim();
        if (innovation) {
          innovations.push(innovation);
        }
      }
    }

    return [...new Set(innovations)];
  }

  // ===== DESIGN DOCUMENT GENERATION =====

  /**
   * Generate the complete design document
   */
  private generateDesignDocument(
    projectName: string,
    designElements: DesignElements,
    options: DesignGenerationOptions
  ): string {
    const sections: string[] = [];

    // Title
    sections.push(`# Design Document: ${projectName}`);
    sections.push('');

    // Overview section
    sections.push('## Overview');
    sections.push('');
    
    if (designElements.overview.problemStatement) {
      sections.push('### Problem Statement');
      sections.push('');
      sections.push(designElements.overview.problemStatement);
      sections.push('');
    }

    if (designElements.overview.solutionApproach) {
      sections.push('### Solution Approach');
      sections.push('');
      sections.push(designElements.overview.solutionApproach);
      sections.push('');
    }

    if (designElements.overview.keyInnovations && designElements.overview.keyInnovations.length > 0) {
      sections.push('### Key Innovations');
      sections.push('');
      designElements.overview.keyInnovations.forEach(innovation => {
        sections.push(`- ${innovation}`);
      });
      sections.push('');
    }

    if (designElements.overview.keyPrinciples && designElements.overview.keyPrinciples.length > 0) {
      sections.push('### Key Design Principles');
      sections.push('');
      designElements.overview.keyPrinciples.forEach(principle => {
        sections.push(`- ${principle}`);
      });
      sections.push('');
    }

    // Architecture section
    if (options.includeArchitectureDiagrams && designElements.architecture.diagrams.length > 0) {
      sections.push('## Architecture');
      sections.push('');

      // Add architecture diagrams
      designElements.architecture.diagrams.forEach(diagram => {
        sections.push(`### ${diagram.title}`);
        sections.push('');
        
        if (diagram.description) {
          sections.push(diagram.description);
          sections.push('');
        }

        sections.push('```mermaid');
        sections.push(diagram.content);
        sections.push('```');
        sections.push('');
      });
    }

    // Component Details section
    if (options.includeComponentDetails && designElements.architecture.components.length > 0) {
      sections.push('## Component Design');
      sections.push('');

      designElements.architecture.components.forEach(component => {
        sections.push(`### ${component.name}`);
        sections.push('');
        sections.push(`**Purpose:** ${component.purpose}`);
        sections.push('');

        if (component.subcomponents && component.subcomponents.length > 0) {
          sections.push('**Subcomponents:**');
          component.subcomponents.forEach(sub => {
            sections.push(`- **${sub.name}**: ${sub.description}`);
          });
          sections.push('');
        }

        if (component.technologyStack && component.technologyStack.length > 0) {
          sections.push('**Technology Stack:**');
          component.technologyStack.forEach(tech => {
            sections.push(`- ${tech}`);
          });
          sections.push('');
        }

        if (component.interfaces && component.interfaces.length > 0) {
          sections.push('**Interfaces:**');
          component.interfaces.forEach(iface => {
            sections.push(`- ${iface}`);
          });
          sections.push('');
        }
      });
    }

    // Data Flow section
    if (options.includeDataFlow && (designElements.dataFlow.phases || designElements.dataFlow.sequences)) {
      sections.push('## Data Flow');
      sections.push('');

      if (designElements.dataFlow.phases && designElements.dataFlow.phases.length > 0) {
        sections.push('### Processing Phases');
        sections.push('');

        designElements.dataFlow.phases.forEach(phase => {
          sections.push(`#### ${phase.name}`);
          sections.push('');
          sections.push(phase.description);
          sections.push('');
          
          if (phase.steps.length > 0) {
            sections.push('**Steps:**');
            phase.steps.forEach((step, index) => {
              sections.push(`${index + 1}. ${step}`);
            });
            sections.push('');
          }
        });
      }

      if (designElements.dataFlow.sequences && designElements.dataFlow.sequences.length > 0) {
        sections.push('### Sequence Flows');
        sections.push('');

        designElements.dataFlow.sequences.forEach(sequence => {
          sections.push(`#### ${sequence.name}`);
          sections.push('');
          sections.push(sequence.description);
          sections.push('');
          
          if (sequence.actors.length > 0) {
            sections.push('**Actors:**');
            sequence.actors.forEach(actor => {
              sections.push(`- ${actor}`);
            });
            sections.push('');
          }
        });
      }
    }

    // API Design section
    if (designElements.apiDesign.endpoints || designElements.apiDesign.dataModels) {
      sections.push('## API Design');
      sections.push('');

      if (designElements.apiDesign.endpoints && designElements.apiDesign.endpoints.length > 0) {
        sections.push('### Endpoints');
        sections.push('');

        sections.push('| Method | Path | Description |');
        sections.push('|--------|------|-------------|');
        
        designElements.apiDesign.endpoints.forEach(endpoint => {
          sections.push(`| ${endpoint.method} | ${endpoint.path} | ${endpoint.description} |`);
        });
        sections.push('');
      }

      if (designElements.apiDesign.dataModels && designElements.apiDesign.dataModels.length > 0) {
        sections.push('### Data Models');
        sections.push('');

        designElements.apiDesign.dataModels.forEach(model => {
          sections.push(`#### ${model.name}`);
          sections.push('');
          
          if (model.properties.length > 0) {
            sections.push('| Property | Type | Description |');
            sections.push('|----------|------|-------------|');
            
            model.properties.forEach(prop => {
              sections.push(`| ${prop.name} | ${prop.type} | ${prop.description || '-'} |`);
            });
            sections.push('');
          }
        });
      }
    }

    // Technology Stack section
    if (options.includeTechnologyStack && Object.keys(designElements.technologyStack).length > 0) {
      sections.push('## Technology Stack');
      sections.push('');

      const techStack = designElements.technologyStack;
      
      if (techStack.languages && techStack.languages.length > 0) {
        sections.push('### Languages');
        techStack.languages.forEach(lang => {
          sections.push(`- ${lang}`);
        });
        sections.push('');
      }

      if (techStack.frameworks && techStack.frameworks.length > 0) {
        sections.push('### Frameworks');
        techStack.frameworks.forEach(framework => {
          sections.push(`- ${framework}`);
        });
        sections.push('');
      }

      if (techStack.databases && techStack.databases.length > 0) {
        sections.push('### Databases');
        techStack.databases.forEach(db => {
          sections.push(`- ${db}`);
        });
        sections.push('');
      }

      if (techStack.infrastructure && techStack.infrastructure.length > 0) {
        sections.push('### Infrastructure');
        techStack.infrastructure.forEach(infra => {
          sections.push(`- ${infra}`);
        });
        sections.push('');
      }

      if (techStack.tools && techStack.tools.length > 0) {
        sections.push('### Tools');
        techStack.tools.forEach(tool => {
          sections.push(`- ${tool}`);
        });
        sections.push('');
      }
    }

    // Implementation Notes section
    if (options.includeImplementationNotes && Object.keys(designElements.implementationNotes).length > 0) {
      sections.push('## Implementation Notes');
      sections.push('');

      const implNotes = designElements.implementationNotes;

      if (implNotes.designDecisions && implNotes.designDecisions.length > 0) {
        sections.push('### Design Decisions');
        sections.push('');

        implNotes.designDecisions.forEach(decision => {
          sections.push(`**Decision:** ${decision.decision}`);
          sections.push('');
          sections.push(`**Rationale:** ${decision.rationale}`);
          sections.push('');
          
          if (decision.alternatives && decision.alternatives.length > 0) {
            sections.push('**Alternatives Considered:**');
            decision.alternatives.forEach(alt => {
              sections.push(`- ${alt}`);
            });
            sections.push('');
          }
        });
      }

      if (implNotes.constraints && implNotes.constraints.length > 0) {
        sections.push('### Constraints');
        sections.push('');
        implNotes.constraints.forEach(constraint => {
          sections.push(`- ${constraint}`);
        });
        sections.push('');
      }

      if (implNotes.assumptions && implNotes.assumptions.length > 0) {
        sections.push('### Assumptions');
        sections.push('');
        implNotes.assumptions.forEach(assumption => {
          sections.push(`- ${assumption}`);
        });
        sections.push('');
      }

      if (implNotes.risks && implNotes.risks.length > 0) {
        sections.push('### Risks and Mitigations');
        sections.push('');
        implNotes.risks.forEach(risk => {
          sections.push(`**Risk:** ${risk.risk}`);
          sections.push('');
          sections.push(`**Mitigation:** ${risk.mitigation}`);
          sections.push('');
        });
      }
    }

    return sections.join('\n');
  }

  /**
   * Validate design completeness
   */
  private validateDesignCompleteness(
    designElements: DesignElements,
    options: DesignGenerationOptions
  ): string[] {
    const errors: string[] = [];

    // Check required sections based on options
    if (options.includeArchitectureDiagrams && designElements.architecture.diagrams.length === 0) {
      errors.push('Architecture diagrams are enabled but no diagrams found in artifacts');
    }

    if (options.includeComponentDetails && designElements.architecture.components.length === 0) {
      errors.push('Component details are enabled but no components found in artifacts');
    }

    // Check for minimum content - only if no other content is available
    const hasAnyContent = designElements.architecture.diagrams.length > 0 ||
                         designElements.architecture.components.length > 0 ||
                         (designElements.apiDesign.endpoints && designElements.apiDesign.endpoints.length > 0) ||
                         (designElements.technologyStack.languages && designElements.technologyStack.languages.length > 0) ||
                         (designElements.implementationNotes.designDecisions && designElements.implementationNotes.designDecisions.length > 0);

    if (!hasAnyContent && !designElements.overview.problemStatement && !designElements.overview.solutionApproach) {
      errors.push('Design document lacks overview content (problem statement or solution approach)');
    }

    // Check architecture completeness - only for non-trivial diagrams
    if (designElements.architecture.diagrams.length > 0) {
      for (const diagram of designElements.architecture.diagrams) {
        if (!diagram.content || diagram.content.trim().length < 20) {
          errors.push(`Diagram "${diagram.title}" has insufficient content`);
        }
      }
    }

    // Check API design completeness
    if (designElements.apiDesign.endpoints && designElements.apiDesign.endpoints.length > 0) {
      for (const endpoint of designElements.apiDesign.endpoints) {
        if (!endpoint.description || endpoint.description.length < 5) {
          errors.push(`API endpoint ${endpoint.method} ${endpoint.path} lacks proper description`);
        }
      }
    }

    return errors;
  }

  /**
   * Generate tasks.md from Opus implementation plans with task hierarchy and dependencies
   * 
   * Requirements: 16.3, 16.4
   */
  async generateTasks(
    artifacts: ParsedArtifact[],
    options: TasksGenerationOptions = {}
  ): Promise<TasksDocument> {
    const {
      projectName = 'Generated Project',
      includeRequirementsReferences = true,
      includeDependencies = true,
      includeComplexityEstimates = true,
      taskIdPrefix = '',
      groupByPhase = true
    } = options;

    // Extract tasks from various artifact types
    const taskHierarchy = this.extractTasksFromArtifacts(artifacts, options);
    
    // Generate tasks document content
    const content = this.generateTasksDocument(projectName, taskHierarchy, options);
    
    // Validate task completeness and dependencies
    const validationErrors = this.validateTasksCompleteness(taskHierarchy);
    
    // Count total tasks (including subtasks)
    const taskCount = this.countTotalTasks(taskHierarchy);
    
    return {
      title: `${projectName} Implementation Plan`,
      content,
      metadata: {
        generatedAt: new Date(),
        sourceArtifacts: artifacts.map(a => a.id),
        taskCount,
        validationErrors
      }
    };
  }

  /**
   * Extract tasks from various Opus artifact types
   */
  private extractTasksFromArtifacts(
    artifacts: ParsedArtifact[],
    options: TasksGenerationOptions
  ): TaskHierarchy[] {
    const tasks: TaskHierarchy[] = [];
    let taskCounter = 1;

    for (const artifact of artifacts) {
      switch (artifact.type) {
        case 'implementation_guide':
          tasks.push(...this.extractTasksFromImplementationGuide(artifact, taskCounter, options));
          taskCounter += this.countTasksInImplementationGuide(artifact);
          break;
        case 'openapi_spec':
          tasks.push(...this.extractTasksFromAPISpec(artifact, taskCounter, options));
          taskCounter += this.countTasksInAPISpec(artifact);
          break;
        case 'mermaid_diagram':
          tasks.push(...this.extractTasksFromArchitecture(artifact, taskCounter, options));
          taskCounter += this.countTasksInArchitecture(artifact);
          break;
        case 'test_strategy':
          tasks.push(...this.extractTasksFromTestStrategy(artifact, taskCounter, options));
          taskCounter += this.countTasksInTestStrategy(artifact);
          break;
        default:
          // Try to extract from general content
          tasks.push(...this.extractTasksFromContent(artifact, taskCounter, options));
          taskCounter += this.countTasksInContent(artifact);
      }
    }

    // Group by phase if requested
    if (options.groupByPhase) {
      return this.groupTasksByPhase(tasks);
    }

    return tasks;
  }

  /**
   * Extract tasks from implementation guide artifacts
   */
  private extractTasksFromImplementationGuide(
    artifact: ParsedArtifact,
    startCounter: number,
    options: TasksGenerationOptions
  ): TaskHierarchy[] {
    const tasks: TaskHierarchy[] = [];
    
    if (artifact.structured?.implementationSteps) {
      const steps = artifact.structured.implementationSteps;
      
      // Group steps by phase
      const phaseGroups = this.groupStepsByPhase(steps);
      let taskId = startCounter;
      
      for (const [phaseName, phaseSteps] of Object.entries(phaseGroups)) {
        const phaseTask: TaskHierarchy = {
          id: this.generateTaskId(taskId, options.taskIdPrefix),
          title: `${phaseName} Phase`,
          description: `Implement ${phaseName.toLowerCase()} functionality`,
          status: 'not_started',
          phase: phaseName,
          subtasks: []
        };

        // Add subtasks for each step in the phase
        let subtaskCounter = 1;
        for (const step of phaseSteps) {
          const subtask: TaskHierarchy = {
            id: `${phaseTask.id}.${subtaskCounter}`,
            title: step.title,
            description: step.description,
            status: 'not_started',
            complexity: step.complexity,
            estimate: step.estimate,
            dependencies: step.dependencies.map(dep => this.mapDependencyToTaskId(dep, steps, options.taskIdPrefix)),
            requirementReferences: this.extractRequirementReferences(step.description)
          };
          
          phaseTask.subtasks!.push(subtask);
          subtaskCounter++;
        }

        tasks.push(phaseTask);
        taskId++;
      }
    }

    return tasks;
  }

  /**
   * Extract tasks from OpenAPI specification artifacts
   */
  private extractTasksFromAPISpec(
    artifact: ParsedArtifact,
    startCounter: number,
    options: TasksGenerationOptions
  ): TaskHierarchy[] {
    const tasks: TaskHierarchy[] = [];
    
    try {
      const apiSpec = artifact.structured?.openapi || JSON.parse(artifact.content);
      let taskId = startCounter;
      
      if (apiSpec.paths) {
        // Create main API implementation task
        const apiTask: TaskHierarchy = {
          id: this.generateTaskId(taskId, options.taskIdPrefix),
          title: 'API Implementation',
          description: 'Implement REST API endpoints and data models',
          status: 'not_started',
          phase: 'API Development',
          subtasks: []
        };

        // Add subtasks for endpoint groups
        const endpointGroups = this.groupEndpointsByResource(Object.keys(apiSpec.paths));
        let subtaskCounter = 1;
        
        for (const [resource, endpoints] of Object.entries(endpointGroups)) {
          const resourceTask: TaskHierarchy = {
            id: `${apiTask.id}.${subtaskCounter}`,
            title: `Implement ${resource} endpoints`,
            description: `Create REST endpoints for ${resource} operations`,
            status: 'not_started',
            complexity: this.estimateEndpointComplexity(endpoints, apiSpec.paths),
            requirementReferences: options.includeRequirementsReferences ? 
              this.extractAPIRequirementReferences(resource) : undefined
          };
          
          apiTask.subtasks!.push(resourceTask);
          subtaskCounter++;
        }

        // Add data models task if schemas exist
        if (apiSpec.components?.schemas) {
          const modelsTask: TaskHierarchy = {
            id: `${apiTask.id}.${subtaskCounter}`,
            title: 'Implement data models',
            description: 'Create data models and validation schemas',
            status: 'not_started',
            complexity: 'moderate' as ComplexityLevel,
            requirementReferences: options.includeRequirementsReferences ? 
              ['Data model requirements'] : undefined
          };
          
          apiTask.subtasks!.push(modelsTask);
        }

        tasks.push(apiTask);
      }
    } catch (error) {
      // If parsing fails, create a generic API task
      tasks.push({
        id: this.generateTaskId(startCounter, options.taskIdPrefix),
        title: 'API Implementation',
        description: 'Implement API based on specification',
        status: 'not_started',
        complexity: 'moderate' as ComplexityLevel
      });
    }

    return tasks;
  }

  /**
   * Extract tasks from Mermaid architecture diagrams
   */
  private extractTasksFromArchitecture(
    artifact: ParsedArtifact,
    startCounter: number,
    options: TasksGenerationOptions
  ): TaskHierarchy[] {
    const tasks: TaskHierarchy[] = [];
    
    // Extract components from diagram
    const components = this.extractComponentNamesFromMermaid(artifact.content);
    
    if (components.length > 0) {
      let taskId = startCounter;
      
      // Create main architecture task
      const archTask: TaskHierarchy = {
        id: this.generateTaskId(taskId, options.taskIdPrefix),
        title: 'System Architecture Implementation',
        description: 'Implement system components and their interactions',
        status: 'not_started',
        phase: 'Architecture',
        subtasks: []
      };

      // Add subtasks for each component
      let subtaskCounter = 1;
      for (const component of components) {
        const componentTask: TaskHierarchy = {
          id: `${archTask.id}.${subtaskCounter}`,
          title: `Implement ${component} component`,
          description: `Create and configure ${component} component with required interfaces`,
          status: 'not_started',
          complexity: this.estimateComponentComplexity(component, artifact.content),
          requirementReferences: options.includeRequirementsReferences ? 
            [`${component} component requirements`] : undefined
        };
        
        archTask.subtasks!.push(componentTask);
        subtaskCounter++;
      }

      // Add integration task if multiple components
      if (components.length > 1) {
        const integrationTask: TaskHierarchy = {
          id: `${archTask.id}.${subtaskCounter}`,
          title: 'Component Integration',
          description: 'Integrate components and establish communication patterns',
          status: 'not_started',
          complexity: 'complex' as ComplexityLevel,
          dependencies: archTask.subtasks!.slice(0, -1).map(t => t.id),
          requirementReferences: options.includeRequirementsReferences ? 
            ['Integration requirements'] : undefined
        };
        
        archTask.subtasks!.push(integrationTask);
      }

      tasks.push(archTask);
    }

    return tasks;
  }

  /**
   * Extract tasks from test strategy artifacts
   */
  private extractTasksFromTestStrategy(
    artifact: ParsedArtifact,
    startCounter: number,
    options: TasksGenerationOptions
  ): TaskHierarchy[] {
    const tasks: TaskHierarchy[] = [];
    
    // Extract test categories
    const testCategories = this.extractTestCategoriesFromContent(artifact.content);
    
    if (testCategories.length > 0) {
      let taskId = startCounter;
      
      // Create main testing task
      const testTask: TaskHierarchy = {
        id: this.generateTaskId(taskId, options.taskIdPrefix),
        title: 'Testing Implementation',
        description: 'Implement comprehensive test suite with coverage targets',
        status: 'not_started',
        phase: 'Testing',
        subtasks: []
      };

      // Add subtasks for each test category
      let subtaskCounter = 1;
      for (const category of testCategories) {
        const categoryTask: TaskHierarchy = {
          id: `${testTask.id}.${subtaskCounter}`,
          title: `Implement ${category.type} tests`,
          description: `Create ${category.type} tests with ${category.coverage}% coverage for ${category.scope}`,
          status: 'not_started',
          complexity: this.estimateTestComplexity(category.type),
          requirementReferences: options.includeRequirementsReferences ? 
            [`${category.type} testing requirements`] : undefined
        };
        
        testTask.subtasks!.push(categoryTask);
        subtaskCounter++;
      }

      // Add property-based testing task if mentioned
      const hasPropertyTesting = artifact.content.toLowerCase().includes('property') || 
                                artifact.content.toLowerCase().includes('invariant');
      
      if (hasPropertyTesting) {
        const propertyTask: TaskHierarchy = {
          id: `${testTask.id}.${subtaskCounter}`,
          title: 'Implement property-based tests',
          description: 'Create property-based tests for system invariants and properties',
          status: 'not_started',
          complexity: 'complex' as ComplexityLevel,
          requirementReferences: options.includeRequirementsReferences ? 
            ['Property-based testing requirements'] : undefined
        };
        
        testTask.subtasks!.push(propertyTask);
      }

      tasks.push(testTask);
    }

    return tasks;
  }

  /**
   * Extract tasks from general content
   */
  private extractTasksFromContent(
    artifact: ParsedArtifact,
    startCounter: number,
    options: TasksGenerationOptions
  ): TaskHierarchy[] {
    const tasks: TaskHierarchy[] = [];
    
    // Look for task-like patterns in content
    const taskPatterns = [
      /(?:implement|create|build|develop|setup|configure)\s+([^.\n]+)/gi,
      /(?:task|step|action):\s*([^.\n]+)/gi,
      /(?:\d+\.)\s*([^.\n]+)/g
    ];

    const foundTasks = new Set<string>();
    
    for (const pattern of taskPatterns) {
      const matches = artifact.content.matchAll(pattern);
      for (const match of matches) {
        const taskText = match[1]?.trim();
        if (taskText && taskText.length > 10 && !foundTasks.has(taskText)) {
          foundTasks.add(taskText);
        }
      }
    }

    if (foundTasks.size > 0) {
      let taskId = startCounter;
      
      // Create main implementation task
      const mainTask: TaskHierarchy = {
        id: this.generateTaskId(taskId, options.taskIdPrefix),
        title: 'General Implementation',
        description: 'Implement functionality based on artifact content',
        status: 'not_started',
        subtasks: []
      };

      // Add subtasks for each found task
      let subtaskCounter = 1;
      for (const taskText of foundTasks) {
        const subtask: TaskHierarchy = {
          id: `${mainTask.id}.${subtaskCounter}`,
          title: this.cleanTaskTitle(taskText),
          description: `Implement: ${taskText}`,
          status: 'not_started',
          complexity: 'moderate' as ComplexityLevel
        };
        
        mainTask.subtasks!.push(subtask);
        subtaskCounter++;
      }

      tasks.push(mainTask);
    }

    return tasks;
  }

  /**
   * Group tasks by phase
   */
  private groupTasksByPhase(tasks: TaskHierarchy[]): TaskHierarchy[] {
    const phaseGroups: Record<string, TaskHierarchy[]> = {};
    const ungroupedTasks: TaskHierarchy[] = [];
    
    for (const task of tasks) {
      if (task.phase) {
        if (!phaseGroups[task.phase]) {
          phaseGroups[task.phase] = [];
        }
        phaseGroups[task.phase].push(task);
      } else {
        ungroupedTasks.push(task);
      }
    }

    const groupedTasks: TaskHierarchy[] = [];
    
    // Add phase groups
    for (const [phaseName, phaseTasks] of Object.entries(phaseGroups)) {
      if (phaseTasks.length === 1) {
        // Single task in phase, just add it
        groupedTasks.push(phaseTasks[0]);
      } else {
        // Multiple tasks, create phase container
        const phaseContainer: TaskHierarchy = {
          id: `phase-${phaseName.toLowerCase().replace(/\s+/g, '-')}`,
          title: `${phaseName} Phase`,
          description: `Complete all ${phaseName.toLowerCase()} tasks`,
          status: 'not_started',
          phase: phaseName,
          subtasks: phaseTasks
        };
        groupedTasks.push(phaseContainer);
      }
    }
    
    // Add ungrouped tasks
    groupedTasks.push(...ungroupedTasks);
    
    return groupedTasks;
  }

  /**
   * Generate the complete tasks document
   */
  private generateTasksDocument(
    projectName: string,
    taskHierarchy: TaskHierarchy[],
    options: TasksGenerationOptions
  ): string {
    const sections = [
      `# Implementation Plan: ${projectName}`,
      '',
      '## Overview',
      '',
      `This implementation plan breaks down ${projectName} into discrete coding tasks. The tasks are organized hierarchically with dependencies and complexity estimates to guide implementation.`,
      '',
      'The implementation follows a structured approach with clear phases, dependencies, and requirements traceability.',
      '',
      '## Tasks',
      ''
    ];

    // Generate task list
    for (const task of taskHierarchy) {
      this.addTaskToDocument(task, sections, 0, options);
    }

    return sections.join('\n');
  }

  /**
   * Add a task and its subtasks to the document
   */
  private addTaskToDocument(
    task: TaskHierarchy,
    sections: string[],
    depth: number,
    options: TasksGenerationOptions
  ): void {
    const indent = '  '.repeat(depth);
    const checkbox = task.status === 'completed' ? '[x]' : 
                    task.status === 'in_progress' ? '[-]' : '[ ]';
    
    // Main task line
    let taskLine = `${indent}- ${checkbox} ${task.id ? task.id + ' ' : ''}${task.title}`;
    sections.push(taskLine);
    
    // Add description if present
    if (task.description) {
      sections.push(`${indent}  - ${task.description}`);
    }
    
    // Add complexity estimate if enabled and present
    if (options.includeComplexityEstimates && task.complexity) {
      sections.push(`${indent}  - _Complexity: ${task.complexity}_`);
    }
    
    // Add estimate if present
    if (task.estimate) {
      sections.push(`${indent}  - _Estimate: ${task.estimate}_`);
    }
    
    // Add dependencies if enabled and present
    if (options.includeDependencies && task.dependencies && task.dependencies.length > 0) {
      sections.push(`${indent}  - _Dependencies: ${task.dependencies.join(', ')}_`);
    }
    
    // Add requirement references if enabled and present
    if (options.includeRequirementsReferences && task.requirementReferences && task.requirementReferences.length > 0) {
      sections.push(`${indent}  - _Requirements: ${task.requirementReferences.join(', ')}_`);
    }
    
    sections.push('');
    
    // Add subtasks recursively
    if (task.subtasks && task.subtasks.length > 0) {
      for (const subtask of task.subtasks) {
        this.addTaskToDocument(subtask, sections, depth + 1, options);
      }
    }
  }

  // ===== HELPER METHODS FOR TASK GENERATION =====

  /**
   * Generate task ID with optional prefix
   */
  private generateTaskId(counter: number, prefix?: string): string {
    const id = counter.toString();
    return prefix ? `${prefix}-${id}` : id;
  }

  /**
   * Map dependency ID to task ID
   */
  private mapDependencyToTaskId(
    dependencyId: string,
    allSteps: ImplementationStep[],
    prefix?: string
  ): string {
    // Find the step index for the dependency
    const stepIndex = allSteps.findIndex(step => step.id === dependencyId);
    if (stepIndex >= 0) {
      return this.generateTaskId(stepIndex + 1, prefix);
    }
    return dependencyId; // Return as-is if not found
  }

  /**
   * Extract requirement references from text
   */
  private extractRequirementReferences(text: string): string[] {
    const references: string[] = [];
    
    // Look for requirement patterns
    const reqPatterns = [
      /(?:requirement|req)\s+(\d+(?:\.\d+)?)/gi,
      /(?:REQ|R)-(\d+(?:\.\d+)?)/gi
    ];

    for (const pattern of reqPatterns) {
      const matches = text.matchAll(pattern);
      for (const match of matches) {
        references.push(`Requirement ${match[1]}`);
      }
    }

    return [...new Set(references)];
  }

  /**
   * Extract API requirement references
   */
  private extractAPIRequirementReferences(resource: string): string[] {
    return [`${resource} API requirements`, 'API design requirements'];
  }

  /**
   * Estimate endpoint complexity based on methods and schemas
   */
  private estimateEndpointComplexity(endpoints: string[], pathSpecs: any): ComplexityLevel {
    let totalMethods = 0;
    let hasComplexSchemas = false;
    
    for (const endpoint of endpoints) {
      const spec = pathSpecs[endpoint];
      if (spec) {
        totalMethods += Object.keys(spec).length;
        
        // Check for complex request/response schemas
        for (const methodSpec of Object.values(spec) as any[]) {
          if (methodSpec.requestBody || methodSpec.responses) {
            hasComplexSchemas = true;
          }
        }
      }
    }
    
    if (totalMethods > 6 || hasComplexSchemas) return 'complex';
    if (totalMethods > 3) return 'moderate';
    return 'simple';
  }

  /**
   * Estimate component complexity based on connections
   */
  private estimateComponentComplexity(component: string, diagramContent: string): ComplexityLevel {
    // Count connections to/from this component
    const connectionPattern = new RegExp(`\\b${component}\\b.*?(?:-->|<--|<->)`, 'g');
    const connections = (diagramContent.match(connectionPattern) || []).length;
    
    if (connections > 4) return 'complex';
    if (connections > 2) return 'moderate';
    return 'simple';
  }

  /**
   * Estimate test complexity based on type
   */
  private estimateTestComplexity(testType: string): ComplexityLevel {
    const complexTypes = ['integration', 'e2e', 'property-based', 'performance'];
    const moderateTypes = ['unit', 'component', 'api'];
    
    if (complexTypes.some(type => testType.toLowerCase().includes(type))) {
      return 'complex';
    }
    if (moderateTypes.some(type => testType.toLowerCase().includes(type))) {
      return 'moderate';
    }
    return 'simple';
  }

  /**
   * Clean task title for better readability
   */
  private cleanTaskTitle(title: string): string {
    // Remove common prefixes and clean up
    let cleaned = title.trim();
    cleaned = cleaned.replace(/^(?:implement|create|build|develop|setup|configure)\s+/i, '');
    cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
    
    // Ensure it doesn't end with punctuation
    cleaned = cleaned.replace(/[.,:;]$/, '');
    
    return cleaned;
  }

  /**
   * Count tasks in implementation guide
   */
  private countTasksInImplementationGuide(artifact: ParsedArtifact): number {
    return artifact.structured?.implementationSteps?.length || 1;
  }

  /**
   * Count tasks in API spec
   */
  private countTasksInAPISpec(artifact: ParsedArtifact): number {
    try {
      const apiSpec = artifact.structured?.openapi || JSON.parse(artifact.content);
      const endpointCount = apiSpec.paths ? Object.keys(apiSpec.paths).length : 0;
      const schemaCount = apiSpec.components?.schemas ? Object.keys(apiSpec.components.schemas).length : 0;
      return Math.max(1, Math.ceil((endpointCount + schemaCount) / 3)); // Group endpoints
    } catch {
      return 1;
    }
  }

  /**
   * Count tasks in architecture diagram
   */
  private countTasksInArchitecture(artifact: ParsedArtifact): number {
    const components = this.extractComponentNamesFromMermaid(artifact.content);
    return Math.max(1, components.length);
  }

  /**
   * Count tasks in test strategy
   */
  private countTasksInTestStrategy(artifact: ParsedArtifact): number {
    const categories = this.extractTestCategoriesFromContent(artifact.content);
    return Math.max(1, categories.length);
  }

  /**
   * Count tasks in general content
   */
  private countTasksInContent(artifact: ParsedArtifact): number {
    // Estimate based on content length and task patterns
    const taskPatterns = artifact.content.match(/(?:implement|create|build|develop|setup|configure)\s+[^.\n]+/gi);
    return Math.max(1, taskPatterns?.length || 1);
  }

  /**
   * Count total tasks including subtasks
   */
  private countTotalTasks(taskHierarchy: TaskHierarchy[]): number {
    let count = 0;
    
    for (const task of taskHierarchy) {
      count++; // Count the task itself
      
      if (task.subtasks && task.subtasks.length > 0) {
        count += this.countTotalTasks(task.subtasks); // Recursively count subtasks
      }
    }
    
    return count;
  }

  /**
   * Validate tasks completeness and dependencies
   */
  private validateTasksCompleteness(taskHierarchy: TaskHierarchy[]): string[] {
    const errors: string[] = [];
    const allTaskIds = new Set<string>();
    
    // Collect all task IDs
    this.collectTaskIds(taskHierarchy, allTaskIds);
    
    // Validate each task
    for (const task of taskHierarchy) {
      this.validateTask(task, allTaskIds, errors);
    }
    
    return errors;
  }

  /**
   * Collect all task IDs recursively
   */
  private collectTaskIds(tasks: TaskHierarchy[], taskIds: Set<string>): void {
    for (const task of tasks) {
      if (task.id) {
        taskIds.add(task.id);
      }
      
      if (task.subtasks && task.subtasks.length > 0) {
        this.collectTaskIds(task.subtasks, taskIds);
      }
    }
  }

  /**
   * Validate individual task
   */
  private validateTask(task: TaskHierarchy, allTaskIds: Set<string>, errors: string[]): void {
    // Check for missing title
    if (!task.title || task.title.trim().length === 0) {
      errors.push(`Task ${task.id || 'unknown'} has no title`);
    }
    
    // Check for invalid dependencies
    if (task.dependencies && task.dependencies.length > 0) {
      for (const dep of task.dependencies) {
        if (!allTaskIds.has(dep)) {
          errors.push(`Task ${task.id} has invalid dependency: ${dep}`);
        }
      }
    }
    
    // Validate subtasks recursively
    if (task.subtasks && task.subtasks.length > 0) {
      for (const subtask of task.subtasks) {
        this.validateTask(subtask, allTaskIds, errors);
      }
    }
  }

  /**
   * Generate .config.kiro file with workflow type and spec metadata
   * 
   * Requirements: 16.5
   */
  async generateConfig(
    artifacts: ParsedArtifact[],
    options: ConfigGenerationOptions = {}
  ): Promise<ConfigKiro> {
    const {
      specId = this.generateSpecId(),
      workflowType = this.determineWorkflowType(artifacts),
      specType = this.determineSpecType(artifacts),
      projectName
    } = options;

    const config: ConfigKiro = {
      specId,
      workflowType,
      specType,
      metadata: {
        generatedAt: new Date(),
        sourceArtifacts: artifacts.map(a => a.id),
        projectName
      }
    };

    return config;
  }

  /**
   * Generate hybrid workflow spec package with mixed Opus and local content
   * 
   * Requirements: 16.6
   */
  async generateHybridWorkflow(
    opusArtifacts: ParsedArtifact[],
    localArtifacts: ParsedArtifact[],
    options: HybridWorkflowOptions
  ): Promise<HybridSpecPackage> {
    const {
      projectName = 'Hybrid Project',
      opusSource,
      localSource,
      includeSourceTracking = true,
      validateConsistency = true
    } = options;

    const hybridPackage: HybridSpecPackage = {
      config: await this.generateHybridConfig(opusArtifacts, localArtifacts, options),
      metadata: {
        generatedAt: new Date(),
        opusArtifacts: opusArtifacts.map(a => a.id),
        localArtifacts: localArtifacts.map(a => a.id),
        hybridType: `${opusSource}-opus-${localSource}-local`
      }
    };

    // Generate documents based on hybrid configuration
    switch (opusSource) {
      case 'design':
        // Opus provides design, local provides requirements/tasks
        hybridPackage.design = await this.generateDesign(opusArtifacts, {
          projectName,
          includeArchitectureDiagrams: true,
          includeSequenceDiagrams: true,
          includeComponentDetails: true,
          includeDataFlow: true,
          includeTechnologyStack: true,
          includeImplementationNotes: true
        });

        if (localSource === 'requirements') {
          hybridPackage.requirements = await this.generateRequirements(localArtifacts, {
            projectName,
            includePropertyBasedTesting: true,
            earsValidation: true
          });
          
          // Generate tasks from both sources
          const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
          hybridPackage.tasks = await this.generateTasks(combinedArtifacts, {
            projectName,
            includeRequirementsReferences: true,
            includeDependencies: true,
            includeComplexityEstimates: true,
            groupByPhase: true
          });
        } else if (localSource === 'tasks') {
          hybridPackage.tasks = await this.generateTasks(localArtifacts, {
            projectName,
            includeRequirementsReferences: true,
            includeDependencies: true,
            includeComplexityEstimates: true,
            groupByPhase: true
          });
          
          // Generate requirements from both sources
          const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
          hybridPackage.requirements = await this.generateRequirements(combinedArtifacts, {
            projectName,
            includePropertyBasedTesting: true,
            earsValidation: true
          });
        }
        break;

      case 'requirements':
        // Opus provides requirements, local provides design/tasks
        hybridPackage.requirements = await this.generateRequirements(opusArtifacts, {
          projectName,
          includePropertyBasedTesting: true,
          earsValidation: true
        });

        if (localSource === 'design') {
          hybridPackage.design = await this.generateDesign(localArtifacts, {
            projectName,
            includeArchitectureDiagrams: true,
            includeSequenceDiagrams: true,
            includeComponentDetails: true,
            includeDataFlow: true,
            includeTechnologyStack: true,
            includeImplementationNotes: true
          });
          
          // Generate tasks from both sources
          const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
          hybridPackage.tasks = await this.generateTasks(combinedArtifacts, {
            projectName,
            includeRequirementsReferences: true,
            includeDependencies: true,
            includeComplexityEstimates: true,
            groupByPhase: true
          });
        } else if (localSource === 'tasks') {
          hybridPackage.tasks = await this.generateTasks(localArtifacts, {
            projectName,
            includeRequirementsReferences: true,
            includeDependencies: true,
            includeComplexityEstimates: true,
            groupByPhase: true
          });
          
          // Generate design from both sources
          const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
          hybridPackage.design = await this.generateDesign(combinedArtifacts, {
            projectName,
            includeArchitectureDiagrams: true,
            includeSequenceDiagrams: true,
            includeComponentDetails: true,
            includeDataFlow: true,
            includeTechnologyStack: true,
            includeImplementationNotes: true
          });
        }
        break;

      case 'tasks':
        // Opus provides tasks, local provides requirements/design
        hybridPackage.tasks = await this.generateTasks(opusArtifacts, {
          projectName,
          includeRequirementsReferences: true,
          includeDependencies: true,
          includeComplexityEstimates: true,
          groupByPhase: true
        });

        if (localSource === 'requirements') {
          hybridPackage.requirements = await this.generateRequirements(localArtifacts, {
            projectName,
            includePropertyBasedTesting: true,
            earsValidation: true
          });
          
          // Generate design from both sources
          const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
          hybridPackage.design = await this.generateDesign(combinedArtifacts, {
            projectName,
            includeArchitectureDiagrams: true,
            includeSequenceDiagrams: true,
            includeComponentDetails: true,
            includeDataFlow: true,
            includeTechnologyStack: true,
            includeImplementationNotes: true
          });
        } else if (localSource === 'design') {
          hybridPackage.design = await this.generateDesign(localArtifacts, {
            projectName,
            includeArchitectureDiagrams: true,
            includeSequenceDiagrams: true,
            includeComponentDetails: true,
            includeDataFlow: true,
            includeTechnologyStack: true,
            includeImplementationNotes: true
          });
          
          // Generate requirements from both sources
          const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
          hybridPackage.requirements = await this.generateRequirements(combinedArtifacts, {
            projectName,
            includePropertyBasedTesting: true,
            earsValidation: true
          });
        }
        break;
    }

    // Add source tracking if enabled
    if (includeSourceTracking) {
      this.addSourceTracking(hybridPackage, opusArtifacts, localArtifacts, options);
    }

    // Validate consistency if enabled
    if (validateConsistency) {
      const validationErrors = this.validateHybridConsistency(hybridPackage, options);
      if (validationErrors.length > 0) {
        hybridPackage.metadata.consistencyValidation = validationErrors;
      }
    }

    return hybridPackage;
  }

  /**
   * Generate hybrid workflow with Opus-generated design and local requirements
   * 
   * Requirements: 16.6
   */
  async generateOpusDesignLocalRequirements(
    opusDesignArtifacts: ParsedArtifact[],
    localRequirementsArtifacts: ParsedArtifact[],
    projectName: string = 'Hybrid Project'
  ): Promise<HybridSpecPackage> {
    return this.generateHybridWorkflow(
      opusDesignArtifacts,
      localRequirementsArtifacts,
      {
        projectName,
        opusSource: 'design',
        localSource: 'requirements',
        includeSourceTracking: true,
        validateConsistency: true
      }
    );
  }

  /**
   * Generate hybrid workflow with Opus-generated requirements and local tasks
   * 
   * Requirements: 16.6
   */
  async generateOpusRequirementsLocalTasks(
    opusRequirementsArtifacts: ParsedArtifact[],
    localTaskArtifacts: ParsedArtifact[],
    projectName: string = 'Hybrid Project'
  ): Promise<HybridSpecPackage> {
    return this.generateHybridWorkflow(
      opusRequirementsArtifacts,
      localTaskArtifacts,
      {
        projectName,
        opusSource: 'requirements',
        localSource: 'tasks',
        includeSourceTracking: true,
        validateConsistency: true
      }
    );
  }

  /**
   * Generate configuration for hybrid workflows
   */
  private async generateHybridConfig(
    opusArtifacts: ParsedArtifact[],
    localArtifacts: ParsedArtifact[],
    options: HybridWorkflowOptions
  ): Promise<ConfigKiro> {
    const combinedArtifacts = [...opusArtifacts, ...localArtifacts];
    
    // Determine workflow type based on the hybrid configuration
    let workflowType: 'requirements-first' | 'design-first' | 'bugfix';
    
    if (options.opusSource === 'design' || options.localSource === 'design') {
      workflowType = 'design-first';
    } else if (options.opusSource === 'requirements' || options.localSource === 'requirements') {
      workflowType = 'requirements-first';
    } else {
      // Default based on artifacts
      workflowType = this.determineWorkflowType(combinedArtifacts);
    }

    return {
      specId: this.generateSpecId(),
      workflowType,
      specType: this.determineSpecType(combinedArtifacts),
      metadata: {
        generatedAt: new Date(),
        sourceArtifacts: combinedArtifacts.map(a => a.id),
        projectName: options.projectName
      }
    };
  }

  /**
   * Add source tracking annotations to hybrid spec documents
   */
  private addSourceTracking(
    hybridPackage: HybridSpecPackage,
    opusArtifacts: ParsedArtifact[],
    localArtifacts: ParsedArtifact[],
    options: HybridWorkflowOptions
  ): void {
    const opusIds = new Set(opusArtifacts.map(a => a.id));
    const localIds = new Set(localArtifacts.map(a => a.id));

    // Add source annotations to each document
    if (hybridPackage.requirements) {
      const sourceInfo = this.determineDocumentSources(hybridPackage.requirements.metadata.sourceArtifacts, opusIds, localIds);
      hybridPackage.requirements.content = this.addSourceAnnotations(
        hybridPackage.requirements.content,
        'requirements',
        sourceInfo,
        options
      );
    }

    if (hybridPackage.design) {
      const sourceInfo = this.determineDocumentSources(hybridPackage.design.metadata.sourceArtifacts, opusIds, localIds);
      hybridPackage.design.content = this.addSourceAnnotations(
        hybridPackage.design.content,
        'design',
        sourceInfo,
        options
      );
    }

    if (hybridPackage.tasks) {
      const sourceInfo = this.determineDocumentSources(hybridPackage.tasks.metadata.sourceArtifacts, opusIds, localIds);
      hybridPackage.tasks.content = this.addSourceAnnotations(
        hybridPackage.tasks.content,
        'tasks',
        sourceInfo,
        options
      );
    }
  }

  /**
   * Determine document sources (Opus vs Local)
   */
  private determineDocumentSources(
    artifactIds: string[],
    opusIds: Set<string>,
    localIds: Set<string>
  ): { opus: string[], local: string[], mixed: boolean } {
    const opus = artifactIds.filter(id => opusIds.has(id));
    const local = artifactIds.filter(id => localIds.has(id));
    
    return {
      opus,
      local,
      mixed: opus.length > 0 && local.length > 0
    };
  }

  /**
   * Add source annotations to document content
   */
  private addSourceAnnotations(
    content: string,
    documentType: string,
    sourceInfo: { opus: string[], local: string[], mixed: boolean },
    options: HybridWorkflowOptions
  ): string {
    const lines = content.split('\n');
    const annotatedLines: string[] = [];

    // Add header annotation
    annotatedLines.push(`<!-- HYBRID WORKFLOW: ${options.opusSource}-opus-${options.localSource}-local -->`);
    annotatedLines.push(`<!-- Generated: ${new Date().toISOString()} -->`);
    
    if (sourceInfo.mixed) {
      annotatedLines.push(`<!-- Mixed Sources: Opus artifacts [${sourceInfo.opus.join(', ')}], Local artifacts [${sourceInfo.local.join(', ')}] -->`);
    } else if (sourceInfo.opus.length > 0) {
      annotatedLines.push(`<!-- Opus Source: artifacts [${sourceInfo.opus.join(', ')}] -->`);
    } else {
      annotatedLines.push(`<!-- Local Source: artifacts [${sourceInfo.local.join(', ')}] -->`);
    }
    
    annotatedLines.push('');
    annotatedLines.push(...lines);

    return annotatedLines.join('\n');
  }

  /**
   * Validate consistency between Opus and local artifacts in hybrid workflow
   */
  private validateHybridConsistency(
    hybridPackage: HybridSpecPackage,
    options: HybridWorkflowOptions
  ): string[] {
    const errors: string[] = [];

    // Check for consistency between requirements and design
    if (hybridPackage.requirements && hybridPackage.design) {
      const reqErrors = this.validateRequirementsDesignConsistency(
        hybridPackage.requirements,
        hybridPackage.design
      );
      errors.push(...reqErrors);
    }

    // Check for consistency between design and tasks
    if (hybridPackage.design && hybridPackage.tasks) {
      const taskErrors = this.validateDesignTasksConsistency(
        hybridPackage.design,
        hybridPackage.tasks
      );
      errors.push(...taskErrors);
    }

    // Check for consistency between requirements and tasks
    if (hybridPackage.requirements && hybridPackage.tasks) {
      const reqTaskErrors = this.validateRequirementsTasksConsistency(
        hybridPackage.requirements,
        hybridPackage.tasks
      );
      errors.push(...reqTaskErrors);
    }

    // Check for hybrid-specific consistency issues
    const hybridErrors = this.validateHybridSpecificConsistency(hybridPackage, options);
    errors.push(...hybridErrors);

    return errors;
  }

  /**
   * Validate consistency between requirements and design documents
   */
  private validateRequirementsDesignConsistency(
    requirements: SpecDocument,
    design: SpecDocument
  ): string[] {
    const errors: string[] = [];

    // Extract key terms from requirements
    const reqTerms = this.extractKeyTerms(requirements.content);
    const designTerms = this.extractKeyTerms(design.content);

    // Check for missing key requirements terms in design
    const missingInDesign = reqTerms.filter(term => 
      !designTerms.some(dTerm => dTerm.toLowerCase().includes(term.toLowerCase()))
    );

    if (missingInDesign.length > 0) {
      errors.push(`Design document missing coverage for requirements terms: ${missingInDesign.join(', ')}`);
    }

    // Check for technology stack consistency
    const reqTech = this.extractTechnologyFromText(requirements.content);
    const designTech = this.extractTechnologyFromText(design.content);

    if (reqTech.languages && designTech.languages) {
      const inconsistentLangs = reqTech.languages.filter(lang => 
        !designTech.languages!.some(dLang => dLang.toLowerCase().includes(lang.toLowerCase()))
      );
      
      if (inconsistentLangs.length > 0) {
        errors.push(`Technology inconsistency: Requirements specify ${inconsistentLangs.join(', ')} but design doesn't include them`);
      }
    }

    return errors;
  }

  /**
   * Validate consistency between design and tasks documents
   */
  private validateDesignTasksConsistency(
    design: SpecDocument,
    tasks: TasksDocument
  ): string[] {
    const errors: string[] = [];

    // Extract component names from design
    const designComponents = this.extractComponentNamesFromContent(design.content);
    
    // Check if tasks cover all design components
    const taskContent = tasks.content.toLowerCase();
    const uncoveredComponents = designComponents.filter(component => 
      !taskContent.includes(component.toLowerCase())
    );

    if (uncoveredComponents.length > 0) {
      errors.push(`Tasks missing implementation for design components: ${uncoveredComponents.join(', ')}`);
    }

    // Check for API endpoint coverage
    const designEndpoints = this.extractAPIEndpointsFromContent(design.content);
    const uncoveredEndpoints = designEndpoints.filter(endpoint => 
      !taskContent.includes(endpoint.toLowerCase())
    );

    if (uncoveredEndpoints.length > 0) {
      errors.push(`Tasks missing implementation for API endpoints: ${uncoveredEndpoints.join(', ')}`);
    }

    return errors;
  }

  /**
   * Validate consistency between requirements and tasks documents
   */
  private validateRequirementsTasksConsistency(
    requirements: SpecDocument,
    tasks: TasksDocument
  ): string[] {
    const errors: string[] = [];

    // Extract requirement IDs from requirements document
    const reqIds = this.extractRequirementIds(requirements.content);
    
    // Check if tasks reference requirements
    const taskContent = tasks.content;
    const unreferencedReqs = reqIds.filter(reqId => 
      !taskContent.includes(reqId)
    );

    if (unreferencedReqs.length > 0 && reqIds.length > 0) {
      errors.push(`Tasks don't reference requirements: ${unreferencedReqs.join(', ')}`);
    }

    return errors;
  }

  /**
   * Validate hybrid-specific consistency issues
   */
  private validateHybridSpecificConsistency(
    hybridPackage: HybridSpecPackage,
    options: HybridWorkflowOptions
  ): string[] {
    const errors: string[] = [];

    // Check for source integration issues
    const opusArtifactCount = hybridPackage.metadata.opusArtifacts.length;
    const localArtifactCount = hybridPackage.metadata.localArtifacts.length;

    if (opusArtifactCount === 0) {
      errors.push('Hybrid workflow has no Opus artifacts - this may not be a true hybrid workflow');
    }

    if (localArtifactCount === 0) {
      errors.push('Hybrid workflow has no local artifacts - this may not be a true hybrid workflow');
    }

    // Check for workflow type consistency
    const workflowType = hybridPackage.config.workflowType;
    if (options.opusSource === 'design' && workflowType !== 'design-first') {
      errors.push(`Workflow type mismatch: Opus provides design but workflow is ${workflowType}`);
    }

    if (options.opusSource === 'requirements' && workflowType !== 'requirements-first') {
      errors.push(`Workflow type mismatch: Opus provides requirements but workflow is ${workflowType}`);
    }

    return errors;
  }

  /**
   * Extract key terms from document content
   */
  private extractKeyTerms(content: string): string[] {
    const terms: string[] = [];
    
    // Extract terms from headers
    const headerPattern = /^#+\s+(.+)$/gm;
    const headerMatches = content.matchAll(headerPattern);
    
    for (const match of headerMatches) {
      const headerText = match[1].trim();
      if (headerText.length > 3 && !headerText.toLowerCase().includes('document')) {
        terms.push(headerText);
      }
    }

    // Extract terms from bold text
    const boldPattern = /\*\*([^*]+)\*\*/g;
    const boldMatches = content.matchAll(boldPattern);
    
    for (const match of boldMatches) {
      const boldText = match[1].trim();
      if (boldText.length > 3) {
        terms.push(boldText);
      }
    }

    return [...new Set(terms)];
  }

  /**
   * Extract component names from content
   */
  private extractComponentNamesFromContent(content: string): string[] {
    const components: string[] = [];
    
    // Look for component patterns
    const componentPatterns = [
      /component[:\s]+([A-Za-z][A-Za-z0-9_]*)/gi,
      /module[:\s]+([A-Za-z][A-Za-z0-9_]*)/gi,
      /service[:\s]+([A-Za-z][A-Za-z0-9_]*)/gi,
      /class[:\s]+([A-Za-z][A-Za-z0-9_]*)/gi
    ];

    for (const pattern of componentPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const componentName = match[1].trim();
        if (componentName.length > 2) {
          components.push(componentName);
        }
      }
    }

    return [...new Set(components)];
  }

  /**
   * Extract API endpoints from content
   */
  private extractAPIEndpointsFromContent(content: string): string[] {
    const endpoints: string[] = [];
    
    // Look for endpoint patterns
    const endpointPatterns = [
      /(?:GET|POST|PUT|DELETE|PATCH)\s+(\/[^\s\n|]*)/gi,
      /endpoint[:\s]+(\/[^\s\n|]*)/gi,
      /path[:\s]+(\/[^\s\n|]*)/gi
    ];

    for (const pattern of endpointPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const endpoint = match[1].trim();
        if (endpoint.startsWith('/') && endpoint.length > 1) {
          endpoints.push(endpoint);
        }
      }
    }

    return [...new Set(endpoints)];
  }

  /**
   * Extract requirement IDs from content
   */
  private extractRequirementIds(content: string): string[] {
    const reqIds: string[] = [];
    
    // Look for requirement ID patterns
    const reqPatterns = [
      /(?:Requirement|REQ)[:\s]+(\d+(?:\.\d+)?)/gi,
      /(?:REQ|R)-(\d+(?:\.\d+)?)/gi,
      /###\s+Requirement\s+(\d+(?:\.\d+)?)/gi
    ];

    for (const pattern of reqPatterns) {
      const matches = content.matchAll(pattern);
      for (const match of matches) {
        const reqId = match[1].trim();
        reqIds.push(`Requirement ${reqId}`);
      }
    }

    return [...new Set(reqIds)];
  }

  /**
   * Generate a unique spec ID (UUID v4)
   */
  private generateSpecId(): string {
    // Simple UUID v4 generation
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  /**
   * Determine workflow type based on artifacts
   */
  private determineWorkflowType(artifacts: ParsedArtifact[]): 'requirements-first' | 'design-first' | 'bugfix' {
    // Check if this is a bugfix workflow
    if (this.isBugfixWorkflow(artifacts)) {
      return 'bugfix';
    }

    // Check artifact types to determine if design-first or requirements-first
    const hasArchitectureDiagrams = artifacts.some(a => 
      a.type === 'mermaid_diagram' && 
      (a.content.includes('graph') || a.content.includes('flowchart') || a.content.includes('classDiagram'))
    );
    
    const hasAPISpecs = artifacts.some(a => a.type === 'openapi_spec');
    
    const hasDetailedImplementationGuide = artifacts.some(a => 
      a.type === 'implementation_guide' && 
      a.structured?.implementationSteps && 
      a.structured.implementationSteps.length > 5
    );

    // If we have detailed architecture diagrams or API specs, it's likely design-first
    if (hasArchitectureDiagrams || hasAPISpecs || hasDetailedImplementationGuide) {
      return 'design-first';
    }

    // Default to requirements-first
    return 'requirements-first';
  }

  /**
   * Determine spec type based on artifacts
   */
  private determineSpecType(artifacts: ParsedArtifact[]): 'feature' | 'bugfix' {
    return this.isBugfixWorkflow(artifacts) ? 'bugfix' : 'feature';
  }

  /**
   * Check if this is a bugfix workflow based on artifact content
   */
  private isBugfixWorkflow(artifacts: ParsedArtifact[]): boolean {
    // Look for bugfix indicators in artifact content
    const bugfixKeywords = [
      'bug', 'fix', 'error', 'issue', 'problem', 'defect', 'failure',
      'timeout', 'crash', 'exception', 'incorrect', 'broken', 'failing'
    ];

    for (const artifact of artifacts) {
      const content = artifact.content.toLowerCase();
      
      // Check for multiple bugfix keywords
      const keywordCount = bugfixKeywords.filter(keyword => 
        content.includes(keyword)
      ).length;
      
      // If we find multiple bugfix keywords, it's likely a bugfix
      if (keywordCount >= 2) {
        return true;
      }

      // Check for specific bugfix patterns
      if (content.includes('bug fix') || 
          content.includes('error fix') || 
          content.includes('issue fix') ||
          content.includes('timeout fix') ||
          content.includes('failure fix')) {
        return true;
      }
    }

    return false;
  }

  /**
   * Generate .config.kiro file with workflow type and spec metadata
   * 
   * Requirements: 16.5
  /**
   * Export .config.kiro file content as JSON string
   */
  exportConfigKiro(config: ConfigKiro): string {
    // Create a clean config object without metadata for the file
    const fileConfig = {
      specId: config.specId,
      workflowType: config.workflowType,
      specType: config.specType
    };

    return JSON.stringify(fileConfig);
  }
}