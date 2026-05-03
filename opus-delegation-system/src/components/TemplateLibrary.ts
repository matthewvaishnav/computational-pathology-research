/**
 * Template Library Component
 * Implements Task 8 - Template Library Management
 * Requirements: 4.1-4.7
 */

import * as yaml from 'yaml';
import * as fs from 'fs';
import * as path from 'path';
import { DelegationType, ContextType, ArtifactType } from '../types/core.js';

// Template Parameter Definition (Requirement 4.4)
export interface TemplateParameter {
  name: string;
  required: boolean;
  type: 'string' | 'list' | 'number' | 'boolean';
  default?: any;
  description?: string;
}

// Template Structure (Requirements 4.1, 4.3)
export interface DelegationTemplate {
  template_id: string;
  name: string;
  category: DelegationType;
  version: string;
  description?: string;
  
  parameters: TemplateParameter[];
  
  context_requirements: ContextType[];
  
  expected_artifacts: Array<{
    type: ArtifactType;
    subtype?: string;
    format?: string;
    granularity?: string;
  }>;
  
  prompt_template: string;
  
  // Template metadata
  metadata?: {
    author?: string;
    created?: Date;
    updated?: Date;
    usageCount?: number;
  };
}

// Template Validation Error
export interface TemplateValidationError {
  field: string;
  message: string;
}

// Template Usage Statistics (Requirement 4.5)
export interface TemplateUsageStats {
  template_id: string;
  usageCount: number;
  lastUsed?: Date;
  successRate?: number;
}

/**
 * Template Library Manager
 * Manages loading, validation, parameterization, and versioning of templates
 */
export class TemplateLibrary {
  private templates: Map<string, DelegationTemplate> = new Map();
  private usageStats: Map<string, TemplateUsageStats> = new Map();
  private templatesDir: string;

  constructor(templatesDir: string = './templates') {
    this.templatesDir = templatesDir;
  }

  /**
   * Load a template from YAML file (Requirement 4.1)
   */
  loadTemplate(filePath: string): DelegationTemplate {
    try {
      const fileContent = fs.readFileSync(filePath, 'utf-8');
      const template = yaml.parse(fileContent) as DelegationTemplate;
      
      // Validate template structure
      const errors = this.validateTemplate(template);
      if (errors.length > 0) {
        throw new Error(`Template validation failed: ${errors.map(e => e.message).join(', ')}`);
      }
      
      // Store template
      this.templates.set(template.template_id, template);
      
      // Initialize usage stats if not exists
      if (!this.usageStats.has(template.template_id)) {
        this.usageStats.set(template.template_id, {
          template_id: template.template_id,
          usageCount: 0
        });
      }
      
      return template;
    } catch (error) {
      throw new Error(`Failed to load template from ${filePath}: ${error}`);
    }
  }

  /**
   * Load all templates from directory (Requirement 4.1)
   */
  loadAllTemplates(): void {
    if (!fs.existsSync(this.templatesDir)) {
      fs.mkdirSync(this.templatesDir, { recursive: true });
      return;
    }

    const files = fs.readdirSync(this.templatesDir);
    const yamlFiles = files.filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));
    
    for (const file of yamlFiles) {
      const filePath = path.join(this.templatesDir, file);
      try {
        this.loadTemplate(filePath);
      } catch (error) {
        console.error(`Failed to load template ${file}:`, error);
      }
    }
  }

  /**
   * Validate template structure (Requirement 4.3)
   */
  validateTemplate(template: any): TemplateValidationError[] {
    const errors: TemplateValidationError[] = [];

    // Required fields
    if (!template.template_id) {
      errors.push({ field: 'template_id', message: 'template_id is required' });
    }
    if (!template.name) {
      errors.push({ field: 'name', message: 'name is required' });
    }
    if (!template.category) {
      errors.push({ field: 'category', message: 'category is required' });
    }
    if (!template.version) {
      errors.push({ field: 'version', message: 'version is required' });
    }
    if (!template.parameters || !Array.isArray(template.parameters)) {
      errors.push({ field: 'parameters', message: 'parameters must be an array' });
    }
    if (!template.context_requirements || !Array.isArray(template.context_requirements)) {
      errors.push({ field: 'context_requirements', message: 'context_requirements must be an array' });
    }
    if (!template.expected_artifacts || !Array.isArray(template.expected_artifacts)) {
      errors.push({ field: 'expected_artifacts', message: 'expected_artifacts must be an array' });
    }
    if (!template.prompt_template) {
      errors.push({ field: 'prompt_template', message: 'prompt_template is required' });
    }

    // Validate parameters
    if (template.parameters && Array.isArray(template.parameters)) {
      template.parameters.forEach((param: any, index: number) => {
        if (!param.name) {
          errors.push({ field: `parameters[${index}].name`, message: 'parameter name is required' });
        }
        if (param.required === undefined) {
          errors.push({ field: `parameters[${index}].required`, message: 'parameter required flag is required' });
        }
        if (!param.type) {
          errors.push({ field: `parameters[${index}].type`, message: 'parameter type is required' });
        }
      });
    }

    return errors;
  }

  /**
   * Get template by ID
   */
  getTemplate(templateId: string): DelegationTemplate | undefined {
    return this.templates.get(templateId);
  }

  /**
   * List all templates
   */
  listTemplates(): DelegationTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * List templates by category
   */
  listTemplatesByCategory(category: DelegationType): DelegationTemplate[] {
    return Array.from(this.templates.values()).filter(t => t.category === category);
  }

  /**
   * Instantiate template with parameters (Requirements 4.4, 4.6)
   */
  instantiateTemplate(
    templateId: string,
    parameters: Record<string, any>,
    contextBundle: string
  ): string {
    const template = this.templates.get(templateId);
    if (!template) {
      throw new Error(`Template not found: ${templateId}`);
    }

    // Validate parameters
    const validationErrors = this.validateParameters(template, parameters);
    if (validationErrors.length > 0) {
      throw new Error(`Parameter validation failed: ${validationErrors.join(', ')}`);
    }

    // Apply defaults for missing optional parameters
    const completeParams = this.applyDefaults(template, parameters);

    // Substitute parameters in prompt template
    let instantiated = template.prompt_template;
    
    // Replace {{parameter}} placeholders
    for (const [key, value] of Object.entries(completeParams)) {
      const placeholder = `{{${key}}}`;
      const replacement = this.formatParameterValue(value);
      instantiated = instantiated.replace(new RegExp(placeholder, 'g'), replacement);
    }

    // Replace {{context_bundle}} placeholder
    instantiated = instantiated.replace(/{{context_bundle}}/g, contextBundle);

    // Update usage statistics
    this.incrementUsageCount(templateId);

    return instantiated;
  }

  /**
   * Validate parameters against template requirements (Requirement 4.6)
   */
  validateParameters(template: DelegationTemplate, parameters: Record<string, any>): string[] {
    const errors: string[] = [];

    // Check required parameters
    for (const param of template.parameters) {
      if (param.required && !(param.name in parameters)) {
        errors.push(`Required parameter missing: ${param.name}`);
      }

      // Type validation
      if (param.name in parameters) {
        const value = parameters[param.name];
        const valid = this.validateParameterType(value, param.type);
        if (!valid) {
          errors.push(`Parameter ${param.name} has invalid type. Expected ${param.type}`);
        }
      }
    }

    return errors;
  }

  /**
   * Validate parameter type
   */
  private validateParameterType(value: any, expectedType: string): boolean {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number';
      case 'boolean':
        return typeof value === 'boolean';
      case 'list':
        return Array.isArray(value);
      default:
        return false;
    }
  }

  /**
   * Apply default values for missing optional parameters (Requirement 4.6)
   */
  private applyDefaults(
    template: DelegationTemplate,
    parameters: Record<string, any>
  ): Record<string, any> {
    const complete = { ...parameters };

    for (const param of template.parameters) {
      if (!(param.name in complete) && param.default !== undefined) {
        complete[param.name] = param.default;
      }
    }

    return complete;
  }

  /**
   * Format parameter value for substitution
   */
  private formatParameterValue(value: any): string {
    if (Array.isArray(value)) {
      return value.join(', ');
    }
    return String(value);
  }

  /**
   * Increment usage count for template (Requirement 4.5)
   */
  private incrementUsageCount(templateId: string): void {
    const stats = this.usageStats.get(templateId);
    if (stats) {
      stats.usageCount++;
      stats.lastUsed = new Date();
    }
  }

  /**
   * Get usage statistics for template (Requirement 4.5)
   */
  getUsageStats(templateId: string): TemplateUsageStats | undefined {
    return this.usageStats.get(templateId);
  }

  /**
   * Get all usage statistics
   */
  getAllUsageStats(): TemplateUsageStats[] {
    return Array.from(this.usageStats.values());
  }

  /**
   * Save template to YAML file
   */
  saveTemplate(template: DelegationTemplate, filePath?: string): void {
    const errors = this.validateTemplate(template);
    if (errors.length > 0) {
      throw new Error(`Template validation failed: ${errors.map(e => e.message).join(', ')}`);
    }

    const targetPath = filePath || path.join(this.templatesDir, `${template.template_id}.yaml`);
    const yamlContent = yaml.stringify(template);
    
    fs.writeFileSync(targetPath, yamlContent, 'utf-8');
    this.templates.set(template.template_id, template);
    
    // Initialize usage stats if not exists
    if (!this.usageStats.has(template.template_id)) {
      this.usageStats.set(template.template_id, {
        template_id: template.template_id,
        usageCount: 0
      });
    }
  }

  /**
   * Create built-in templates (Requirement 4.2)
   */
  createBuiltInTemplates(): void {
    const builtInTemplates: DelegationTemplate[] = [
      this.createFederatedLearningTemplate(),
      this.createPacsIntegrationTemplate(),
      this.createPropertyBasedTestTemplate(),
      this.createWsiStreamingTemplate(),
      this.createRefactoringAnalysisTemplate()
    ];

    // Ensure templates directory exists
    if (!fs.existsSync(this.templatesDir)) {
      fs.mkdirSync(this.templatesDir, { recursive: true });
    }

    // Save each template
    for (const template of builtInTemplates) {
      this.saveTemplate(template);
    }
  }

  /**
   * Create Federated Learning Architecture template (Requirement 4.2)
   */
  private createFederatedLearningTemplate(): DelegationTemplate {
    return {
      template_id: 'federated_learning_architecture',
      name: 'Federated Learning System Architecture',
      category: DelegationType.ARCHITECTURE_DESIGN,
      version: '1.0.0',
      description: 'Design a federated learning system with distributed training and privacy preservation',
      
      parameters: [
        { name: 'system_name', required: true, type: 'string', description: 'Name of the federated learning system' },
        { name: 'node_types', required: true, type: 'list', description: 'Types of nodes in the system (e.g., coordinator, worker, aggregator)' },
        { name: 'aggregation_strategy', required: false, type: 'string', default: 'federated_averaging', description: 'Strategy for aggregating model updates' },
        { name: 'privacy_requirements', required: false, type: 'string', default: 'differential_privacy', description: 'Privacy preservation requirements' }
      ],
      
      context_requirements: [
        ContextType.ARCHITECTURE_DOCS,
        ContextType.CODE_SNIPPETS,
        ContextType.REQUIREMENTS_DOCS,
        ContextType.CONSTRAINTS
      ],
      
      expected_artifacts: [
        { type: ArtifactType.MERMAID_DIAGRAM, subtype: 'architecture' },
        { type: ArtifactType.OPENAPI_SPEC, format: 'yaml' },
        { type: ArtifactType.IMPLEMENTATION_GUIDE, granularity: 'detailed' }
      ],
      
      prompt_template: `Design a federated learning architecture for {{system_name}}.

**Node Types:** {{node_types}}
**Aggregation Strategy:** {{aggregation_strategy}}
**Privacy Requirements:** {{privacy_requirements}}

Please provide:
1. System architecture diagram (Mermaid) showing all node types and their interactions
2. Node communication API (OpenAPI YAML) for model updates and aggregation
3. Implementation plan with dependencies and complexity estimates

Consider the following context:
{{context_bundle}}`
    };
  }

  /**
   * Create PACS Integration Design template (Requirement 4.2)
   */
  private createPacsIntegrationTemplate(): DelegationTemplate {
    return {
      template_id: 'pacs_integration_design',
      name: 'PACS/DICOM Integration Design',
      category: DelegationType.INTEGRATION_DESIGN,
      version: '1.0.0',
      description: 'Design integration with PACS systems using DICOM protocol',
      
      parameters: [
        { name: 'pacs_system', required: true, type: 'string', description: 'Name of the PACS system to integrate with' },
        { name: 'dicom_operations', required: true, type: 'list', description: 'DICOM operations to support (e.g., C-FIND, C-MOVE, C-STORE)' },
        { name: 'authentication_method', required: false, type: 'string', default: 'AE_title', description: 'Authentication method for PACS' },
        { name: 'data_flow', required: false, type: 'string', default: 'bidirectional', description: 'Data flow direction (push, pull, bidirectional)' }
      ],
      
      context_requirements: [
        ContextType.EXTERNAL_INTERFACES,
        ContextType.API_ENDPOINTS,
        ContextType.ARCHITECTURE_DOCS,
        ContextType.CONSTRAINTS
      ],
      
      expected_artifacts: [
        { type: ArtifactType.MERMAID_DIAGRAM, subtype: 'sequence' },
        { type: ArtifactType.OPENAPI_SPEC, format: 'yaml' },
        { type: ArtifactType.IMPLEMENTATION_GUIDE, granularity: 'detailed' }
      ],
      
      prompt_template: `Design a PACS integration for {{pacs_system}}.

**DICOM Operations:** {{dicom_operations}}
**Authentication Method:** {{authentication_method}}
**Data Flow:** {{data_flow}}

Please provide:
1. Sequence diagram (Mermaid) showing DICOM message flow
2. Integration API specification (OpenAPI YAML) for PACS operations
3. Implementation plan including error handling and retry logic

Consider the following context:
{{context_bundle}}`
    };
  }

  /**
   * Create Property-Based Test Suite template (Requirement 4.2)
   */
  private createPropertyBasedTestTemplate(): DelegationTemplate {
    return {
      template_id: 'property_based_test_suite',
      name: 'Property-Based Test Suite Design',
      category: DelegationType.TEST_STRATEGY,
      version: '1.0.0',
      description: 'Design comprehensive property-based test suite',
      
      parameters: [
        { name: 'component_name', required: true, type: 'string', description: 'Name of the component to test' },
        { name: 'properties', required: true, type: 'list', description: 'Properties to verify (e.g., idempotence, commutativity, invariants)' },
        { name: 'test_framework', required: false, type: 'string', default: 'fast-check', description: 'Property-based testing framework' },
        { name: 'coverage_target', required: false, type: 'number', default: 90, description: 'Target code coverage percentage' }
      ],
      
      context_requirements: [
        ContextType.CODE_SNIPPETS,
        ContextType.TEST_FILES,
        ContextType.REQUIREMENTS_DOCS
      ],
      
      expected_artifacts: [
        { type: ArtifactType.TEST_STRATEGY, granularity: 'detailed' },
        { type: ArtifactType.CODE_SNIPPET, subtype: 'test_stubs' },
        { type: ArtifactType.IMPLEMENTATION_GUIDE, granularity: 'detailed' }
      ],
      
      prompt_template: `Design a property-based test suite for {{component_name}}.

**Properties to Verify:** {{properties}}
**Test Framework:** {{test_framework}}
**Coverage Target:** {{coverage_target}}%

Please provide:
1. Test strategy document with property definitions and generators
2. Test code stubs with property-based test examples
3. Implementation plan for achieving coverage target

Consider the following context:
{{context_bundle}}`
    };
  }

  /**
   * Create WSI Streaming Architecture template (Requirement 4.2)
   */
  private createWsiStreamingTemplate(): DelegationTemplate {
    return {
      template_id: 'wsi_streaming_architecture',
      name: 'WSI Streaming Architecture',
      category: DelegationType.ARCHITECTURE_DESIGN,
      version: '1.0.0',
      description: 'Design real-time whole slide image streaming architecture',
      
      parameters: [
        { name: 'system_name', required: true, type: 'string', description: 'Name of the WSI streaming system' },
        { name: 'tile_size', required: false, type: 'number', default: 256, description: 'Tile size in pixels' },
        { name: 'compression', required: false, type: 'string', default: 'jpeg', description: 'Image compression format' },
        { name: 'caching_strategy', required: false, type: 'string', default: 'lru', description: 'Tile caching strategy' }
      ],
      
      context_requirements: [
        ContextType.ARCHITECTURE_DOCS,
        ContextType.CODE_SNIPPETS,
        ContextType.CONSTRAINTS
      ],
      
      expected_artifacts: [
        { type: ArtifactType.MERMAID_DIAGRAM, subtype: 'architecture' },
        { type: ArtifactType.OPENAPI_SPEC, format: 'yaml' },
        { type: ArtifactType.IMPLEMENTATION_GUIDE, granularity: 'detailed' }
      ],
      
      prompt_template: `Design a WSI streaming architecture for {{system_name}}.

**Tile Size:** {{tile_size}}px
**Compression:** {{compression}}
**Caching Strategy:** {{caching_strategy}}

Please provide:
1. System architecture diagram (Mermaid) showing tile server, cache, and client
2. Streaming API specification (OpenAPI YAML) for tile requests
3. Implementation plan with performance optimization strategies

Consider the following context:
{{context_bundle}}`
    };
  }

  /**
   * Create Refactoring Analysis template (Requirement 4.2)
   */
  private createRefactoringAnalysisTemplate(): DelegationTemplate {
    return {
      template_id: 'refactoring_analysis',
      name: 'Refactoring Analysis',
      category: DelegationType.REFACTORING_ANALYSIS,
      version: '1.0.0',
      description: 'Analyze code for refactoring opportunities and create refactoring plan',
      
      parameters: [
        { name: 'target_component', required: true, type: 'string', description: 'Component or module to refactor' },
        { name: 'refactoring_goals', required: true, type: 'list', description: 'Refactoring goals (e.g., reduce complexity, improve testability, extract interfaces)' },
        { name: 'constraints', required: false, type: 'list', default: [], description: 'Refactoring constraints (e.g., maintain API compatibility, no breaking changes)' },
        { name: 'risk_tolerance', required: false, type: 'string', default: 'medium', description: 'Risk tolerance (low, medium, high)' }
      ],
      
      context_requirements: [
        ContextType.CODE_SNIPPETS,
        ContextType.DEPENDENCY_GRAPHS,
        ContextType.TEST_FILES,
        ContextType.ARCHITECTURE_DOCS
      ],
      
      expected_artifacts: [
        { type: ArtifactType.MERMAID_DIAGRAM, subtype: 'class' },
        { type: ArtifactType.IMPLEMENTATION_GUIDE, granularity: 'detailed' },
        { type: ArtifactType.CODE_SNIPPET, subtype: 'refactored_examples' }
      ],
      
      prompt_template: `Analyze {{target_component}} for refactoring opportunities.

**Refactoring Goals:** {{refactoring_goals}}
**Constraints:** {{constraints}}
**Risk Tolerance:** {{risk_tolerance}}

Please provide:
1. Current vs. proposed architecture diagram (Mermaid class diagram)
2. Refactoring plan with step-by-step instructions and risk assessment
3. Code examples showing before/after refactoring

Consider the following context:
{{context_bundle}}`
    };
  }
}
