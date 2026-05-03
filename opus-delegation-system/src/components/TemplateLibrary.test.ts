/**
 * Unit tests for Template Library Component
 * Tests Task 8.1-8.5 - Template loading, validation, parameterization, versioning
 * Requirements: 4.1-4.7
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { TemplateLibrary, DelegationTemplate } from './TemplateLibrary.js';
import { DelegationType, ContextType, ArtifactType } from '../types/core.js';
import * as fs from 'fs';
import * as path from 'path';

describe('TemplateLibrary', () => {
  let library: TemplateLibrary;
  const testTemplatesDir = './test-templates';

  beforeEach(() => {
    library = new TemplateLibrary(testTemplatesDir);
    
    // Create test templates directory
    if (!fs.existsSync(testTemplatesDir)) {
      fs.mkdirSync(testTemplatesDir, { recursive: true });
    }
  });

  afterEach(() => {
    // Clean up test templates directory
    if (fs.existsSync(testTemplatesDir)) {
      const files = fs.readdirSync(testTemplatesDir);
      for (const file of files) {
        fs.unlinkSync(path.join(testTemplatesDir, file));
      }
      fs.rmdirSync(testTemplatesDir);
    }
  });

  describe('Template Validation (Requirement 4.3)', () => {
    it('should validate complete template successfully', () => {
      const template: DelegationTemplate = {
        template_id: 'test_template',
        name: 'Test Template',
        category: DelegationType.ARCHITECTURE_DESIGN,
        version: '1.0.0',
        parameters: [],
        context_requirements: [ContextType.CODE_SNIPPETS],
        expected_artifacts: [{ type: ArtifactType.MERMAID_DIAGRAM }],
        prompt_template: 'Test prompt'
      };

      const errors = library.validateTemplate(template);
      expect(errors).toHaveLength(0);
    });

    it('should detect missing required fields', () => {
      const template: any = {
        name: 'Test Template'
        // Missing other required fields
      };

      const errors = library.validateTemplate(template);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors.some(e => e.field === 'template_id')).toBe(true);
      expect(errors.some(e => e.field === 'category')).toBe(true);
      expect(errors.some(e => e.field === 'version')).toBe(true);
    });

    it('should validate parameters array', () => {
      const template: any = {
        template_id: 'test',
        name: 'Test',
        category: DelegationType.API_DESIGN,
        version: '1.0.0',
        parameters: 'not an array',
        context_requirements: [],
        expected_artifacts: [],
        prompt_template: 'test'
      };

      const errors = library.validateTemplate(template);
      expect(errors.some(e => e.field === 'parameters')).toBe(true);
    });

    it('should validate parameter structure', () => {
      const template: any = {
        template_id: 'test',
        name: 'Test',
        category: DelegationType.API_DESIGN,
        version: '1.0.0',
        parameters: [
          { name: 'param1', required: true, type: 'string' },
          { required: true, type: 'string' }, // Missing name
          { name: 'param3', type: 'string' } // Missing required flag
        ],
        context_requirements: [],
        expected_artifacts: [],
        prompt_template: 'test'
      };

      const errors = library.validateTemplate(template);
      expect(errors.some(e => e.field.includes('parameters[1].name'))).toBe(true);
      expect(errors.some(e => e.field.includes('parameters[2].required'))).toBe(true);
    });
  });

  describe('Template Loading (Requirement 4.1)', () => {
    it('should load template from YAML file', () => {
      const template: DelegationTemplate = {
        template_id: 'load_test',
        name: 'Load Test Template',
        category: DelegationType.TEST_STRATEGY,
        version: '1.0.0',
        parameters: [
          { name: 'test_param', required: true, type: 'string' }
        ],
        context_requirements: [ContextType.TEST_FILES],
        expected_artifacts: [{ type: ArtifactType.TEST_STRATEGY }],
        prompt_template: 'Test {{test_param}}'
      };

      // Save template
      library.saveTemplate(template);

      // Create new library instance and load
      const newLibrary = new TemplateLibrary(testTemplatesDir);
      const filePath = path.join(testTemplatesDir, 'load_test.yaml');
      const loaded = newLibrary.loadTemplate(filePath);

      expect(loaded.template_id).toBe('load_test');
      expect(loaded.name).toBe('Load Test Template');
      expect(loaded.parameters).toHaveLength(1);
    });

    it('should load all templates from directory', () => {
      // Create multiple templates
      const templates: DelegationTemplate[] = [
        {
          template_id: 'template1',
          name: 'Template 1',
          category: DelegationType.ARCHITECTURE_DESIGN,
          version: '1.0.0',
          parameters: [],
          context_requirements: [],
          expected_artifacts: [],
          prompt_template: 'Test 1'
        },
        {
          template_id: 'template2',
          name: 'Template 2',
          category: DelegationType.API_DESIGN,
          version: '1.0.0',
          parameters: [],
          context_requirements: [],
          expected_artifacts: [],
          prompt_template: 'Test 2'
        }
      ];

      templates.forEach(t => library.saveTemplate(t));

      // Load all templates
      const newLibrary = new TemplateLibrary(testTemplatesDir);
      newLibrary.loadAllTemplates();

      const loaded = newLibrary.listTemplates();
      expect(loaded).toHaveLength(2);
    });

    it('should throw error for invalid template file', () => {
      const invalidYaml = 'invalid: yaml: content: [unclosed';
      const filePath = path.join(testTemplatesDir, 'invalid.yaml');
      fs.writeFileSync(filePath, invalidYaml);

      expect(() => library.loadTemplate(filePath)).toThrow();
    });
  });

  describe('Template Parameterization (Requirements 4.4, 4.6)', () => {
    let template: DelegationTemplate;

    beforeEach(() => {
      template = {
        template_id: 'param_test',
        name: 'Parameterization Test',
        category: DelegationType.ARCHITECTURE_DESIGN,
        version: '1.0.0',
        parameters: [
          { name: 'system_name', required: true, type: 'string' },
          { name: 'node_types', required: true, type: 'list' },
          { name: 'optional_param', required: false, type: 'string', default: 'default_value' },
          { name: 'number_param', required: false, type: 'number', default: 42 }
        ],
        context_requirements: [],
        expected_artifacts: [],
        prompt_template: 'System: {{system_name}}, Nodes: {{node_types}}, Optional: {{optional_param}}, Number: {{number_param}}\n\nContext: {{context_bundle}}'
      };

      library.saveTemplate(template);
    });

    it('should substitute parameters in template', () => {
      const params = {
        system_name: 'TestSystem',
        node_types: ['worker', 'coordinator'],
        optional_param: 'custom_value',
        number_param: 100
      };

      const result = library.instantiateTemplate('param_test', params, 'Context data here');

      expect(result).toContain('System: TestSystem');
      expect(result).toContain('Nodes: worker, coordinator');
      expect(result).toContain('Optional: custom_value');
      expect(result).toContain('Number: 100');
      expect(result).toContain('Context: Context data here');
    });

    it('should apply default values for missing optional parameters', () => {
      const params = {
        system_name: 'TestSystem',
        node_types: ['worker']
      };

      const result = library.instantiateTemplate('param_test', params, 'Context');

      expect(result).toContain('Optional: default_value');
      expect(result).toContain('Number: 42');
    });

    it('should throw error for missing required parameters', () => {
      const params = {
        system_name: 'TestSystem'
        // Missing required node_types
      };

      expect(() => library.instantiateTemplate('param_test', params, 'Context')).toThrow(/Required parameter missing: node_types/);
    });

    it('should validate parameter types', () => {
      const params = {
        system_name: 123, // Should be string
        node_types: ['worker']
      };

      expect(() => library.instantiateTemplate('param_test', params, 'Context')).toThrow(/invalid type/);
    });

    it('should format list parameters correctly', () => {
      const params = {
        system_name: 'TestSystem',
        node_types: ['worker', 'coordinator', 'aggregator']
      };

      const result = library.instantiateTemplate('param_test', params, 'Context');

      expect(result).toContain('Nodes: worker, coordinator, aggregator');
    });

    it('should throw error for non-existent template', () => {
      expect(() => library.instantiateTemplate('non_existent', {}, 'Context')).toThrow(/Template not found/);
    });
  });

  describe('Template Versioning and Usage Tracking (Requirement 4.5)', () => {
    let template: DelegationTemplate;

    beforeEach(() => {
      template = {
        template_id: 'usage_test',
        name: 'Usage Test',
        category: DelegationType.API_DESIGN,
        version: '1.0.0',
        parameters: [],
        context_requirements: [],
        expected_artifacts: [],
        prompt_template: 'Test'
      };

      library.saveTemplate(template);
    });

    it('should track template usage count', () => {
      const params = {};
      
      // Use template multiple times
      library.instantiateTemplate('usage_test', params, 'Context 1');
      library.instantiateTemplate('usage_test', params, 'Context 2');
      library.instantiateTemplate('usage_test', params, 'Context 3');

      const stats = library.getUsageStats('usage_test');
      expect(stats).toBeDefined();
      expect(stats!.usageCount).toBe(3);
    });

    it('should track last used timestamp', () => {
      const params = {};
      const beforeUse = new Date();
      
      library.instantiateTemplate('usage_test', params, 'Context');

      const stats = library.getUsageStats('usage_test');
      expect(stats).toBeDefined();
      expect(stats!.lastUsed).toBeDefined();
      expect(stats!.lastUsed!.getTime()).toBeGreaterThanOrEqual(beforeUse.getTime());
    });

    it('should return all usage statistics', () => {
      const template2: DelegationTemplate = {
        template_id: 'usage_test_2',
        name: 'Usage Test 2',
        category: DelegationType.TEST_STRATEGY,
        version: '1.0.0',
        parameters: [],
        context_requirements: [],
        expected_artifacts: [],
        prompt_template: 'Test 2'
      };

      library.saveTemplate(template2);
      library.instantiateTemplate('usage_test', {}, 'Context');
      library.instantiateTemplate('usage_test_2', {}, 'Context');

      const allStats = library.getAllUsageStats();
      expect(allStats.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Template Retrieval', () => {
    beforeEach(() => {
      const templates: DelegationTemplate[] = [
        {
          template_id: 'arch1',
          name: 'Architecture 1',
          category: DelegationType.ARCHITECTURE_DESIGN,
          version: '1.0.0',
          parameters: [],
          context_requirements: [],
          expected_artifacts: [],
          prompt_template: 'Test'
        },
        {
          template_id: 'arch2',
          name: 'Architecture 2',
          category: DelegationType.ARCHITECTURE_DESIGN,
          version: '1.0.0',
          parameters: [],
          context_requirements: [],
          expected_artifacts: [],
          prompt_template: 'Test'
        },
        {
          template_id: 'api1',
          name: 'API 1',
          category: DelegationType.API_DESIGN,
          version: '1.0.0',
          parameters: [],
          context_requirements: [],
          expected_artifacts: [],
          prompt_template: 'Test'
        }
      ];

      templates.forEach(t => library.saveTemplate(t));
    });

    it('should get template by ID', () => {
      const template = library.getTemplate('arch1');
      expect(template).toBeDefined();
      expect(template!.name).toBe('Architecture 1');
    });

    it('should return undefined for non-existent template', () => {
      const template = library.getTemplate('non_existent');
      expect(template).toBeUndefined();
    });

    it('should list all templates', () => {
      const templates = library.listTemplates();
      expect(templates).toHaveLength(3);
    });

    it('should list templates by category', () => {
      const archTemplates = library.listTemplatesByCategory(DelegationType.ARCHITECTURE_DESIGN);
      expect(archTemplates).toHaveLength(2);
      expect(archTemplates.every(t => t.category === DelegationType.ARCHITECTURE_DESIGN)).toBe(true);

      const apiTemplates = library.listTemplatesByCategory(DelegationType.API_DESIGN);
      expect(apiTemplates).toHaveLength(1);
    });
  });

  describe('Built-in Templates (Requirement 4.2)', () => {
    it('should create all built-in templates', () => {
      library.createBuiltInTemplates();

      const templates = library.listTemplates();
      expect(templates.length).toBeGreaterThanOrEqual(5);

      // Check for specific built-in templates
      expect(library.getTemplate('federated_learning_architecture')).toBeDefined();
      expect(library.getTemplate('pacs_integration_design')).toBeDefined();
      expect(library.getTemplate('property_based_test_suite')).toBeDefined();
      expect(library.getTemplate('wsi_streaming_architecture')).toBeDefined();
      expect(library.getTemplate('refactoring_analysis')).toBeDefined();
    });

    it('should validate federated learning template', () => {
      library.createBuiltInTemplates();
      const template = library.getTemplate('federated_learning_architecture');

      expect(template).toBeDefined();
      expect(template!.category).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(template!.parameters.some(p => p.name === 'system_name')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'node_types')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'aggregation_strategy')).toBe(true);
    });

    it('should validate PACS integration template', () => {
      library.createBuiltInTemplates();
      const template = library.getTemplate('pacs_integration_design');

      expect(template).toBeDefined();
      expect(template!.category).toBe(DelegationType.INTEGRATION_DESIGN);
      expect(template!.parameters.some(p => p.name === 'pacs_system')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'dicom_operations')).toBe(true);
    });

    it('should validate property-based test template', () => {
      library.createBuiltInTemplates();
      const template = library.getTemplate('property_based_test_suite');

      expect(template).toBeDefined();
      expect(template!.category).toBe(DelegationType.TEST_STRATEGY);
      expect(template!.parameters.some(p => p.name === 'component_name')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'properties')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'test_framework')).toBe(true);
    });

    it('should validate WSI streaming template', () => {
      library.createBuiltInTemplates();
      const template = library.getTemplate('wsi_streaming_architecture');

      expect(template).toBeDefined();
      expect(template!.category).toBe(DelegationType.ARCHITECTURE_DESIGN);
      expect(template!.parameters.some(p => p.name === 'system_name')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'tile_size')).toBe(true);
    });

    it('should validate refactoring analysis template', () => {
      library.createBuiltInTemplates();
      const template = library.getTemplate('refactoring_analysis');

      expect(template).toBeDefined();
      expect(template!.category).toBe(DelegationType.REFACTORING_ANALYSIS);
      expect(template!.parameters.some(p => p.name === 'target_component')).toBe(true);
      expect(template!.parameters.some(p => p.name === 'refactoring_goals')).toBe(true);
    });

    it('should instantiate federated learning template', () => {
      library.createBuiltInTemplates();

      const params = {
        system_name: 'MedicalFL',
        node_types: ['hospital_node', 'coordinator', 'aggregator'],
        aggregation_strategy: 'secure_aggregation',
        privacy_requirements: 'differential_privacy'
      };

      const result = library.instantiateTemplate(
        'federated_learning_architecture',
        params,
        'Existing ML models and data schemas...'
      );

      expect(result).toContain('MedicalFL');
      expect(result).toContain('hospital_node, coordinator, aggregator');
      expect(result).toContain('secure_aggregation');
      expect(result).toContain('differential_privacy');
      expect(result).toContain('Existing ML models');
    });

    it('should instantiate PACS integration template', () => {
      library.createBuiltInTemplates();

      const params = {
        pacs_system: 'Orthanc',
        dicom_operations: ['C-FIND', 'C-MOVE', 'C-STORE'],
        authentication_method: 'AE_title',
        data_flow: 'bidirectional'
      };

      const result = library.instantiateTemplate(
        'pacs_integration_design',
        params,
        'PACS configuration and constraints...'
      );

      expect(result).toContain('Orthanc');
      expect(result).toContain('C-FIND, C-MOVE, C-STORE');
      expect(result).toContain('AE_title');
      expect(result).toContain('bidirectional');
    });
  });

  describe('Template Completeness (Requirement 4.7)', () => {
    it('should ensure all templates have required fields', () => {
      library.createBuiltInTemplates();
      const templates = library.listTemplates();

      for (const template of templates) {
        expect(template.template_id).toBeDefined();
        expect(template.name).toBeDefined();
        expect(template.category).toBeDefined();
        expect(template.version).toBeDefined();
        expect(template.parameters).toBeDefined();
        expect(Array.isArray(template.parameters)).toBe(true);
        expect(template.context_requirements).toBeDefined();
        expect(Array.isArray(template.context_requirements)).toBe(true);
        expect(template.expected_artifacts).toBeDefined();
        expect(Array.isArray(template.expected_artifacts)).toBe(true);
        expect(template.prompt_template).toBeDefined();
        expect(template.prompt_template.length).toBeGreaterThan(0);
      }
    });

    it('should ensure all templates have valid parameters', () => {
      library.createBuiltInTemplates();
      const templates = library.listTemplates();

      for (const template of templates) {
        for (const param of template.parameters) {
          expect(param.name).toBeDefined();
          expect(param.required).toBeDefined();
          expect(param.type).toBeDefined();
          expect(['string', 'number', 'boolean', 'list']).toContain(param.type);
        }
      }
    });

    it('should ensure all templates have context placeholders', () => {
      library.createBuiltInTemplates();
      const templates = library.listTemplates();

      for (const template of templates) {
        expect(template.prompt_template).toContain('{{context_bundle}}');
      }
    });
  });
});
