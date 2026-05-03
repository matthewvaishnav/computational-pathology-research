/**
 * Security tests for TemplateLibrary - Path Traversal Prevention
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { TemplateLibrary, DelegationTemplate } from './TemplateLibrary.js';
import { DelegationType, ContextType, ArtifactType } from '../types/core.js';
import * as fs from 'fs';
import * as path from 'path';

describe('TemplateLibrary Security', () => {
  let library: TemplateLibrary;
  let testTemplatesDir: string;

  beforeEach(() => {
    testTemplatesDir = path.join(process.cwd(), 'test-templates-security');
    if (!fs.existsSync(testTemplatesDir)) {
      fs.mkdirSync(testTemplatesDir);
    }
    library = new TemplateLibrary(testTemplatesDir);
  });

  afterEach(() => {
    if (fs.existsSync(testTemplatesDir)) {
      const files = fs.readdirSync(testTemplatesDir);
      for (const file of files) {
        fs.unlinkSync(path.join(testTemplatesDir, file));
      }
      fs.rmdirSync(testTemplatesDir);
    }
  });

  describe('Path Traversal Prevention', () => {
    it('should block path traversal in loadTemplate', () => {
      // Attempt to read outside templates directory
      expect(() => {
        library.loadTemplate('../../../etc/passwd');
      }).toThrow(/Path traversal detected/);

      expect(() => {
        library.loadTemplate('../../package.json');
      }).toThrow(/Path traversal detected/);
    });

    it('should block absolute paths outside templates dir in loadTemplate', () => {
      expect(() => {
        library.loadTemplate('/etc/passwd');
      }).toThrow(/Path traversal detected/);

      expect(() => {
        library.loadTemplate('C:\\Windows\\System32\\config\\SAM');
      }).toThrow(/Path traversal detected/);
    });

    it('should block path traversal in saveTemplate', () => {
      const template: DelegationTemplate = {
        template_id: 'test-template',
        name: 'Test Template',
        version: '1.0.0',
        category: 'architecture',
        delegation_type: DelegationType.ARCHITECTURE_DESIGN,
        problem_description: 'Test',
        objectives: ['Test'],
        constraints: [],
        context_requirements: [ContextType.CODE_SNIPPETS],
        expected_artifacts: [ArtifactType.MERMAID_DIAGRAM],
        output_format_guidance: 'Test',
        example_context: 'Test',
        example_artifacts: 'Test',
        parameters: [],
        prompt_template: 'Test template',
      };

      // Attempt to write outside templates directory
      expect(() => {
        library.saveTemplate(template, '../../../malicious.yaml');
      }).toThrow(/Path traversal detected/);

      expect(() => {
        library.saveTemplate(template, '../../package.json');
      }).toThrow(/Path traversal detected/);
    });

    it('should block absolute paths outside templates dir in saveTemplate', () => {
      const template: DelegationTemplate = {
        template_id: 'test-template',
        name: 'Test Template',
        version: '1.0.0',
        category: 'architecture',
        delegation_type: DelegationType.ARCHITECTURE_DESIGN,
        problem_description: 'Test',
        objectives: ['Test'],
        constraints: [],
        context_requirements: [ContextType.CODE_SNIPPETS],
        expected_artifacts: [ArtifactType.MERMAID_DIAGRAM],
        output_format_guidance: 'Test',
        example_context: 'Test',
        example_artifacts: 'Test',
        parameters: [],
        prompt_template: 'Test template',
      };

      expect(() => {
        library.saveTemplate(template, '/tmp/malicious.yaml');
      }).toThrow(/Path traversal detected/);
    });

    it('should allow valid paths within templates directory', () => {
      const template: DelegationTemplate = {
        template_id: 'valid-template',
        name: 'Valid Template',
        version: '1.0.0',
        category: 'architecture',
        delegation_type: DelegationType.ARCHITECTURE_DESIGN,
        problem_description: 'Test',
        objectives: ['Test'],
        constraints: [],
        context_requirements: [ContextType.CODE_SNIPPETS],
        expected_artifacts: [ArtifactType.MERMAID_DIAGRAM],
        output_format_guidance: 'Test',
        example_context: 'Test',
        example_artifacts: 'Test',
        parameters: [],
        prompt_template: 'Test template',
      };

      // Save with relative path within templates dir
      const validPath = path.join(testTemplatesDir, 'valid.yaml');
      expect(() => {
        library.saveTemplate(template, validPath);
      }).not.toThrow();

      // Verify file was created
      expect(fs.existsSync(validPath)).toBe(true);

      // Load it back
      expect(() => {
        library.loadTemplate(validPath);
      }).not.toThrow();
    });

    it('should allow default save path (no explicit path)', () => {
      const template: DelegationTemplate = {
        template_id: 'default-path-template',
        name: 'Default Path Template',
        version: '1.0.0',
        category: 'architecture',
        delegation_type: DelegationType.ARCHITECTURE_DESIGN,
        problem_description: 'Test',
        objectives: ['Test'],
        constraints: [],
        context_requirements: [ContextType.CODE_SNIPPETS],
        expected_artifacts: [ArtifactType.MERMAID_DIAGRAM],
        output_format_guidance: 'Test',
        example_context: 'Test',
        example_artifacts: 'Test',
        parameters: [],
        prompt_template: 'Test template',
      };

      // Save without explicit path (uses templatesDir)
      expect(() => {
        library.saveTemplate(template);
      }).not.toThrow();

      // Verify file was created in templates dir
      const expectedPath = path.join(testTemplatesDir, 'default-path-template.yaml');
      expect(fs.existsSync(expectedPath)).toBe(true);
    });
  });
});
