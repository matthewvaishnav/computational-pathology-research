/**
 * Artifact Exporter Component
 * Implements Task 18 - Artifact Export and Integration
 * Requirements: 10.1-10.7
 */

import { ParsedArtifact, ArtifactType } from '../types/core.js';
import { ImplementationGuide } from './ImplementationGuideGenerator.js';
import * as fs from 'fs';
import * as path from 'path';
import { stringify as stringifyYaml } from 'yaml';

/**
 * Export format options
 */
export type ExportFormat = 'yaml' | 'json' | 'markdown' | 'html' | 'png' | 'svg' | 'zip';

/**
 * Export result
 */
export interface ExportResult {
  success: boolean;
  outputPath?: string;
  error?: string;
}

/**
 * Artifact Exporter Component
 * Exports artifacts in various formats for documentation and implementation
 */
export class ArtifactExporter {
  private outputDir: string;

  constructor(outputDir: string = './exports') {
    this.outputDir = outputDir;
    this.ensureOutputDirectory();
  }

  /**
   * Ensure output directory exists
   */
  private ensureOutputDirectory(): void {
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
    }
  }

  /**
   * Export Mermaid diagram
   * Requirement 10.1: Export diagrams as PNG/SVG images
   * Task 18.1: Create Mermaid diagram exporter
   * 
   * Note: PNG/SVG export requires Mermaid CLI (mmdc) to be installed
   * For now, exports as .mmd text file
   */
  public exportMermaidDiagram(
    artifact: ParsedArtifact,
    filename: string,
    format: 'mmd' | 'png' | 'svg' = 'mmd'
  ): ExportResult {
    if (artifact.type !== ArtifactType.MERMAID_DIAGRAM) {
      return {
        success: false,
        error: 'Artifact is not a Mermaid diagram',
      };
    }

    try {
      if (format === 'mmd') {
        // Export as Mermaid text file
        const outputPath = path.join(this.outputDir, `${filename}.mmd`);
        fs.writeFileSync(outputPath, artifact.content, 'utf-8');

        return {
          success: true,
          outputPath,
        };
      } else {
        // PNG/SVG export requires Mermaid CLI
        return {
          success: false,
          error: `${format.toUpperCase()} export requires Mermaid CLI (mmdc) to be installed. Use 'npm install -g @mermaid-js/mermaid-cli' to install.`,
        };
      }
    } catch (error) {
      return {
        success: false,
        error: `Failed to export Mermaid diagram: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  /**
   * Export OpenAPI specification
   * Requirement 10.2: Export as YAML files, generate HTML documentation
   * Task 18.2: Create OpenAPI specification exporter
   */
  public exportOpenAPISpec(
    artifact: ParsedArtifact,
    filename: string,
    format: 'yaml' | 'json' | 'html' = 'yaml'
  ): ExportResult {
    if (artifact.type !== ArtifactType.OPENAPI_SPEC) {
      return {
        success: false,
        error: 'Artifact is not an OpenAPI specification',
      };
    }

    try {
      const openapi = artifact.structured?.openapi;
      if (!openapi) {
        return {
          success: false,
          error: 'OpenAPI specification structure not found',
        };
      }

      if (format === 'yaml') {
        const outputPath = path.join(this.outputDir, `${filename}.yaml`);
        const yamlContent = stringifyYaml(openapi);
        fs.writeFileSync(outputPath, yamlContent, 'utf-8');

        return {
          success: true,
          outputPath,
        };
      } else if (format === 'json') {
        const outputPath = path.join(this.outputDir, `${filename}.json`);
        fs.writeFileSync(outputPath, JSON.stringify(openapi, null, 2), 'utf-8');

        return {
          success: true,
          outputPath,
        };
      } else if (format === 'html') {
        // HTML documentation generation using Redoc
        const outputPath = path.join(this.outputDir, `${filename}.html`);
        const htmlContent = this.generateRedocHTML(openapi, filename);
        fs.writeFileSync(outputPath, htmlContent, 'utf-8');

        return {
          success: true,
          outputPath,
        };
      }

      return {
        success: false,
        error: `Unsupported format: ${format}`,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to export OpenAPI spec: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  /**
   * HTML escape function to prevent XSS
   */
  private escapeHtml(str: string): string {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /**
   * Recursively sanitize object for safe HTML embedding
   */
  private sanitizeObject(obj: any): any {
    if (typeof obj === 'string') return this.escapeHtml(obj);
    if (Array.isArray(obj)) return obj.map(item => this.sanitizeObject(item));
    if (obj && typeof obj === 'object') {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[key] = this.sanitizeObject(value);
      }
      return sanitized;
    }
    return obj;
  }

  /**
   * Generate Redoc HTML documentation
   */
  private generateRedocHTML(openapi: any, title: string): string {
    // Sanitize title and OpenAPI spec
    const safeTitle = this.escapeHtml(title);
    const safeSpec = this.sanitizeObject(openapi);
    const specJson = JSON.stringify(safeSpec, null, 2);

    return `<!DOCTYPE html>
<html>
<head>
  <title>${safeTitle} - API Documentation</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://cdn.redoc.ly; style-src 'self' 'unsafe-inline' https://cdn.redoc.ly; img-src 'self' data: https:; font-src 'self' data: https://cdn.redoc.ly;">
  <style>
    body {
      margin: 0;
      padding: 0;
    }
  </style>
</head>
<body>
  <div id="redoc-container"></div>
  <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
  <script>
    const spec = ${specJson};
    Redoc.init(spec, {}, document.getElementById('redoc-container'));
  </script>
</body>
</html>`;
  }

  /**
   * Export implementation guide
   * Requirement 10.3: Export as markdown files
   * Task 18.3: Create implementation guide exporter
   */
  public exportImplementationGuide(
    guide: ImplementationGuide,
    filename: string
  ): ExportResult {
    try {
      const outputPath = path.join(this.outputDir, `${filename}.md`);
      
      // Generate markdown content
      let md = `# ${guide.title}\n\n`;
      md += `${guide.overview}\n\n`;
      md += `**Generated:** ${guide.generatedAt.toISOString()}\n\n`;

      // Prerequisites
      if (guide.prerequisites.length > 0) {
        md += '## Prerequisites\n\n';
        for (const prereq of guide.prerequisites) {
          md += `- ${prereq}\n`;
        }
        md += '\n';
      }

      // Phases
      for (const phase of guide.phases) {
        md += `## ${phase.name}\n\n`;
        md += `**Complexity:** ${phase.complexity}\n`;
        if (phase.estimatedTime) {
          md += `**Estimated Time:** ${phase.estimatedTime}\n`;
        }
        md += '\n';

        for (const step of phase.steps) {
          md += `### ${step.action}\n\n`;
          md += `${step.description}\n\n`;

          if (step.dependencies.length > 0) {
            md += `**Dependencies:** ${step.dependencies.join(', ')}\n\n`;
          }

          md += `**Complexity:** ${step.complexity}\n\n`;
        }
      }

      // Code mappings
      if (guide.codeMappings.length > 0) {
        md += '## Code Mappings\n\n';
        md += '| Artifact | Target File | Type |\n';
        md += '|----------|-------------|------|\n';

        for (const mapping of guide.codeMappings) {
          md += `| ${mapping.artifactReference} | \`${mapping.targetFilePath}\` | ${mapping.codeType} |\n`;
        }
        md += '\n';
      }

      // Risks
      if (guide.risks.length > 0) {
        md += '## Risk Register\n\n';
        md += '| Risk | Mitigation |\n';
        md += '|------|------------|\n';

        for (const risk of guide.risks) {
          md += `| ${risk.risk} | ${risk.mitigation} |\n`;
        }
        md += '\n';
      }

      fs.writeFileSync(outputPath, md, 'utf-8');

      return {
        success: true,
        outputPath,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to export implementation guide: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  /**
   * Export test strategy
   * Requirement 10.4: Export as test file templates with stubs
   * Task 18.4: Create test strategy exporter
   */
  public exportTestStrategy(
    artifact: ParsedArtifact,
    filename: string
  ): ExportResult {
    if (artifact.type !== ArtifactType.TEST_STRATEGY) {
      return {
        success: false,
        error: 'Artifact is not a test strategy',
      };
    }

    try {
      const content = artifact.content;
      const outputPath = path.join(this.outputDir, `${filename}.test.ts`);

      // Generate test file template
      let testCode = `/**
 * Test Suite: ${filename}
 * Generated from test strategy
 */

import { describe, it, expect } from 'vitest';

`;

      // Check for different test types
      const hasUnitTests = /unit test/i.test(content);
      const hasIntegrationTests = /integration test/i.test(content);
      const hasPropertyTests = /property.*test|property-based/i.test(content);

      if (hasUnitTests) {
        testCode += `describe('Unit Tests', () => {
  it('should pass unit test', () => {
    // TODO: Implement unit test based on test strategy
    expect(true).toBe(true);
  });
});

`;
      }

      if (hasIntegrationTests) {
        testCode += `describe('Integration Tests', () => {
  it('should pass integration test', () => {
    // TODO: Implement integration test based on test strategy
    expect(true).toBe(true);
  });
});

`;
      }

      if (hasPropertyTests) {
        testCode += `import * as fc from 'fast-check';

describe('Property-Based Tests', () => {
  it('should satisfy property', () => {
    fc.assert(
      fc.property(fc.integer(), (value) => {
        // TODO: Implement property test based on test strategy
        return true;
      })
    );
  });
});

`;
      }

      // Add test strategy as comment
      testCode += `/*
Test Strategy:

${content}
*/
`;

      fs.writeFileSync(outputPath, testCode, 'utf-8');

      return {
        success: true,
        outputPath,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to export test strategy: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  /**
   * Export complete delegation package
   * Requirement 10.5: Export complete delegation packages as ZIP archives
   * Task 18.5: Create package exporter
   * 
   * Note: ZIP creation requires additional library (e.g., archiver)
   * For now, creates directory structure
   */
  public exportDelegationPackage(
    sessionId: string,
    artifacts: ParsedArtifact[],
    contextBundle: string,
    implementationGuide?: ImplementationGuide
  ): ExportResult {
    try {
      const packageDir = path.join(this.outputDir, `delegation-${sessionId}`);

      // Create package directory structure
      if (!fs.existsSync(packageDir)) {
        fs.mkdirSync(packageDir, { recursive: true });
      }

      const artifactsDir = path.join(packageDir, 'artifacts');
      if (!fs.existsSync(artifactsDir)) {
        fs.mkdirSync(artifactsDir, { recursive: true });
      }

      // Export context bundle
      const contextPath = path.join(packageDir, 'context.md');
      fs.writeFileSync(contextPath, contextBundle, 'utf-8');

      // Export artifacts
      for (let i = 0; i < artifacts.length; i++) {
        const artifact = artifacts[i];
        const artifactFilename = `artifact-${i + 1}-${artifact.type}`;

        switch (artifact.type) {
          case ArtifactType.MERMAID_DIAGRAM:
            const mermaidPath = path.join(artifactsDir, `${artifactFilename}.mmd`);
            fs.writeFileSync(mermaidPath, artifact.content, 'utf-8');
            break;

          case ArtifactType.OPENAPI_SPEC:
            if (artifact.structured?.openapi) {
              const openapiPath = path.join(artifactsDir, `${artifactFilename}.yaml`);
              fs.writeFileSync(openapiPath, stringifyYaml(artifact.structured.openapi), 'utf-8');
            }
            break;

          case ArtifactType.IMPLEMENTATION_GUIDE:
          case ArtifactType.TEST_STRATEGY:
            const mdPath = path.join(artifactsDir, `${artifactFilename}.md`);
            fs.writeFileSync(mdPath, artifact.content, 'utf-8');
            break;

          default:
            const defaultPath = path.join(artifactsDir, `${artifactFilename}.txt`);
            fs.writeFileSync(defaultPath, artifact.content, 'utf-8');
        }
      }

      // Export implementation guide
      if (implementationGuide) {
        const guidePath = path.join(packageDir, 'implementation-guide.md');
        const guideResult = this.exportImplementationGuide(implementationGuide, 'implementation-guide');
        if (guideResult.success && guideResult.outputPath) {
          // Move to package directory
          fs.renameSync(guideResult.outputPath, guidePath);
        }
      }

      // Create README
      const readmePath = path.join(packageDir, 'README.md');
      const readme = `# Delegation Package: ${sessionId}

This package contains all artifacts from the Opus delegation session.

## Contents

- \`context.md\` - Context bundle sent to Opus
- \`artifacts/\` - All generated artifacts
- \`implementation-guide.md\` - Implementation guide (if available)

## Artifacts

${artifacts.map((a, i) => `${i + 1}. ${a.type}`).join('\n')}

Generated: ${new Date().toISOString()}
`;

      fs.writeFileSync(readmePath, readme, 'utf-8');

      return {
        success: true,
        outputPath: packageDir,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to export delegation package: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  /**
   * Export artifact in appropriate format
   */
  public exportArtifact(
    artifact: ParsedArtifact,
    filename: string,
    format?: ExportFormat
  ): ExportResult {
    switch (artifact.type) {
      case ArtifactType.MERMAID_DIAGRAM:
        return this.exportMermaidDiagram(artifact, filename, format as 'mmd' | 'png' | 'svg');

      case ArtifactType.OPENAPI_SPEC:
        return this.exportOpenAPISpec(artifact, filename, format as 'yaml' | 'json' | 'html');

      case ArtifactType.TEST_STRATEGY:
        return this.exportTestStrategy(artifact, filename);

      case ArtifactType.IMPLEMENTATION_GUIDE:
      case ArtifactType.CODE_SNIPPET:
      default:
        // Export as markdown
        try {
          const outputPath = path.join(this.outputDir, `${filename}.md`);
          fs.writeFileSync(outputPath, artifact.content, 'utf-8');
          return {
            success: true,
            outputPath,
          };
        } catch (error) {
          return {
            success: false,
            error: `Failed to export artifact: ${error instanceof Error ? error.message : 'Unknown error'}`,
          };
        }
    }
  }

  /**
   * Get output directory
   */
  public getOutputDir(): string {
    return this.outputDir;
  }

  /**
   * Set output directory
   */
  public setOutputDir(dir: string): void {
    this.outputDir = dir;
    this.ensureOutputDirectory();
  }

  /**
   * Clean output directory
   */
  public cleanOutputDir(): void {
    if (fs.existsSync(this.outputDir)) {
      fs.rmSync(this.outputDir, { recursive: true, force: true });
    }
    this.ensureOutputDirectory();
  }
}
