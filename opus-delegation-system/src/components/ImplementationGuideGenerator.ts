/**
 * Implementation Guide Generator Component
 * Implements Task 14 - Implementation Guide Generation
 * Requirements: 7.1-7.7
 */

import {
  ParsedArtifact,
  ArtifactType,
  Step,
  ComplexityLevel,
  MermaidAST,
  OpenAPISpec,
} from '../types/core.js';

/**
 * Implementation phase grouping
 */
export interface ImplementationPhase {
  name: string;
  complexity: ComplexityLevel;
  steps: Step[];
  estimatedTime?: string;
}

/**
 * Code mapping for implementation
 */
export interface CodeMapping {
  artifactReference: string;
  targetFilePath: string;
  codeType: 'interface' | 'class' | 'function' | 'route' | 'test';
  boilerplate?: string;
}

/**
 * Implementation guide structure
 */
export interface ImplementationGuide {
  title: string;
  overview: string;
  prerequisites: string[];
  phases: ImplementationPhase[];
  codeMappings: CodeMapping[];
  risks: Array<{ risk: string; mitigation: string }>;
  generatedAt: Date;
}

/**
 * Implementation Guide Generator Component
 * Transforms validated artifacts into actionable implementation instructions
 */
export class ImplementationGuideGenerator {
  /**
   * Generate implementation guide from parsed artifacts
   * Requirement 7.1: Generate step-by-step instructions from artifacts
   * Task 14.1: Create guide structure generator
   */
  public generateGuide(
    artifacts: ParsedArtifact[],
    projectName: string
  ): ImplementationGuide {
    const steps = this.extractStepsFromArtifacts(artifacts);
    const sortedSteps = this.sortStepsByDependencies(steps);
    const phases = this.organizeIntoPhases(sortedSteps);
    const codeMappings = this.generateCodeMappings(artifacts);
    const prerequisites = this.extractPrerequisites(artifacts);
    const risks = this.identifyRisks(artifacts, steps);

    return {
      title: `Implementation Guide: ${projectName}`,
      overview: this.generateOverview(artifacts),
      prerequisites,
      phases,
      codeMappings,
      risks,
      generatedAt: new Date(),
    };
  }

  /**
   * Extract all implementation steps from artifacts
   */
  private extractStepsFromArtifacts(artifacts: ParsedArtifact[]): Step[] {
    const steps: Step[] = [];

    for (const artifact of artifacts) {
      switch (artifact.type) {
        case ArtifactType.IMPLEMENTATION_GUIDE:
          if (artifact.structured?.implementationSteps) {
            steps.push(...artifact.structured.implementationSteps);
          }
          break;

        case ArtifactType.MERMAID_DIAGRAM:
          // Generate steps from architecture diagram
          steps.push(...this.generateStepsFromDiagram(artifact));
          break;

        case ArtifactType.OPENAPI_SPEC:
          // Generate steps from API specification
          steps.push(...this.generateStepsFromAPI(artifact));
          break;

        case ArtifactType.TEST_STRATEGY:
          // Generate steps from test strategy
          steps.push(...this.generateStepsFromTests(artifact));
          break;
      }
    }

    return steps;
  }

  /**
   * Generate implementation steps from architecture diagram
   * Requirement 7.2: Map architecture components to file paths
   */
  private generateStepsFromDiagram(artifact: ParsedArtifact): Step[] {
    const steps: Step[] = [];
    const mermaid = artifact.structured?.mermaid;

    if (!mermaid || !mermaid.nodes) {
      return steps;
    }

    // Create steps for each component
    for (const node of mermaid.nodes) {
      steps.push({
        id: `impl-${node.id}`,
        action: `Implement ${node.label || node.id} component`,
        description: `Create the ${node.label || node.id} component as shown in the architecture diagram`,
        dependencies: [],
        complexity: ComplexityLevel.MODERATE,
      });
    }

    // Add integration steps for edges
    if (mermaid.edges && mermaid.edges.length > 0) {
      steps.push({
        id: 'impl-integration',
        action: 'Integrate components',
        description: 'Connect components according to architecture diagram relationships',
        dependencies: mermaid.nodes.map((n) => `impl-${n.id}`),
        complexity: ComplexityLevel.MODERATE,
      });
    }

    return steps;
  }

  /**
   * Generate implementation steps from API specification
   * Requirement 7.2: Map API endpoints to route handlers
   */
  private generateStepsFromAPI(artifact: ParsedArtifact): Step[] {
    const steps: Step[] = [];
    const openapi = artifact.structured?.openapi;

    if (!openapi || !openapi.paths) {
      return steps;
    }

    // Create data model steps
    if (openapi.components?.schemas) {
      steps.push({
        id: 'impl-data-models',
        action: 'Define data models',
        description: 'Create TypeScript interfaces for all data models from OpenAPI schemas',
        dependencies: [],
        complexity: ComplexityLevel.SIMPLE,
      });
    }

    // Create endpoint implementation steps
    const paths = Object.keys(openapi.paths);
    for (let i = 0; i < paths.length; i++) {
      const path = paths[i];
      const pathItem = openapi.paths[path];
      const methods = Object.keys(pathItem).filter((k) =>
        ['get', 'post', 'put', 'patch', 'delete'].includes(k)
      );

      for (const method of methods) {
        const operation = pathItem[method];
        const operationId = operation.operationId || `${method}-${path.replace(/\//g, '-')}`;

        steps.push({
          id: `impl-endpoint-${operationId}`,
          action: `Implement ${method.toUpperCase()} ${path}`,
          description: operation.summary || operation.description || `Implement ${method} endpoint for ${path}`,
          dependencies: openapi.components?.schemas ? ['impl-data-models'] : [],
          complexity: ComplexityLevel.MODERATE,
        });
      }
    }

    return steps;
  }

  /**
   * Generate implementation steps from test strategy
   * Requirement 7.2: Map test strategies to test files
   */
  private generateStepsFromTests(artifact: ParsedArtifact): Step[] {
    const steps: Step[] = [];
    const content = artifact.content.toLowerCase();

    // Check for different test types
    const hasUnitTests = /unit test/i.test(content);
    const hasIntegrationTests = /integration test/i.test(content);
    const hasPropertyTests = /property.*test|property-based/i.test(content);

    if (hasUnitTests) {
      steps.push({
        id: 'impl-unit-tests',
        action: 'Implement unit tests',
        description: 'Create unit tests as specified in test strategy',
        dependencies: [],
        complexity: ComplexityLevel.MODERATE,
      });
    }

    if (hasIntegrationTests) {
      steps.push({
        id: 'impl-integration-tests',
        action: 'Implement integration tests',
        description: 'Create integration tests as specified in test strategy',
        dependencies: hasUnitTests ? ['impl-unit-tests'] : [],
        complexity: ComplexityLevel.COMPLEX,
      });
    }

    if (hasPropertyTests) {
      steps.push({
        id: 'impl-property-tests',
        action: 'Implement property-based tests',
        description: 'Create property-based tests with generators as specified',
        dependencies: [],
        complexity: ComplexityLevel.COMPLEX,
      });
    }

    return steps;
  }

  /**
   * Sort steps by dependencies using topological sort
   * Requirement 7.3: Suggest execution order
   * Task 14.3: Implement dependency analysis
   */
  private sortStepsByDependencies(steps: Step[]): Step[] {
    // Build dependency graph
    const graph = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    const stepMap = new Map<string, Step>();

    for (const step of steps) {
      stepMap.set(step.id, step);
      graph.set(step.id, step.dependencies);
      inDegree.set(step.id, 0);
    }

    // Calculate in-degrees
    for (const step of steps) {
      for (const dep of step.dependencies) {
        if (inDegree.has(dep)) {
          inDegree.set(step.id, (inDegree.get(step.id) || 0) + 1);
        }
      }
    }

    // Topological sort using Kahn's algorithm
    const queue: string[] = [];
    const sorted: Step[] = [];

    // Add nodes with no dependencies
    for (const [id, degree] of inDegree.entries()) {
      if (degree === 0) {
        queue.push(id);
      }
    }

    while (queue.length > 0) {
      const current = queue.shift()!;
      const step = stepMap.get(current);
      if (step) {
        sorted.push(step);
      }

      // Process dependents
      for (const [id, deps] of graph.entries()) {
        if (deps.includes(current)) {
          const newDegree = (inDegree.get(id) || 0) - 1;
          inDegree.set(id, newDegree);
          if (newDegree === 0) {
            queue.push(id);
          }
        }
      }
    }

    // Check for circular dependencies
    if (sorted.length !== steps.length) {
      // Return original order if circular dependencies detected
      return steps;
    }

    return sorted;
  }

  /**
   * Organize steps into phases by complexity
   * Requirement 7.6: Organize into phases with complexity estimates
   */
  private organizeIntoPhases(steps: Step[]): ImplementationPhase[] {
    const phases: ImplementationPhase[] = [];

    // Group by complexity and logical phases
    const foundationSteps = steps.filter(
      (s) =>
        s.complexity === ComplexityLevel.SIMPLE ||
        s.id.includes('data-model') ||
        s.id.includes('interface')
    );

    const coreSteps = steps.filter(
      (s) =>
        s.complexity === ComplexityLevel.MODERATE &&
        !foundationSteps.includes(s) &&
        !s.id.includes('integration') &&
        !s.id.includes('test')
    );

    const integrationSteps = steps.filter(
      (s) => s.id.includes('integration') || s.complexity === ComplexityLevel.COMPLEX
    );

    const testSteps = steps.filter((s) => s.id.includes('test'));

    if (foundationSteps.length > 0) {
      phases.push({
        name: 'Phase 1: Foundation',
        complexity: ComplexityLevel.SIMPLE,
        steps: foundationSteps,
        estimatedTime: this.estimatePhaseTime(foundationSteps),
      });
    }

    if (coreSteps.length > 0) {
      phases.push({
        name: 'Phase 2: Core Implementation',
        complexity: ComplexityLevel.MODERATE,
        steps: coreSteps,
        estimatedTime: this.estimatePhaseTime(coreSteps),
      });
    }

    if (integrationSteps.length > 0) {
      phases.push({
        name: 'Phase 3: Integration',
        complexity: ComplexityLevel.COMPLEX,
        steps: integrationSteps,
        estimatedTime: this.estimatePhaseTime(integrationSteps),
      });
    }

    if (testSteps.length > 0) {
      phases.push({
        name: 'Phase 4: Testing',
        complexity: ComplexityLevel.MODERATE,
        steps: testSteps,
        estimatedTime: this.estimatePhaseTime(testSteps),
      });
    }

    return phases;
  }

  /**
   * Estimate time for a phase based on step complexity
   */
  private estimatePhaseTime(steps: Step[]): string {
    let totalHours = 0;

    for (const step of steps) {
      switch (step.complexity) {
        case ComplexityLevel.SIMPLE:
          totalHours += 2;
          break;
        case ComplexityLevel.MODERATE:
          totalHours += 4;
          break;
        case ComplexityLevel.COMPLEX:
          totalHours += 8;
          break;
      }
    }

    if (totalHours < 8) {
      return `${totalHours} hours`;
    } else {
      const days = Math.ceil(totalHours / 8);
      return `${days} day${days > 1 ? 's' : ''}`;
    }
  }

  /**
   * Generate code mappings from artifacts
   * Requirement 7.2: Map artifacts to code locations
   * Task 14.2: Implement artifact-to-code mapping
   */
  private generateCodeMappings(artifacts: ParsedArtifact[]): CodeMapping[] {
    const mappings: CodeMapping[] = [];

    for (const artifact of artifacts) {
      switch (artifact.type) {
        case ArtifactType.MERMAID_DIAGRAM:
          mappings.push(...this.mapDiagramToCode(artifact));
          break;

        case ArtifactType.OPENAPI_SPEC:
          mappings.push(...this.mapAPIToCode(artifact));
          break;

        case ArtifactType.TEST_STRATEGY:
          mappings.push(...this.mapTestsToCode(artifact));
          break;
      }
    }

    return mappings;
  }

  /**
   * Map architecture diagram components to file paths
   */
  private mapDiagramToCode(artifact: ParsedArtifact): CodeMapping[] {
    const mappings: CodeMapping[] = [];
    const mermaid = artifact.structured?.mermaid;

    if (!mermaid || !mermaid.nodes) {
      return mappings;
    }

    for (const node of mermaid.nodes) {
      const componentName = this.toComponentName(node.label || node.id);
      mappings.push({
        artifactReference: `Architecture diagram node: ${node.id}`,
        targetFilePath: `src/components/${componentName}.ts`,
        codeType: 'class',
        boilerplate: this.generateClassBoilerplate(componentName, node.label),
      });
    }

    return mappings;
  }

  /**
   * Map API endpoints to route handlers
   */
  private mapAPIToCode(artifact: ParsedArtifact): CodeMapping[] {
    const mappings: CodeMapping[] = [];
    const openapi = artifact.structured?.openapi;

    if (!openapi || !openapi.paths) {
      return mappings;
    }

    // Map schemas to interfaces
    if (openapi.components?.schemas) {
      for (const [schemaName, schema] of Object.entries(openapi.components.schemas)) {
        mappings.push({
          artifactReference: `OpenAPI schema: ${schemaName}`,
          targetFilePath: `src/types/${this.toKebabCase(schemaName)}.ts`,
          codeType: 'interface',
          boilerplate: this.generateInterfaceBoilerplate(schemaName, schema),
        });
      }
    }

    // Map endpoints to route handlers
    for (const [path, pathItem] of Object.entries(openapi.paths)) {
      const methods = Object.keys(pathItem).filter((k) =>
        ['get', 'post', 'put', 'patch', 'delete'].includes(k)
      );

      for (const method of methods) {
        const operation = pathItem[method];
        const routeName = this.pathToRouteName(path, method);

        mappings.push({
          artifactReference: `${method.toUpperCase()} ${path}`,
          targetFilePath: `src/routes/${routeName}.ts`,
          codeType: 'route',
          boilerplate: this.generateRouteBoilerplate(path, method, operation),
        });
      }
    }

    return mappings;
  }

  /**
   * Map test strategies to test files
   */
  private mapTestsToCode(artifact: ParsedArtifact): CodeMapping[] {
    const mappings: CodeMapping[] = [];
    const content = artifact.content;

    // Extract test file references or generate default
    const hasUnitTests = /unit test/i.test(content);
    const hasIntegrationTests = /integration test/i.test(content);
    const hasPropertyTests = /property.*test|property-based/i.test(content);

    if (hasUnitTests) {
      mappings.push({
        artifactReference: 'Unit tests from test strategy',
        targetFilePath: 'tests/unit/',
        codeType: 'test',
        boilerplate: this.generateTestBoilerplate('unit'),
      });
    }

    if (hasIntegrationTests) {
      mappings.push({
        artifactReference: 'Integration tests from test strategy',
        targetFilePath: 'tests/integration/',
        codeType: 'test',
        boilerplate: this.generateTestBoilerplate('integration'),
      });
    }

    if (hasPropertyTests) {
      mappings.push({
        artifactReference: 'Property-based tests from test strategy',
        targetFilePath: 'tests/properties/',
        codeType: 'test',
        boilerplate: this.generateTestBoilerplate('property'),
      });
    }

    return mappings;
  }

  /**
   * Generate class boilerplate
   * Requirement 7.4: Generate boilerplate from design specifications
   * Task 14.4: Create code template generator
   */
  private generateClassBoilerplate(className: string, description?: string): string {
    return `/**
 * ${description || className}
 */
export class ${className} {
  constructor() {
    // TODO: Initialize ${className}
  }

  // TODO: Implement ${className} methods
}
`;
  }

  /**
   * Generate interface boilerplate from OpenAPI schema
   */
  private generateInterfaceBoilerplate(name: string, schema: any): string {
    let code = `/**
 * ${schema.description || name}
 */
export interface ${name} {\n`;

    if (schema.properties) {
      for (const [propName, propSchema] of Object.entries(schema.properties)) {
        const prop = propSchema as any;
        const optional = schema.required?.includes(propName) ? '' : '?';
        const type = this.openAPITypeToTS(prop.type);
        const comment = prop.description ? `  /** ${prop.description} */\n` : '';
        code += `${comment}  ${propName}${optional}: ${type};\n`;
      }
    }

    code += '}\n';
    return code;
  }

  /**
   * Generate route handler boilerplate
   */
  private generateRouteBoilerplate(path: string, method: string, operation: any): string {
    const handlerName = this.pathToHandlerName(path, method);
    const summary = operation.summary || `Handle ${method.toUpperCase()} ${path}`;

    return `/**
 * ${summary}
 */
export async function ${handlerName}(req: Request, res: Response): Promise<void> {
  try {
    // TODO: Implement ${method.toUpperCase()} ${path}
    
    res.status(200).json({ message: 'Not implemented' });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
}
`;
  }

  /**
   * Generate test boilerplate
   */
  private generateTestBoilerplate(testType: string): string {
    switch (testType) {
      case 'unit':
        return `import { describe, it, expect } from 'vitest';

describe('Component', () => {
  it('should pass unit test', () => {
    // TODO: Implement unit test
    expect(true).toBe(true);
  });
});
`;

      case 'integration':
        return `import { describe, it, expect } from 'vitest';

describe('Integration', () => {
  it('should pass integration test', () => {
    // TODO: Implement integration test
    expect(true).toBe(true);
  });
});
`;

      case 'property':
        return `import { describe, it } from 'vitest';
import * as fc from 'fast-check';

describe('Properties', () => {
  it('should satisfy property', () => {
    fc.assert(
      fc.property(fc.integer(), (value) => {
        // TODO: Implement property test
        return true;
      })
    );
  });
});
`;

      default:
        return '// TODO: Implement test\n';
    }
  }

  /**
   * Extract prerequisites from artifacts
   */
  private extractPrerequisites(artifacts: ParsedArtifact[]): string[] {
    const prerequisites: string[] = [];

    for (const artifact of artifacts) {
      if (artifact.type === ArtifactType.OPENAPI_SPEC) {
        prerequisites.push('Node.js and npm installed');
        prerequisites.push('TypeScript configured');
        prerequisites.push('Express or similar web framework');
      }

      if (artifact.type === ArtifactType.TEST_STRATEGY) {
        prerequisites.push('Testing framework (Vitest/Jest) configured');
        if (/property.*test/i.test(artifact.content)) {
          prerequisites.push('fast-check library for property-based testing');
        }
      }
    }

    return [...new Set(prerequisites)]; // Remove duplicates
  }

  /**
   * Identify implementation risks
   */
  private identifyRisks(artifacts: ParsedArtifact[], steps: Step[]): Array<{ risk: string; mitigation: string }> {
    const risks: Array<{ risk: string; mitigation: string }> = [];

    // Check for complex dependencies
    const complexSteps = steps.filter((s) => s.complexity === ComplexityLevel.COMPLEX);
    if (complexSteps.length > steps.length / 2) {
      risks.push({
        risk: 'High proportion of complex implementation steps',
        mitigation: 'Break down complex steps into smaller, manageable tasks. Consider pair programming for complex components.',
      });
    }

    // Check for circular dependencies
    const hasCycles = this.detectCycles(steps);
    if (hasCycles) {
      risks.push({
        risk: 'Circular dependencies detected in implementation steps',
        mitigation: 'Refactor dependencies to create a clear execution order. Consider introducing interfaces to break cycles.',
      });
    }

    // Check for missing test coverage
    const hasTests = artifacts.some((a) => a.type === ArtifactType.TEST_STRATEGY);
    if (!hasTests) {
      risks.push({
        risk: 'No test strategy defined',
        mitigation: 'Define comprehensive test strategy including unit, integration, and property-based tests.',
      });
    }

    return risks;
  }

  /**
   * Detect cycles in step dependencies
   */
  private detectCycles(steps: Step[]): boolean {
    const graph = new Map<string, string[]>();
    for (const step of steps) {
      graph.set(step.id, step.dependencies);
    }

    const visited = new Set<string>();
    const recStack = new Set<string>();

    const hasCycle = (node: string): boolean => {
      visited.add(node);
      recStack.add(node);

      const neighbors = graph.get(node) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          if (hasCycle(neighbor)) {
            return true;
          }
        } else if (recStack.has(neighbor)) {
          return true;
        }
      }

      recStack.delete(node);
      return false;
    };

    for (const step of steps) {
      if (!visited.has(step.id)) {
        if (hasCycle(step.id)) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Generate overview from artifacts
   */
  private generateOverview(artifacts: ParsedArtifact[]): string {
    const types = new Set(artifacts.map((a) => a.type));
    const parts: string[] = [];

    parts.push('This implementation guide provides step-by-step instructions for implementing the system based on the provided artifacts.');

    if (types.has(ArtifactType.MERMAID_DIAGRAM)) {
      parts.push('The architecture diagram defines the system components and their relationships.');
    }

    if (types.has(ArtifactType.OPENAPI_SPEC)) {
      parts.push('The API specification defines the endpoints, data models, and integration contracts.');
    }

    if (types.has(ArtifactType.TEST_STRATEGY)) {
      parts.push('The test strategy outlines the testing approach and coverage requirements.');
    }

    parts.push('Follow the phases in order, completing all steps in each phase before proceeding to the next.');

    return parts.join(' ');
  }

  /**
   * Convert label to component name (PascalCase)
   */
  private toComponentName(label: string): string {
    return label
      .split(/[\s_-]+/)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join('');
  }

  /**
   * Convert name to kebab-case
   */
  private toKebabCase(name: string): string {
    return name
      .replace(/([a-z])([A-Z])/g, '$1-$2')
      .replace(/[\s_]+/g, '-')
      .toLowerCase();
  }

  /**
   * Convert API path to route name
   */
  private pathToRouteName(path: string, method: string): string {
    const cleanPath = path.replace(/^\//, '').replace(/\//g, '-').replace(/[{}]/g, '');
    return `${method}-${cleanPath || 'root'}`;
  }

  /**
   * Convert API path to handler function name
   */
  private pathToHandlerName(path: string, method: string): string {
    const parts = path.split('/').filter((p) => p && !p.startsWith('{'));
    const pathPart = parts.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join('');
    return `${method}${pathPart || 'Root'}`;
  }

  /**
   * Convert OpenAPI type to TypeScript type
   */
  private openAPITypeToTS(type: string): string {
    switch (type) {
      case 'integer':
      case 'number':
        return 'number';
      case 'string':
        return 'string';
      case 'boolean':
        return 'boolean';
      case 'array':
        return 'any[]';
      case 'object':
        return 'Record<string, any>';
      default:
        return 'any';
    }
  }

  /**
   * Export implementation guide as markdown
   */
  public exportAsMarkdown(guide: ImplementationGuide): string {
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

    return md;
  }
}
