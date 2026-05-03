/**
 * Artifact Validator Component
 * Implements Task 12 - Artifact Validation and Completeness Checking
 * Requirements: 6.1-6.7, 12.1-12.7
 */

import {
  ParsedArtifact,
  ArtifactType,
  ValidationResult,
  ValidationError,
  MermaidAST,
  OpenAPISpec,
  Step,
} from '../types/core.js';

/**
 * Quality dimension scores
 */
export interface QualityScores {
  completeness: number; // 0-100
  clarity: number; // 0-100
  implementability: number; // 0-100
}

/**
 * Validation configuration
 */
export interface ValidationConfig {
  qualityThreshold: number; // Default: 70
  strictMode: boolean; // Default: false
}

/**
 * Artifact Validator Component
 * Validates Opus-generated artifacts for completeness and quality
 */
export class ArtifactValidator {
  private config: ValidationConfig;

  constructor(config: Partial<ValidationConfig> = {}) {
    this.config = {
      qualityThreshold: config.qualityThreshold ?? 70,
      strictMode: config.strictMode ?? false,
    };
  }

  /**
   * Validate a parsed artifact
   * Requirement 6.6: Provide completeness score (0-100%)
   */
  public validate(artifact: ParsedArtifact): ValidationResult {
    switch (artifact.type) {
      case ArtifactType.MERMAID_DIAGRAM:
        return this.validateMermaidDiagram(artifact);
      case ArtifactType.OPENAPI_SPEC:
        return this.validateOpenAPISpec(artifact);
      case ArtifactType.IMPLEMENTATION_GUIDE:
        return this.validateImplementationGuide(artifact);
      case ArtifactType.TEST_STRATEGY:
        return this.validateTestStrategy(artifact);
      default:
        return this.createDefaultValidation();
    }
  }

  /**
   * Validate architecture diagram
   * Requirement 6.1: Check all nodes have descriptions, all edges have labels
   * Task 12.1: Create architecture diagram validator
   */
  private validateMermaidDiagram(artifact: ParsedArtifact): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: string[] = [];
    const followUpQuestions: string[] = [];

    const mermaid = artifact.structured?.mermaid;
    if (!mermaid) {
      errors.push({
        type: 'missing_structure',
        message: 'Mermaid diagram structure not found',
        severity: 'error',
      });
      return this.createValidationResult(0, { completeness: 0, clarity: 0, implementability: 0 }, errors, warnings, followUpQuestions);
    }

    let completenessScore = 100;
    let clarityScore = 100;
    let implementabilityScore = 100;

    // Check all nodes have descriptions (labels)
    const nodesWithoutLabels = mermaid.nodes.filter((node) => !node.label || node.label.trim() === '');
    if (nodesWithoutLabels.length > 0) {
      const penalty = (nodesWithoutLabels.length / mermaid.nodes.length) * 30;
      completenessScore -= penalty;
      clarityScore -= penalty;

      errors.push({
        type: 'missing_node_labels',
        message: `${nodesWithoutLabels.length} node(s) missing labels: ${nodesWithoutLabels.map((n) => n.id).join(', ')}`,
        severity: 'error',
      });

      followUpQuestions.push(
        `Please provide descriptive labels for the following nodes: ${nodesWithoutLabels.map((n) => n.id).join(', ')}`
      );
    }

    // Check all edges have labels
    const edgesWithoutLabels = mermaid.edges.filter((edge) => !edge.label || edge.label.trim() === '');
    if (edgesWithoutLabels.length > 0) {
      const penalty = (edgesWithoutLabels.length / Math.max(mermaid.edges.length, 1)) * 20;
      completenessScore -= penalty;
      clarityScore -= penalty;

      warnings.push(
        `${edgesWithoutLabels.length} edge(s) missing labels. Consider adding labels to clarify relationships.`
      );

      followUpQuestions.push(
        `Please add labels to edges to clarify the relationships between components.`
      );
    }

    // Detect orphan nodes (nodes with no connections)
    const connectedNodeIds = new Set<string>();
    mermaid.edges.forEach((edge) => {
      connectedNodeIds.add(edge.from);
      connectedNodeIds.add(edge.to);
    });

    const orphanNodes = mermaid.nodes.filter((node) => !connectedNodeIds.has(node.id));
    if (orphanNodes.length > 0) {
      const penalty = (orphanNodes.length / mermaid.nodes.length) * 15;
      implementabilityScore -= penalty;

      warnings.push(
        `${orphanNodes.length} orphan node(s) detected: ${orphanNodes.map((n) => n.id).join(', ')}. These nodes have no connections.`
      );

      followUpQuestions.push(
        `The following nodes are not connected to any other components: ${orphanNodes.map((n) => n.id).join(', ')}. Should they be connected, or are they standalone?`
      );
    }

    // Verify naming consistency (check for similar but different names)
    const namingIssues = this.checkNamingConsistency(mermaid.nodes.map((n) => n.id));
    if (namingIssues.length > 0) {
      clarityScore -= 10;
      warnings.push(...namingIssues);
    }

    // Check for minimum complexity (at least 2 nodes and 1 edge for meaningful diagram)
    if (mermaid.nodes.length < 2) {
      completenessScore -= 30;
      warnings.push('Diagram has fewer than 2 nodes. Consider adding more components for a complete architecture.');
    }

    if (mermaid.edges.length < 1 && mermaid.nodes.length >= 2) {
      completenessScore -= 20;
      warnings.push('Diagram has no edges. Consider adding relationships between components.');
    }

    const qualityScores: QualityScores = {
      completeness: Math.max(0, completenessScore),
      clarity: Math.max(0, clarityScore),
      implementability: Math.max(0, implementabilityScore),
    };

    const overallScore = (qualityScores.completeness + qualityScores.clarity + qualityScores.implementability) / 3;

    return this.createValidationResult(overallScore, qualityScores, errors, warnings, followUpQuestions);
  }

  /**
   * Validate API specification
   * Requirement 6.2: Check all endpoints have request/response schemas, error codes
   * Task 12.2: Create API specification validator
   */
  private validateOpenAPISpec(artifact: ParsedArtifact): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: string[] = [];
    const followUpQuestions: string[] = [];

    const openapi = artifact.structured?.openapi;
    if (!openapi) {
      errors.push({
        type: 'missing_structure',
        message: 'OpenAPI specification structure not found',
        severity: 'error',
      });
      return this.createValidationResult(0, { completeness: 0, clarity: 0, implementability: 0 }, errors, warnings, followUpQuestions);
    }

    let completenessScore = 100;
    let clarityScore = 100;
    let implementabilityScore = 100;

    // Check all endpoints have request/response schemas
    const paths = openapi.paths || {};
    const pathKeys = Object.keys(paths);

    if (pathKeys.length === 0) {
      completenessScore -= 50;
      errors.push({
        type: 'missing_endpoints',
        message: 'No API endpoints defined',
        severity: 'error',
      });
      followUpQuestions.push('Please define at least one API endpoint with request and response schemas.');
    }

    let endpointsWithoutSchemas = 0;
    let endpointsWithoutErrorResponses = 0;
    let endpointsWithoutAuth = 0;
    let endpointsWithoutExamples = 0;

    for (const path of pathKeys) {
      const pathItem = paths[path];
      const methods = ['get', 'post', 'put', 'patch', 'delete'];

      for (const method of methods) {
        const operation = pathItem[method];
        if (!operation) continue;

        // Check for request/response schemas
        const hasRequestSchema = this.hasRequestSchema(operation, method);
        const hasResponseSchema = this.hasResponseSchema(operation);

        if (!hasRequestSchema && ['post', 'put', 'patch'].includes(method)) {
          endpointsWithoutSchemas++;
          warnings.push(`${method.toUpperCase()} ${path}: Missing request body schema`);
        }

        if (!hasResponseSchema) {
          endpointsWithoutSchemas++;
          warnings.push(`${method.toUpperCase()} ${path}: Missing response schema`);
        }

        // Check for error responses (400, 500)
        const responses = operation.responses || {};
        const has400 = '400' in responses || '4XX' in responses;
        const has500 = '500' in responses || '5XX' in responses;

        if (!has400 || !has500) {
          endpointsWithoutErrorResponses++;
          warnings.push(`${method.toUpperCase()} ${path}: Missing error response definitions (400, 500)`);
        }

        // Check for authentication requirements
        const hasSecurity = operation.security || openapi.security;
        if (!hasSecurity) {
          endpointsWithoutAuth++;
        }

        // Check for examples
        const hasExamples = this.hasExamples(operation);
        if (!hasExamples) {
          endpointsWithoutExamples++;
        }
      }
    }

    const totalEndpoints = this.countEndpoints(paths);

    if (endpointsWithoutSchemas > 0) {
      const penalty = (endpointsWithoutSchemas / totalEndpoints) * 30;
      completenessScore -= penalty;
      followUpQuestions.push('Please add request and response schemas for all endpoints.');
    }

    if (endpointsWithoutErrorResponses > 0) {
      const penalty = (endpointsWithoutErrorResponses / totalEndpoints) * 20;
      completenessScore -= penalty;
      followUpQuestions.push('Please define error responses (400, 500) for all endpoints.');
    }

    if (endpointsWithoutAuth > 0) {
      const penalty = (endpointsWithoutAuth / totalEndpoints) * 15;
      implementabilityScore -= penalty;
      warnings.push(`${endpointsWithoutAuth} endpoint(s) missing authentication requirements. Consider adding security definitions.`);
    }

    if (endpointsWithoutExamples > 0) {
      const penalty = (endpointsWithoutExamples / totalEndpoints) * 10;
      clarityScore -= penalty;
      warnings.push(`${endpointsWithoutExamples} endpoint(s) missing examples. Consider adding examples for complex schemas.`);
    }

    // Check for API documentation
    if (!openapi.info?.description || openapi.info.description.trim().length < 20) {
      clarityScore -= 10;
      warnings.push('API description is missing or too brief. Consider adding a comprehensive description.');
    }

    const qualityScores: QualityScores = {
      completeness: Math.max(0, completenessScore),
      clarity: Math.max(0, clarityScore),
      implementability: Math.max(0, implementabilityScore),
    };

    const overallScore = (qualityScores.completeness + qualityScores.clarity + qualityScores.implementability) / 3;

    return this.createValidationResult(overallScore, qualityScores, errors, warnings, followUpQuestions);
  }

  /**
   * Validate implementation plan
   * Requirement 6.3: Check each step has clear action, dependencies
   * Task 12.3: Create implementation plan validator
   */
  private validateImplementationGuide(artifact: ParsedArtifact): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: string[] = [];
    const followUpQuestions: string[] = [];

    const steps = artifact.structured?.implementationSteps;
    if (!steps || steps.length === 0) {
      errors.push({
        type: 'missing_steps',
        message: 'No implementation steps found',
        severity: 'error',
      });
      return this.createValidationResult(0, { completeness: 0, clarity: 0, implementability: 0 }, errors, warnings, followUpQuestions);
    }

    let completenessScore = 100;
    let clarityScore = 100;
    let implementabilityScore = 100;

    // Check each step has clear action verb
    const stepsWithoutActionVerbs = steps.filter((step) => !this.hasActionVerb(step.action));
    if (stepsWithoutActionVerbs.length > 0) {
      const penalty = (stepsWithoutActionVerbs.length / steps.length) * 25;
      clarityScore -= penalty;

      warnings.push(
        `${stepsWithoutActionVerbs.length} step(s) missing clear action verbs. Use verbs like "Create", "Implement", "Configure", etc.`
      );

      followUpQuestions.push(
        'Please rephrase steps to start with clear action verbs (e.g., "Create", "Implement", "Configure").'
      );
    }

    // Check dependencies are explicitly stated
    const stepsWithoutDependencies = steps.filter((step, index) => index > 0 && step.dependencies.length === 0);
    if (stepsWithoutDependencies.length > steps.length / 2) {
      implementabilityScore -= 15;
      warnings.push('Many steps have no explicit dependencies. Consider adding dependency information for proper sequencing.');
    }

    // Detect circular dependencies
    const circularDeps = this.detectCircularDependencies(steps);
    if (circularDeps.length > 0) {
      implementabilityScore -= 30;
      errors.push({
        type: 'circular_dependencies',
        message: `Circular dependencies detected: ${circularDeps.join(' -> ')}`,
        severity: 'error',
      });
      followUpQuestions.push('Please resolve circular dependencies in the implementation steps.');
    }

    // Check complexity estimates are present
    const stepsWithoutComplexity = steps.filter((step) => !step.complexity);
    if (stepsWithoutComplexity.length > 0) {
      const penalty = (stepsWithoutComplexity.length / steps.length) * 10;
      completenessScore -= penalty;
      warnings.push(`${stepsWithoutComplexity.length} step(s) missing complexity estimates.`);
    }

    // Check for step descriptions
    const stepsWithoutDescriptions = steps.filter((step) => !step.description || step.description.trim().length < 10);
    if (stepsWithoutDescriptions.length > 0) {
      const penalty = (stepsWithoutDescriptions.length / steps.length) * 20;
      clarityScore -= penalty;
      warnings.push(`${stepsWithoutDescriptions.length} step(s) have insufficient descriptions.`);
      followUpQuestions.push('Please add detailed descriptions for all implementation steps.');
    }

    // Check for minimum number of steps
    if (steps.length < 3) {
      completenessScore -= 20;
      warnings.push('Implementation guide has fewer than 3 steps. Consider breaking down into more granular steps.');
    }

    const qualityScores: QualityScores = {
      completeness: Math.max(0, completenessScore),
      clarity: Math.max(0, clarityScore),
      implementability: Math.max(0, implementabilityScore),
    };

    const overallScore = (qualityScores.completeness + qualityScores.clarity + qualityScores.implementability) / 3;

    return this.createValidationResult(overallScore, qualityScores, errors, warnings, followUpQuestions);
  }

  /**
   * Validate test strategy
   * Requirement 6.4: Check coverage targets, property-based tests, edge cases
   * Task 12.4: Create test strategy validator
   */
  private validateTestStrategy(artifact: ParsedArtifact): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: string[] = [];
    const followUpQuestions: string[] = [];

    const content = artifact.content.toLowerCase();

    let completenessScore = 100;
    let clarityScore = 100;
    let implementabilityScore = 100;

    // Check coverage targets specified
    const hasCoverageTarget = /coverage.*\d+%|target.*\d+%|\d+%.*coverage/i.test(content);
    if (!hasCoverageTarget) {
      completenessScore -= 20;
      warnings.push('No coverage targets specified. Consider adding target coverage percentages.');
      followUpQuestions.push('What are the target coverage percentages for unit, integration, and end-to-end tests?');
    }

    // Check property-based tests include generators
    const hasPropertyTests = /property.*test|property-based|invariant|metamorphic/i.test(content);
    if (hasPropertyTests) {
      const hasGenerators = /generator|arbitrary|strategy.*generat/i.test(content);
      if (!hasGenerators) {
        completenessScore -= 15;
        warnings.push('Property-based tests mentioned but no test data generators specified.');
        followUpQuestions.push('Please specify test data generators for property-based tests.');
      }
    } else {
      completenessScore -= 25;
      warnings.push('No property-based tests specified. Consider adding property-based testing for critical invariants.');
      followUpQuestions.push('What invariants or properties should be tested with property-based testing?');
    }

    // Check edge cases identified
    const hasEdgeCases = /edge case|boundary|corner case|limit|extreme/i.test(content);
    if (!hasEdgeCases) {
      completenessScore -= 20;
      warnings.push('No edge cases identified. Consider specifying boundary conditions and corner cases.');
      followUpQuestions.push('What edge cases and boundary conditions should be tested?');
    }

    // Check test data requirements defined
    const hasTestData = /test data|fixture|mock|stub|sample data/i.test(content);
    if (!hasTestData) {
      implementabilityScore -= 15;
      warnings.push('Test data requirements not clearly defined.');
      followUpQuestions.push('What test data, fixtures, or mocks are required for testing?');
    }

    // Check for different test types
    const hasUnitTests = /unit test/i.test(content);
    const hasIntegrationTests = /integration test/i.test(content);
    const hasE2ETests = /e2e|end-to-end|end to end/i.test(content);

    let testTypesCount = 0;
    if (hasUnitTests) testTypesCount++;
    if (hasIntegrationTests) testTypesCount++;
    if (hasE2ETests) testTypesCount++;

    if (testTypesCount === 0) {
      completenessScore -= 30;
      errors.push({
        type: 'missing_test_types',
        message: 'No test types specified (unit, integration, e2e)',
        severity: 'error',
      });
      followUpQuestions.push('Please specify which types of tests are needed: unit, integration, and/or end-to-end tests.');
    } else if (testTypesCount === 1) {
      completenessScore -= 15;
      warnings.push('Only one test type specified. Consider adding multiple test levels for comprehensive coverage.');
    }

    // Check for test organization
    const hasTestOrganization = /test suite|test file|test structure|organize|directory/i.test(content);
    if (!hasTestOrganization) {
      clarityScore -= 10;
      warnings.push('Test organization not specified. Consider describing test file structure and naming conventions.');
    }

    const qualityScores: QualityScores = {
      completeness: Math.max(0, completenessScore),
      clarity: Math.max(0, clarityScore),
      implementability: Math.max(0, implementabilityScore),
    };

    const overallScore = (qualityScores.completeness + qualityScores.clarity + qualityScores.implementability) / 3;

    return this.createValidationResult(overallScore, qualityScores, errors, warnings, followUpQuestions);
  }

  /**
   * Check naming consistency across node IDs
   */
  private checkNamingConsistency(nodeIds: string[]): string[] {
    const issues: string[] = [];
    const normalized = new Map<string, string[]>();

    // Group similar names (case-insensitive, ignoring underscores/hyphens)
    for (const id of nodeIds) {
      const key = id.toLowerCase().replace(/[-_]/g, '');
      if (!normalized.has(key)) {
        normalized.set(key, []);
      }
      normalized.get(key)!.push(id);
    }

    // Find groups with multiple variations
    for (const [key, ids] of normalized.entries()) {
      if (ids.length > 1) {
        issues.push(`Inconsistent naming detected: ${ids.join(', ')}. Consider using consistent naming conventions.`);
      }
    }

    return issues;
  }

  /**
   * Check if operation has request schema
   */
  private hasRequestSchema(operation: any, method: string): boolean {
    if (!['post', 'put', 'patch'].includes(method)) {
      return true; // GET/DELETE don't need request bodies
    }

    return !!(operation.requestBody?.content);
  }

  /**
   * Check if operation has response schema
   */
  private hasResponseSchema(operation: any): boolean {
    const responses = operation.responses || {};
    const successCodes = ['200', '201', '204', '2XX'];

    for (const code of successCodes) {
      if (responses[code]?.content) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if operation has examples
   */
  private hasExamples(operation: any): boolean {
    // Check request body examples
    const requestBody = operation.requestBody?.content;
    if (requestBody) {
      for (const mediaType of Object.values(requestBody)) {
        if ((mediaType as any).example || (mediaType as any).examples) {
          return true;
        }
      }
    }

    // Check response examples
    const responses = operation.responses || {};
    for (const response of Object.values(responses)) {
      const content = (response as any).content;
      if (content) {
        for (const mediaType of Object.values(content)) {
          if ((mediaType as any).example || (mediaType as any).examples) {
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * Count total endpoints in OpenAPI spec
   */
  private countEndpoints(paths: Record<string, any>): number {
    let count = 0;
    const methods = ['get', 'post', 'put', 'patch', 'delete'];

    for (const path of Object.values(paths)) {
      for (const method of methods) {
        if (path[method]) {
          count++;
        }
      }
    }

    return count;
  }

  /**
   * Check if action starts with action verb
   */
  private hasActionVerb(action: string): boolean {
    const actionVerbs = [
      'create', 'implement', 'configure', 'setup', 'install', 'build', 'deploy',
      'test', 'validate', 'verify', 'add', 'update', 'modify', 'remove', 'delete',
      'define', 'design', 'develop', 'write', 'generate', 'initialize', 'prepare',
      'integrate', 'connect', 'enable', 'disable', 'start', 'stop', 'run', 'execute'
    ];

    const firstWord = action.trim().toLowerCase().split(/\s+/)[0];
    return actionVerbs.includes(firstWord);
  }

  /**
   * Detect circular dependencies in steps
   */
  private detectCircularDependencies(steps: Step[]): string[] {
    const graph = new Map<string, string[]>();

    // Build dependency graph
    for (const step of steps) {
      graph.set(step.id, step.dependencies);
    }

    // DFS to detect cycles
    const visited = new Set<string>();
    const recStack = new Set<string>();
    const cyclePath: string[] = [];

    const hasCycle = (node: string, path: string[]): boolean => {
      visited.add(node);
      recStack.add(node);
      path.push(node);

      const neighbors = graph.get(node) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          if (hasCycle(neighbor, path)) {
            return true;
          }
        } else if (recStack.has(neighbor)) {
          // Cycle detected
          const cycleStart = path.indexOf(neighbor);
          cyclePath.push(...path.slice(cycleStart), neighbor);
          return true;
        }
      }

      path.pop();
      recStack.delete(node);
      return false;
    };

    for (const step of steps) {
      if (!visited.has(step.id)) {
        if (hasCycle(step.id, [])) {
          return cyclePath;
        }
      }
    }

    return [];
  }

  /**
   * Create validation result
   * Requirement 6.7: Completeness score ≥ 80% indicates ready for implementation
   */
  private createValidationResult(
    overallScore: number,
    qualityScores: QualityScores,
    errors: ValidationError[],
    warnings: string[],
    followUpQuestions: string[]
  ): ValidationResult {
    const isValid = errors.length === 0 && overallScore >= this.config.qualityThreshold;

    return {
      completenessScore: overallScore,
      qualityScores,
      isValid,
      errors,
      warnings,
      followUpQuestions,
    };
  }

  /**
   * Create default validation for unsupported artifact types
   */
  private createDefaultValidation(): ValidationResult {
    return {
      completenessScore: 100,
      qualityScores: {
        completeness: 100,
        clarity: 100,
        implementability: 100,
      },
      isValid: true,
      errors: [],
      warnings: [],
      followUpQuestions: [],
    };
  }

  /**
   * Validate multiple artifacts and return aggregate results
   */
  public validateAll(artifacts: ParsedArtifact[]): ValidationResult[] {
    return artifacts.map((artifact) => this.validate(artifact));
  }

  /**
   * Get overall session validation summary
   */
  public getSessionSummary(validationResults: ValidationResult[]): {
    averageCompleteness: number;
    averageQuality: QualityScores;
    totalErrors: number;
    totalWarnings: number;
    readyForImplementation: boolean;
  } {
    if (validationResults.length === 0) {
      return {
        averageCompleteness: 0,
        averageQuality: { completeness: 0, clarity: 0, implementability: 0 },
        totalErrors: 0,
        totalWarnings: 0,
        readyForImplementation: false,
      };
    }

    const avgCompleteness = validationResults.reduce((sum, r) => sum + r.completenessScore, 0) / validationResults.length;

    const avgQuality: QualityScores = {
      completeness: validationResults.reduce((sum, r) => sum + r.qualityScores.completeness, 0) / validationResults.length,
      clarity: validationResults.reduce((sum, r) => sum + r.qualityScores.clarity, 0) / validationResults.length,
      implementability: validationResults.reduce((sum, r) => sum + r.qualityScores.implementability, 0) / validationResults.length,
    };

    const totalErrors = validationResults.reduce((sum, r) => sum + r.errors.length, 0);
    const totalWarnings = validationResults.reduce((sum, r) => sum + r.warnings.length, 0);

    const readyForImplementation = totalErrors === 0 && avgCompleteness >= this.config.qualityThreshold;

    return {
      averageCompleteness: avgCompleteness,
      averageQuality: avgQuality,
      totalErrors,
      totalWarnings,
      readyForImplementation,
    };
  }
}
