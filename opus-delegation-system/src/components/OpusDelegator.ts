/**
 * Opus Delegator Component
 * Implements Task 9 - Delegation request generation, multi-round session management, follow-up generation
 * Requirements: 3.1-3.7, 9.1-9.7
 */

import { 
  DelegationType, 
  DelegationSession, 
  DelegationRound, 
  ParsedArtifact,
  ValidationResult,
  ComplexityLevel,
  ArtifactType
} from '../types/core.js';
import { TemplateLibrary, DelegationTemplate } from './TemplateLibrary.js';
import { ContextBundle } from './ContextPackager.js';

// Delegation Request Structure (Requirement 3.1-3.6)
export interface DelegationRequest {
  id: string;
  sessionId: string;
  roundNumber: number;
  problemDescription: string;
  objectives: string[];
  constraints: string[];
  expectedArtifacts: ExpectedArtifact[];
  outputFormatRequirements: OutputFormatRequirement[];
  contextBundle: string; // Markdown formatted context
  questionsToAddress: string[];
  previousRoundSummary?: string; // For multi-round sessions
  artifactReferences?: string[]; // References to previous artifacts
  generatedAt: Date;
}

// Expected artifact specification
export interface ExpectedArtifact {
  type: ArtifactType;
  description: string;
  format: string; // e.g., "Mermaid diagram", "OpenAPI YAML", "Markdown"
  structureGuidance: string;
}

// Output format requirements
export interface OutputFormatRequirement {
  artifactType: ArtifactType;
  format: string;
  example?: string;
  instructions: string;
}

// Session state for multi-round tracking (Requirement 9.1-9.5)
export interface SessionState {
  session: DelegationSession;
  currentRound: number;
  conversationContext: ConversationContext;
  artifactVersionHistory: Map<string, ArtifactVersion[]>;
  completionCriteria: CompletionCriteria;
}

// Conversation context across rounds
export interface ConversationContext {
  previousQuestions: string[];
  previousResponses: string[];
  refinementRequests: string[];
  clarificationHistory: Array<{
    question: string;
    answer: string;
    roundNumber: number;
  }>;
}

// Artifact version tracking
export interface ArtifactVersion {
  version: number;
  artifact: ParsedArtifact;
  roundNumber: number;
  changes?: string;
  timestamp: Date;
}

// Completion criteria for session
export interface CompletionCriteria {
  minCompletenessScore: number; // Default 80%
  minQualityScore: number; // Default 70%
  maxRounds: number; // Default 5
  requiredArtifacts: ArtifactType[];
}

// Follow-up request generation (Requirement 9.6)
export interface FollowUpRequest {
  sessionId: string;
  roundNumber: number;
  incompleteArtifacts: Array<{
    artifactId: string;
    artifactType: ArtifactType;
    missingElements: string[];
    qualityIssues: string[];
  }>;
  clarifyingQuestions: string[];
  refinementRequests: string[];
  artifactReferences: string[];
}

/**
 * Opus Delegator - Orchestrates delegation workflow
 */
export class OpusDelegator {
  private templateLibrary: TemplateLibrary;
  private activeSessions: Map<string, SessionState> = new Map();

  constructor(templateLibrary: TemplateLibrary) {
    this.templateLibrary = templateLibrary;
  }

  /**
   * Generate initial delegation request (Task 9.1)
   * Requirements: 3.1, 3.2, 3.3, 3.6
   */
  public generateDelegationRequest(
    sessionId: string,
    problemTitle: string,
    problemDescription: string,
    delegationType: DelegationType,
    contextBundle: ContextBundle,
    templateId?: string,
    _templateParams?: Record<string, any>
  ): DelegationRequest {
    // Get template or use default
    let template: DelegationTemplate | undefined;
    if (templateId) {
      template = this.templateLibrary.getTemplate(templateId);
      if (!template) {
        throw new Error(`Template not found: ${templateId}`);
      }
    } else {
      // Find template by delegation type
      const templates = this.templateLibrary.listTemplatesByCategory(delegationType);
      template = templates[0]; // Use first matching template
    }

    if (!template) {
      throw new Error(`No template found for delegation type: ${delegationType}`);
    }

    // Extract objectives and constraints from context bundle
    const objectives = this.extractObjectives(problemDescription, template);
    const constraints = contextBundle.constraints;

    // Define expected artifacts based on template
    const expectedArtifacts = this.defineExpectedArtifacts(template);

    // Define output format requirements
    const outputFormatRequirements = this.defineOutputFormatRequirements(template);

    // Generate questions to address
    const questionsToAddress = this.generateInitialQuestions(problemDescription, delegationType);

    // Format context bundle as markdown
    const contextMarkdown = this.formatContextBundle(contextBundle);

    const request: DelegationRequest = {
      id: this.generateRequestId(),
      sessionId,
      roundNumber: 1,
      problemDescription,
      objectives,
      constraints,
      expectedArtifacts,
      outputFormatRequirements,
      contextBundle: contextMarkdown,
      questionsToAddress,
      generatedAt: new Date()
    };

    return request;
  }

  /**
   * Format delegation request as copy-paste ready text (Task 9.1)
   * Requirement: 3.6
   */
  public formatDelegationRequestAsText(request: DelegationRequest): string {
    const sections: string[] = [];

    // Header
    sections.push(`# Delegation Request: Round ${request.roundNumber}\n`);

    // Previous round summary (for multi-round)
    if (request.previousRoundSummary) {
      sections.push(`## Previous Round Summary\n`);
      sections.push(request.previousRoundSummary);
      sections.push('');
    }

    // Objective
    sections.push(`## Objective\n`);
    sections.push(request.problemDescription);
    sections.push('');

    // Objectives list
    if (request.objectives.length > 0) {
      sections.push(`## Goals\n`);
      request.objectives.forEach((obj, i) => {
        sections.push(`${i + 1}. ${obj}`);
      });
      sections.push('');
    }

    // Constraints
    if (request.constraints.length > 0) {
      sections.push(`## Constraints\n`);
      request.constraints.forEach(constraint => {
        sections.push(`- ${constraint}`);
      });
      sections.push('');
    }

    // Expected Artifacts
    sections.push(`## Expected Artifacts\n`);
    sections.push('Please generate the following:\n');
    request.expectedArtifacts.forEach((artifact, i) => {
      sections.push(`${i + 1}. **${artifact.description}** — ${artifact.structureGuidance}`);
    });
    sections.push('');

    // Output Format Requirements
    sections.push(`## Output Format Requirements\n`);
    request.outputFormatRequirements.forEach(req => {
      sections.push(`- **${req.artifactType}**: ${req.instructions}`);
      if (req.example) {
        sections.push(`  Example: ${req.example}`);
      }
    });
    sections.push('');

    // Questions to Address
    if (request.questionsToAddress.length > 0) {
      sections.push(`## Questions to Address\n`);
      request.questionsToAddress.forEach((q, i) => {
        sections.push(`${i + 1}. ${q}`);
      });
      sections.push('');
    }

    // Artifact References (for multi-round)
    if (request.artifactReferences && request.artifactReferences.length > 0) {
      sections.push(`## Previous Artifacts to Refine\n`);
      request.artifactReferences.forEach(ref => {
        sections.push(`- ${ref}`);
      });
      sections.push('');
    }

    // Context Bundle
    sections.push(`## Context\n`);
    sections.push(request.contextBundle);

    return sections.join('\n');
  }

  /**
   * Initialize multi-round session (Task 9.3)
   * Requirement: 9.1
   */
  public initializeSession(
    problemTitle: string,
    problemDescription: string,
    delegationType: DelegationType,
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
  ): DelegationSession {
    const session: DelegationSession = {
      id: this.generateSessionId(),
      createdAt: new Date(),
      updatedAt: new Date(),
      status: 'active',
      problem: {
        title: problemTitle,
        description: problemDescription,
        type: delegationType,
        complexity
      },
      rounds: [],
      finalArtifacts: [],
      metrics: {
        totalTime: 0,
        contextSize: 0,
        roundCount: 0,
        finalCompleteness: 0
      }
    };

    // Initialize session state
    const sessionState: SessionState = {
      session,
      currentRound: 0,
      conversationContext: {
        previousQuestions: [],
        previousResponses: [],
        refinementRequests: [],
        clarificationHistory: []
      },
      artifactVersionHistory: new Map(),
      completionCriteria: {
        minCompletenessScore: 80,
        minQualityScore: 70,
        maxRounds: 5,
        requiredArtifacts: this.getRequiredArtifacts(delegationType)
      }
    };

    this.activeSessions.set(session.id, sessionState);

    return session;
  }

  /**
   * Add round to session (Task 9.3)
   * Requirement: 9.2, 9.3
   */
  public addRound(
    sessionId: string,
    request: string,
    response: string,
    artifacts: ParsedArtifact[],
    validation: ValidationResult
  ): void {
    const sessionState = this.activeSessions.get(sessionId);
    if (!sessionState) {
      throw new Error(`Session not found: ${sessionId}`);
    }

    const roundNumber = sessionState.currentRound + 1;

    const round: DelegationRound = {
      roundNumber,
      request,
      response,
      artifacts,
      validation,
      timestamp: new Date()
    };

    sessionState.session.rounds.push(round);
    sessionState.currentRound = roundNumber;
    sessionState.session.updatedAt = new Date();
    sessionState.session.metrics.roundCount = roundNumber;

    // Update conversation context
    sessionState.conversationContext.previousQuestions.push(
      ...this.extractQuestionsFromRequest(request)
    );
    sessionState.conversationContext.previousResponses.push(response);

    // Update artifact version history (Task 9.3)
    this.updateArtifactVersionHistory(sessionState, artifacts, roundNumber);

    // Update final artifacts if validation passed
    if (validation.isValid) {
      sessionState.session.finalArtifacts = artifacts;
      sessionState.session.metrics.finalCompleteness = validation.completenessScore;
    }
  }

  /**
   * Update artifact version history (Task 9.3)
   * Requirement: 9.4
   */
  private updateArtifactVersionHistory(
    sessionState: SessionState,
    artifacts: ParsedArtifact[],
    roundNumber: number
  ): void {
    for (const artifact of artifacts) {
      const artifactKey = `${artifact.type}`;
      
      if (!sessionState.artifactVersionHistory.has(artifactKey)) {
        sessionState.artifactVersionHistory.set(artifactKey, []);
      }

      const versions = sessionState.artifactVersionHistory.get(artifactKey)!;
      const version: ArtifactVersion = {
        version: versions.length + 1,
        artifact,
        roundNumber,
        timestamp: new Date()
      };

      // Detect changes from previous version
      if (versions.length > 0) {
        const previousVersion = versions[versions.length - 1];
        version.changes = this.detectArtifactChanges(previousVersion.artifact, artifact);
      }

      versions.push(version);
    }
  }

  /**
   * Detect changes between artifact versions
   */
  private detectArtifactChanges(previous: ParsedArtifact, current: ParsedArtifact): string {
    // Simple change detection - in production would use proper diff
    if (previous.content === current.content) {
      return 'No changes';
    }

    const prevLength = previous.content.length;
    const currLength = current.content.length;
    const diff = currLength - prevLength;

    if (diff > 0) {
      return `Added ${diff} characters`;
    } else if (diff < 0) {
      return `Removed ${Math.abs(diff)} characters`;
    } else {
      return 'Content modified';
    }
  }

  /**
   * Generate follow-up request for incomplete artifacts (Task 9.4)
   * Requirement: 9.6
   */
  public generateFollowUpRequest(
    sessionId: string,
    validation: ValidationResult,
    artifacts: ParsedArtifact[]
  ): DelegationRequest {
    const sessionState = this.activeSessions.get(sessionId);
    if (!sessionState) {
      throw new Error(`Session not found: ${sessionId}`);
    }

    const session = sessionState.session;
    const nextRound = sessionState.currentRound + 1;

    // Identify incomplete artifacts
    const incompleteArtifacts = this.identifyIncompleteArtifacts(artifacts, validation);

    // Generate clarifying questions
    const clarifyingQuestions = this.generateClarifyingQuestions(
      validation,
      incompleteArtifacts,
      sessionState
    );

    // Generate refinement requests
    const refinementRequests = this.generateRefinementRequests(
      validation,
      incompleteArtifacts
    );

    // Create artifact references
    const artifactReferences = artifacts.map(a => 
      `${a.type} (version ${this.getArtifactVersion(sessionState, a.type)})`
    );

    // Generate previous round summary
    const previousRoundSummary = this.generateRoundSummary(
      sessionState.currentRound,
      artifacts,
      validation
    );

    const followUpRequest: DelegationRequest = {
      id: this.generateRequestId(),
      sessionId,
      roundNumber: nextRound,
      problemDescription: session.problem.description,
      objectives: this.extractObjectivesFromValidation(validation),
      constraints: [], // Constraints already in context
      expectedArtifacts: this.defineExpectedArtifactsFromIncomplete(incompleteArtifacts),
      outputFormatRequirements: [], // Already defined in initial request
      contextBundle: '', // Will be filled with previous artifacts
      questionsToAddress: [...clarifyingQuestions, ...refinementRequests],
      previousRoundSummary,
      artifactReferences,
      generatedAt: new Date()
    };

    return followUpRequest;
  }

  /**
   * Detect session completion (Task 9.3)
   * Requirement: 9.5
   */
  public detectSessionCompletion(sessionId: string, validation: ValidationResult): boolean {
    const sessionState = this.activeSessions.get(sessionId);
    if (!sessionState) {
      return false;
    }

    const criteria = sessionState.completionCriteria;

    // Check if max rounds reached
    if (sessionState.currentRound >= criteria.maxRounds) {
      return true;
    }

    // Check if completeness and quality thresholds met
    const meetsCompleteness = validation.completenessScore >= criteria.minCompletenessScore;
    const meetsQuality = validation.qualityScores.implementability >= criteria.minQualityScore;

    // Check if all required artifacts present
    const session = sessionState.session;
    const artifactTypes = new Set(
      session.rounds[session.rounds.length - 1]?.artifacts.map(a => a.type) || []
    );
    const hasAllRequired = criteria.requiredArtifacts.every(type => artifactTypes.has(type));

    return meetsCompleteness && meetsQuality && hasAllRequired;
  }

  /**
   * Get session state
   */
  public getSession(sessionId: string): DelegationSession | undefined {
    return this.activeSessions.get(sessionId)?.session;
  }

  /**
   * Get artifact version history
   */
  public getArtifactVersionHistory(
    sessionId: string,
    artifactType: ArtifactType
  ): ArtifactVersion[] {
    const sessionState = this.activeSessions.get(sessionId);
    if (!sessionState) {
      return [];
    }

    return sessionState.artifactVersionHistory.get(artifactType) || [];
  }

  // ========== Helper Methods ==========

  private extractObjectives(description: string, template: DelegationTemplate): string[] {
    // Extract objectives from problem description and template
    const objectives: string[] = [];

    // Add template-specific objectives
    for (const artifact of template.expected_artifacts) {
      objectives.push(`Generate ${artifact.type} artifact`);
    }

    return objectives;
  }

  private defineExpectedArtifacts(template: DelegationTemplate): ExpectedArtifact[] {
    return template.expected_artifacts.map(artifact => ({
      type: artifact.type,
      description: this.getArtifactDescription(artifact.type),
      format: artifact.format || this.getDefaultFormat(artifact.type),
      structureGuidance: this.getStructureGuidance(artifact.type)
    }));
  }

  private defineOutputFormatRequirements(template: DelegationTemplate): OutputFormatRequirement[] {
    return template.expected_artifacts.map(artifact => ({
      artifactType: artifact.type,
      format: artifact.format || this.getDefaultFormat(artifact.type),
      instructions: this.getFormatInstructions(artifact.type)
    }));
  }

  private generateInitialQuestions(description: string, type: DelegationType): string[] {
    const questions: string[] = [];

    switch (type) {
      case DelegationType.ARCHITECTURE_DESIGN:
        questions.push('What are the key components and their responsibilities?');
        questions.push('How do components communicate with each other?');
        questions.push('What are the scalability and performance requirements?');
        break;
      case DelegationType.API_DESIGN:
        questions.push('What are the main API endpoints and their purposes?');
        questions.push('What data models are required?');
        questions.push('What authentication and authorization mechanisms are needed?');
        break;
      case DelegationType.TEST_STRATEGY:
        questions.push('What properties should be verified?');
        questions.push('What edge cases need to be tested?');
        questions.push('What test data generators are required?');
        break;
      default:
        questions.push('What are the key requirements?');
        questions.push('What are the main challenges?');
    }

    return questions;
  }

  private formatContextBundle(bundle: ContextBundle): string {
    // Context bundle is already formatted as markdown by ContextPackager
    return `[Context bundle with ${bundle.codeSnippets.length} code snippets, ${bundle.documentationExcerpts.length} documentation excerpts]`;
  }

  private extractQuestionsFromRequest(request: string): string[] {
    // Simple extraction - look for numbered questions
    const questions: string[] = [];
    const lines = request.split('\n');
    
    for (const line of lines) {
      if (line.match(/^\d+\.\s+.+\?$/)) {
        questions.push(line.replace(/^\d+\.\s+/, ''));
      }
    }

    return questions;
  }

  private identifyIncompleteArtifacts(
    artifacts: ParsedArtifact[],
    validation: ValidationResult
  ): Array<{ artifactId: string; artifactType: ArtifactType; missingElements: string[]; qualityIssues: string[] }> {
    const incomplete: Array<{ artifactId: string; artifactType: ArtifactType; missingElements: string[]; qualityIssues: string[] }> = [];

    for (const artifact of artifacts) {
      const missingElements: string[] = [];
      const qualityIssues: string[] = [];

      // Extract issues from validation errors
      for (const error of validation.errors) {
        if (error.location?.includes(artifact.id)) {
          if (error.message.includes('missing')) {
            missingElements.push(error.message);
          } else {
            qualityIssues.push(error.message);
          }
        }
      }

      if (missingElements.length > 0 || qualityIssues.length > 0) {
        incomplete.push({
          artifactId: artifact.id,
          artifactType: artifact.type,
          missingElements,
          qualityIssues
        });
      }
    }

    return incomplete;
  }

  private generateClarifyingQuestions(
    validation: ValidationResult,
    incompleteArtifacts: Array<{ artifactId: string; artifactType: ArtifactType; missingElements: string[]; qualityIssues: string[] }>,
    _sessionState: SessionState
  ): string[] {
    const questions: string[] = [];

    // Generate questions from validation follow-ups
    questions.push(...validation.followUpQuestions);

    // Generate questions for incomplete artifacts
    for (const incomplete of incompleteArtifacts) {
      if (incomplete.missingElements.length > 0) {
        questions.push(
          `For ${incomplete.artifactType}, please provide: ${incomplete.missingElements.join(', ')}`
        );
      }
    }

    return questions;
  }

  private generateRefinementRequests(
    validation: ValidationResult,
    incompleteArtifacts: Array<{ artifactId: string; artifactType: ArtifactType; missingElements: string[]; qualityIssues: string[] }>
  ): string[] {
    const requests: string[] = [];

    // Generate refinement requests for quality issues
    for (const incomplete of incompleteArtifacts) {
      if (incomplete.qualityIssues.length > 0) {
        requests.push(
          `Please refine ${incomplete.artifactType} to address: ${incomplete.qualityIssues.join(', ')}`
        );
      }
    }

    return requests;
  }

  private generateRoundSummary(
    roundNumber: number,
    artifacts: ParsedArtifact[],
    validation: ValidationResult
  ): string {
    const summary: string[] = [];

    summary.push(`Round ${roundNumber} produced ${artifacts.length} artifacts:`);
    artifacts.forEach(a => {
      summary.push(`- ${a.type}`);
    });

    summary.push(`\nCompleteness: ${validation.completenessScore}%`);
    summary.push(`Quality: ${validation.qualityScores.implementability}%`);

    if (validation.errors.length > 0) {
      summary.push(`\nIssues to address: ${validation.errors.length}`);
    }

    return summary.join('\n');
  }

  private extractObjectivesFromValidation(validation: ValidationResult): string[] {
    const objectives: string[] = [];

    // Extract objectives from follow-up questions
    for (const question of validation.followUpQuestions) {
      objectives.push(`Address: ${question}`);
    }

    return objectives;
  }

  private defineExpectedArtifactsFromIncomplete(
    incompleteArtifacts: Array<{ artifactId: string; artifactType: ArtifactType; missingElements: string[]; qualityIssues: string[] }>
  ): ExpectedArtifact[] {
    return incompleteArtifacts.map(incomplete => ({
      type: incomplete.artifactType,
      description: `Refined ${incomplete.artifactType}`,
      format: this.getDefaultFormat(incomplete.artifactType),
      structureGuidance: `Address: ${[...incomplete.missingElements, ...incomplete.qualityIssues].join(', ')}`
    }));
  }

  private getArtifactVersion(sessionState: SessionState, artifactType: ArtifactType): number {
    const versions = sessionState.artifactVersionHistory.get(artifactType);
    return versions ? versions.length : 0;
  }

  private getRequiredArtifacts(delegationType: DelegationType): ArtifactType[] {
    switch (delegationType) {
      case DelegationType.ARCHITECTURE_DESIGN:
        return [ArtifactType.MERMAID_DIAGRAM, ArtifactType.IMPLEMENTATION_GUIDE];
      case DelegationType.API_DESIGN:
        return [ArtifactType.OPENAPI_SPEC, ArtifactType.IMPLEMENTATION_GUIDE];
      case DelegationType.TEST_STRATEGY:
        return [ArtifactType.TEST_STRATEGY, ArtifactType.CODE_SNIPPET];
      default:
        return [ArtifactType.IMPLEMENTATION_GUIDE];
    }
  }

  private getArtifactDescription(type: ArtifactType): string {
    switch (type) {
      case ArtifactType.MERMAID_DIAGRAM:
        return 'Architecture diagram showing system components and relationships';
      case ArtifactType.OPENAPI_SPEC:
        return 'API specification with endpoints, schemas, and examples';
      case ArtifactType.IMPLEMENTATION_GUIDE:
        return 'Step-by-step implementation plan with dependencies';
      case ArtifactType.TEST_STRATEGY:
        return 'Comprehensive test strategy with property-based tests';
      case ArtifactType.CODE_SNIPPET:
        return 'Code examples and templates';
      default:
        return 'Artifact';
    }
  }

  private getDefaultFormat(type: ArtifactType): string {
    switch (type) {
      case ArtifactType.MERMAID_DIAGRAM:
        return 'Mermaid syntax in fenced code block';
      case ArtifactType.OPENAPI_SPEC:
        return 'OpenAPI 3.0 YAML in fenced code block';
      case ArtifactType.IMPLEMENTATION_GUIDE:
        return 'Markdown with numbered steps';
      case ArtifactType.TEST_STRATEGY:
        return 'Markdown with test descriptions';
      case ArtifactType.CODE_SNIPPET:
        return 'Code in fenced code blocks with language identifier';
      default:
        return 'Markdown';
    }
  }

  private getStructureGuidance(type: ArtifactType): string {
    switch (type) {
      case ArtifactType.MERMAID_DIAGRAM:
        return 'Use Mermaid graph syntax with labeled nodes and edges';
      case ArtifactType.OPENAPI_SPEC:
        return 'Include paths, components, and examples';
      case ArtifactType.IMPLEMENTATION_GUIDE:
        return 'Organize into phases with clear dependencies';
      case ArtifactType.TEST_STRATEGY:
        return 'Include property definitions and test generators';
      case ArtifactType.CODE_SNIPPET:
        return 'Provide working examples with comments';
      default:
        return 'Follow standard format';
    }
  }

  private getFormatInstructions(type: ArtifactType): string {
    switch (type) {
      case ArtifactType.MERMAID_DIAGRAM:
        return 'Use ```mermaid code blocks with valid Mermaid syntax';
      case ArtifactType.OPENAPI_SPEC:
        return 'Use ```yaml code blocks with OpenAPI 3.0 specification';
      case ArtifactType.IMPLEMENTATION_GUIDE:
        return 'Use markdown numbered lists with sub-items for dependencies';
      case ArtifactType.TEST_STRATEGY:
        return 'Use markdown sections with test case descriptions';
      case ArtifactType.CODE_SNIPPET:
        return 'Use fenced code blocks with appropriate language identifiers';
      default:
        return 'Use markdown format';
    }
  }

  private generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateRequestId(): string {
    return `request-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
