/**
 * Opus Delegation System - Main Entry Point
 * Exports core components and types
 */

// Core types
export * from './types/core.js';

// Components
export { ProblemClassifier } from './components/ProblemClassifier.js';
export { ContextExtractor } from './components/ContextExtractor.js';
export { ContextPackager } from './components/ContextPackager.js';
export { TemplateLibrary, DelegationTemplate, TemplateParameter, TemplateUsageStats } from './components/TemplateLibrary.js';
export { OpusDelegator } from './components/OpusDelegator.js';
export { ArtifactParser, ArtifactParseError } from './components/ArtifactParser.js';
export { ArtifactValidator } from './components/ArtifactValidator.js';