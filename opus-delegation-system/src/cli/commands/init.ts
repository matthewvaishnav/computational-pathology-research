// Init command - Initialize new delegation

import { Command } from 'commander';
import { ProblemClassifier } from '../../components/ProblemClassifier.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';
import type { DelegationType } from '../../types/core.js';

export const initCommand = new Command('init')
  .description('Initialize new delegation with problem description and type')
  .requiredOption('-t, --type <type>', 'Delegation type (architecture_design, api_design, etc.)')
  .requiredOption('-p, --problem <description>', 'Problem description')
  .option('-c, --complexity <level>', 'Complexity level (simple, moderate, complex)', 'moderate')
  .action(async (options) => {
    try {
      const { type, problem, complexity } = options;

      // Validate delegation type
      const validTypes = [
        'architecture_design',
        'api_design',
        'test_strategy',
        'integration_design',
        'refactoring_analysis',
        'formal_verification',
      ];

      if (!validTypes.includes(type)) {
        console.error(`Invalid delegation type: ${type}`);
        console.error(`Valid types: ${validTypes.join(', ')}`);
        process.exit(1);
      }

      // Validate complexity
      const validComplexity = ['simple', 'moderate', 'complex'];
      if (!validComplexity.includes(complexity)) {
        console.error(`Invalid complexity: ${complexity}`);
        console.error(`Valid complexity levels: ${validComplexity.join(', ')}`);
        process.exit(1);
      }

      // Classify problem
      const classifier = new ProblemClassifier();
      const classification = classifier.classifyProblem(problem);

      console.log('\n=== Problem Classification ===');
      console.log(`Type: ${classification.classification.delegationType}`);
      console.log(`Complexity: ${classification.classification.complexity}`);
      console.log(`Should Delegate: ${classification.shouldDelegate ? 'Yes' : 'No'}`);
      console.log(`Confidence: ${(classification.classification.confidence * 100).toFixed(1)}%`);
      console.log(`\nRecommendation: ${classification.recommendation}`);

      // Create session
      const historyManager = new SessionHistoryManager();
      const session = historyManager.createSession(
        `Delegation: ${type}`,
        problem,
        type as DelegationType,
        complexity as 'simple' | 'moderate' | 'complex'
      );

      console.log('\n=== Session Created ===');
      console.log(`Session ID: ${session.id}`);
      console.log(`Created: ${session.createdAt.toISOString()}`);
      console.log('\nNext steps:');
      console.log(`1. Extract context: opus-delegate context --session ${session.id}`);
      console.log(`2. Generate request: opus-delegate request --session ${session.id}`);
    } catch (error) {
      console.error('Error initializing delegation:', error);
      process.exit(1);
    }
  });
