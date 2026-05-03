// Validate command - Validate parsed artifacts

import { Command } from 'commander';
import { ArtifactValidator } from '../../components/ArtifactValidator.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';

export const validateCommand = new Command('validate')
  .description('Validate parsed artifacts')
  .requiredOption('-s, --session <id>', 'Session ID')
  .action(async (options) => {
    try {
      const { session: sessionId } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`Session not found: ${sessionId}`);
        process.exit(1);
      }

      if (session.rounds.length === 0) {
        console.error('No rounds found. Parse response first.');
        process.exit(1);
      }

      const lastRound = session.rounds[session.rounds.length - 1];
      const validator = new ArtifactValidator();
      const results = validator.validateAll(lastRound.artifacts);

      console.log('\n=== Validation Results ===\n');
      results.forEach((result) => {
        console.log(`${result.artifactType}:`);
        console.log(`  Valid: ${result.isValid ? 'Yes' : 'No'}`);
        console.log(`  Completeness: ${result.completenessScore}%`);

        if (result.issues.length > 0) {
          console.log('  Issues:');
          result.issues.forEach((issue) => {
            console.log(`    - [${issue.severity}] ${issue.message}`);
          });
        }

        if (result.suggestions.length > 0) {
          console.log('  Suggestions:');
          result.suggestions.forEach((suggestion) => {
            console.log(`    - ${suggestion}`);
          });
        }
        console.log('');
      });

      const allValid = results.every((r) => r.isValid);
      if (allValid) {
        console.log(`Next: opus-delegate guide --session ${sessionId}`);
      } else {
        console.log(`Next: opus-delegate followup --session ${sessionId}`);
      }
    } catch (error) {
      console.error('Error validating artifacts:', error);
      process.exit(1);
    }
  });
