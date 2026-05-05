// Followup command - Generate follow-up request

import { Command } from 'commander';
import { OpusDelegator } from '../../components/OpusDelegator.js';
import { TemplateLibrary } from '../../components/TemplateLibrary.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';

export const followupCommand = new Command('followup')
  .description('Generate follow-up request for incomplete artifacts')
  .requiredOption('-s, --session <id>', 'Session ID')
  .option('-o, --output <file>', 'Output file (default: stdout)')
  .action(async (options) => {
    try {
      const { session: sessionId, output } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`Session not found: ${sessionId}`);
        process.exit(1);
      }

      if (session.rounds.length === 0) {
        console.error('No rounds found');
        process.exit(1);
      }

      const lastRound = session.rounds[session.rounds.length - 1];
      
      const templateLibrary = new TemplateLibrary();
      const delegator = new OpusDelegator(templateLibrary);
      const followup = delegator.generateFollowUpRequest(
        sessionId, 
        lastRound.validation[0], // Take first validation result
        lastRound.artifacts
      );

      if (output) {
        const fs = await import('fs');
        fs.writeFileSync(output, JSON.stringify(followup, null, 2));
        console.log(`Follow-up request written to: ${output}`);
      } else {
        console.log('\n=== Follow-up Request ===\n');
        console.log(JSON.stringify(followup, null, 2));
      }

      console.log('\n\nCopy to use.ai and paste response');
      console.log(`Then: opus-delegate parse --session ${sessionId} < response.md`);
    } catch (error) {
      console.error('Error generating follow-up:', error);
      process.exit(1);
    }
  });
