// Request command - Generate delegation request

import { Command } from 'commander';
import { OpusDelegator } from '../../components/OpusDelegator.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';

export const requestCommand = new Command('request')
  .description('Generate delegation request from template')
  .requiredOption('-s, --session <id>', 'Session ID')
  .option('-t, --template <id>', 'Template ID')
  .option('-o, --output <file>', 'Output file (default: stdout)')
  .action(async (options) => {
    try {
      const { session: sessionId, template, output } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`Session not found: ${sessionId}`);
        process.exit(1);
      }

      const delegator = new OpusDelegator();
      const request = delegator.generateDelegationRequest(
        sessionId,
        template || session.problem.type,
        {},
        '' // Context bundle would be loaded here
      );

      if (output) {
        const fs = await import('fs');
        fs.writeFileSync(output, request);
        console.log(`Request written to: ${output}`);
      } else {
        console.log('\n=== Delegation Request ===\n');
        console.log(request);
      }

      console.log('\n\nCopy request to use.ai and paste Opus response');
      console.log(`Then: opus-delegate parse --session ${sessionId} < response.md`);
    } catch (error) {
      console.error('Error generating request:', error);
      process.exit(1);
    }
  });
