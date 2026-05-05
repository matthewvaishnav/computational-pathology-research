// Request command - Generate delegation request

import { Command } from 'commander';
import { OpusDelegator } from '../../components/OpusDelegator.js';
import { TemplateLibrary } from '../../components/TemplateLibrary.js';
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

      const templateLibrary = new TemplateLibrary();
      const delegator = new OpusDelegator(templateLibrary);
      
      // Create a minimal context bundle for now
      const contextBundle = {
        title: session.problem.title,
        markdown: `# ${session.problem.title}\n\n${session.problem.description}`,
        files: [],
        totalSize: 0,
        compressionApplied: false
      };
      
      const request = delegator.generateDelegationRequest(
        sessionId,
        session.problem.title,
        session.problem.description,
        session.problem.type,
        contextBundle,
        template
      );

      if (output) {
        const fs = await import('fs');
        fs.writeFileSync(output, JSON.stringify(request, null, 2));
        console.log(`Request written to: ${output}`);
      } else {
        console.log('\n=== Delegation Request ===\n');
        console.log(JSON.stringify(request, null, 2));
      }

      console.log('\n\nCopy request to use.ai and paste Opus response');
      console.log(`Then: opus-delegate parse --session ${sessionId} < response.md`);
    } catch (error) {
      console.error('Error generating request:', error);
      process.exit(1);
    }
  });
