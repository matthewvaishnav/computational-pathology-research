// Guide command - Generate implementation guide

import { Command } from 'commander';
import { ImplementationGuideGenerator } from '../../components/ImplementationGuideGenerator.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';

export const guideCommand = new Command('guide')
  .description('Generate implementation guide from validated artifacts')
  .requiredOption('-s, --session <id>', 'Session ID')
  .option('-o, --output <file>', 'Output file')
  .option('-n, --name <name>', 'Project name')
  .action(async (options) => {
    try {
      const { session: sessionId, output, name } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`Session not found: ${sessionId}`);
        process.exit(1);
      }

      if (session.finalArtifacts.length === 0) {
        console.error('No validated artifacts found');
        process.exit(1);
      }

      const generator = new ImplementationGuideGenerator();
      const projectName = name || session.problem.title;
      const guide = generator.generateGuide(session.finalArtifacts, projectName);

      const markdown = generator.exportAsMarkdown(guide);

      if (output) {
        const fs = await import('fs');
        fs.writeFileSync(output, markdown);
        console.log(`Implementation guide written to: ${output}`);
      } else {
        console.log('\n=== Implementation Guide ===\n');
        console.log(markdown);
      }

      console.log(`\n${guide.steps.length} implementation steps generated`);
      console.log(`Next: opus-delegate export --session ${sessionId}`);
    } catch (error) {
      console.error('Error generating guide:', error);
      process.exit(1);
    }
  });
