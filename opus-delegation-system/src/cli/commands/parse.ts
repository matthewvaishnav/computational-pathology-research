// Parse command - Parse Opus response

import { Command } from 'commander';
import { ArtifactParser } from '../../components/ArtifactParser.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';
import { validateFilePath } from '../../utils/pathValidation.js';
import * as fs from 'fs';

export const parseCommand = new Command('parse')
  .description('Parse Opus response from stdin or file')
  .requiredOption('-s, --session <id>', 'Session ID')
  .option('-f, --file <path>', 'Input file (default: stdin)')
  .action(async (options) => {
    try {
      const { session: sessionId, file } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`❌ Error: Session not found: ${sessionId}`);
        process.exit(1);
      }

      let response: string;
      if (file) {
        // Validate file path to prevent path traversal
        const safePath = await validateFilePath(file, process.cwd());
        response = fs.readFileSync(safePath, 'utf-8');
      } else {
        // Read from stdin
        response = fs.readFileSync(0, 'utf-8');
      }

      const parser = new ArtifactParser();
      const roundNumber = session.rounds.length + 1;
      const artifacts = parser.parseResponse(response, sessionId, roundNumber);

      console.log(`\n✅ Parsed ${artifacts.length} artifacts:`);
      artifacts.forEach((artifact) => {
        console.log(`  - ${artifact.type} (${artifact.content.length} chars)`);
      });

      console.log(`\n➡️  Next: opus-delegate validate --session ${sessionId}`);
    } catch (error) {
      if (error instanceof Error && error.message.includes('Path traversal')) {
        console.error('❌ Security Error:', error.message);
        console.error('💡 Tip: File path must be within current directory');
      } else {
        console.error('❌ Error parsing response:', error instanceof Error ? error.message : error);
      }
      process.exit(1);
    }
  });
