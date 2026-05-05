// Context command - Extract context for delegation

import { Command } from 'commander';
import { ContextExtractor } from '../../components/ContextExtractor.js';
import { ContextPackager } from '../../components/ContextPackager.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';

export const contextCommand = new Command('context')
  .description('Extract context for existing delegation')
  .requiredOption('-s, --session <id>', 'Session ID')
  .option('-d, --deep', 'Use deep extraction strategy', false)
  .option('-r, --repo <path>', 'Repository path', process.cwd())
  .action(async (options) => {
    try {
      const { session: sessionId, deep, repo } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`Session not found: ${sessionId}`);
        process.exit(1);
      }

      console.log(`\nExtracting context for session: ${sessionId}`);
      console.log(`Problem: ${session.problem.description}`);
      console.log(`Type: ${session.problem.type}`);

      const extractor = new ContextExtractor();
      const files = await extractor.extractContext(
        session.problem.type,
        session.problem.description,
        repo,
        { depth: deep ? 3 : 1 }
      );

      console.log(`\nExtracted ${files.files.length} files`);

      const packager = new ContextPackager();
      const bundle = await packager.packageContext(
        session.problem.title,
        session.problem.description,
        files.files,
        files.snippets,
        files.strategy
      );

      console.log(`Context size: ${bundle.totalSize} characters`);
      console.log(`Compression applied: ${bundle.compressionApplied ? 'Yes' : 'No'}`);

      console.log('\nContext bundle ready');
      console.log(`Next: opus-delegate request --session ${sessionId}`);
    } catch (error) {
      console.error('Error extracting context:', error);
      process.exit(1);
    }
  });
