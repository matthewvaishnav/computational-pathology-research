// List command - List sessions

import { Command } from 'commander';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';
import type { DelegationType } from '../../types/core.js';

export const listCommand = new Command('list')
  .description('List sessions with filtering options')
  .option('-t, --type <type>', 'Filter by delegation type')
  .option('-s, --status <status>', 'Filter by status (active, completed, abandoned)')
  .action(async (options) => {
    try {
      const { type, status } = options;

      const historyManager = new SessionHistoryManager();
      const sessions = historyManager.searchSessions({
        problemType: type as DelegationType,
        status: status as 'active' | 'completed' | 'abandoned',
      });

      if (sessions.length === 0) {
        console.log('No sessions found');
        return;
      }

      console.log(`\nFound ${sessions.length} session(s):\n`);

      sessions.forEach((session) => {
        console.log(`ID: ${session.id}`);
        console.log(`  Title: ${session.problem.title}`);
        console.log(`  Type: ${session.problem.type}`);
        console.log(`  Complexity: ${session.problem.complexity}`);
        console.log(`  Created: ${session.createdAt.toISOString()}`);
        console.log(`  Rounds: ${session.rounds.length}`);
        console.log(`  Artifacts: ${session.finalArtifacts.length}`);
        console.log('');
      });
    } catch (error) {
      console.error('Error listing sessions:', error);
      process.exit(1);
    }
  });
