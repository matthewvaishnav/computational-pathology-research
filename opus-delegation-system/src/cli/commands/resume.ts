// Resume command - Resume interrupted session

import { Command } from 'commander';
import { CheckpointManager } from '../../utils/checkpoint.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';

export const resumeCommand = new Command('resume')
  .description('Resume interrupted delegation session')
  .requiredOption('-s, --session <id>', 'Session ID')
  .action(async (options) => {
    try {
      const { session: sessionId } = options;

      console.log('⏳ Loading checkpoint...');
      const checkpointManager = new CheckpointManager();
      const state = checkpointManager.resumeFromCheckpoint(sessionId);

      if (!state) {
        console.error('❌ Error: No checkpoint found for session:', sessionId);
        console.error('💡 Tip: Use "opus-delegate list" to see available sessions');
        process.exit(1);
      }

      console.log('⏳ Loading session data...');
      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error('❌ Error: Session not found:', sessionId);
        console.error('💡 Tip: The session may have been deleted or corrupted');
        process.exit(1);
      }

      console.log('✅ Session resumed successfully\n');
      console.log(`📋 Session: ${sessionId}`);
      console.log(`🎯 Problem: ${session.problem.description}`);
      console.log(`🔄 Rounds completed: ${session.rounds.length}`);

      if (session.rounds.length === 0) {
        console.log('\n➡️  Next step: opus-delegate context --session', sessionId);
      } else if (session.finalArtifacts.length === 0) {
        console.log('\n➡️  Next step: opus-delegate validate --session', sessionId);
      } else if (!session.implementationGuide) {
        console.log('\n➡️  Next step: opus-delegate guide --session', sessionId);
      } else {
        console.log('\n➡️  Next step: opus-delegate export --session', sessionId);
      }
    } catch (error) {
      console.error('❌ Error resuming session:', error instanceof Error ? error.message : error);
      console.error('💡 Tip: Check that the session directory exists and is readable');
      process.exit(1);
    }
  });
