#!/usr/bin/env node

// CLI entry point for Opus Delegation System

import { Command } from 'commander';
import { initCommand } from './commands/init.js';
import { contextCommand } from './commands/context.js';
import { requestCommand } from './commands/request.js';
import { parseCommand } from './commands/parse.js';
import { validateCommand } from './commands/validate.js';
import { followupCommand } from './commands/followup.js';
import { guideCommand } from './commands/guide.js';
import { exportCommand } from './commands/export.js';
import { listCommand } from './commands/list.js';
import { resumeCommand } from './commands/resume.js';

const program = new Command();

program
  .name('opus-delegate')
  .description('Opus Delegation System - Delegate complex problems to Claude Opus 4.5')
  .version('1.0.0');

// Register commands
program.addCommand(initCommand);
program.addCommand(contextCommand);
program.addCommand(requestCommand);
program.addCommand(parseCommand);
program.addCommand(validateCommand);
program.addCommand(followupCommand);
program.addCommand(guideCommand);
program.addCommand(exportCommand);
program.addCommand(listCommand);
program.addCommand(resumeCommand);

program.parse(process.argv);
