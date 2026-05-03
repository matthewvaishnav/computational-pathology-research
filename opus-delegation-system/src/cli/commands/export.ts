// Export command - Export artifacts

import { Command } from 'commander';
import { ArtifactExporter } from '../../components/ArtifactExporter.js';
import { SessionHistoryManager } from '../../components/SessionHistoryManager.js';
import { validateDirectoryPath } from '../../utils/pathValidation.js';

export const exportCommand = new Command('export')
  .description('Export artifacts in various formats')
  .requiredOption('-s, --session <id>', 'Session ID')
  .option('-f, --format <format>', 'Export format (all, yaml, html, markdown)', 'all')
  .option('-o, --output <dir>', 'Output directory', './artifacts')
  .action(async (options) => {
    try {
      const { session: sessionId, format, output } = options;

      const historyManager = new SessionHistoryManager();
      const session = historyManager.getSession(sessionId);

      if (!session) {
        console.error(`❌ Error: Session not found: ${sessionId}`);
        process.exit(1);
      }

      // Validate output directory to prevent path traversal
      const safeOutput = validateDirectoryPath(output, process.cwd());
      const exporter = new ArtifactExporter(safeOutput);

      console.log(`\n⏳ Exporting artifacts to: ${safeOutput}`);

      let exportCount = 0;

      for (const artifact of session.finalArtifacts) {
        if (artifact.type === 'mermaid_diagram' && (format === 'all' || format === 'mmd')) {
          exporter.exportMermaidDiagram(artifact, `diagram-${artifact.id}`, 'mmd');
          exportCount++;
        }

        if (
          artifact.type === 'openapi_specification' &&
          (format === 'all' || format === 'yaml')
        ) {
          exporter.exportOpenAPISpec(artifact, `api-${artifact.id}`, 'yaml');
          exportCount++;
        }

        if (
          artifact.type === 'openapi_specification' &&
          (format === 'all' || format === 'html')
        ) {
          exporter.exportOpenAPISpec(artifact, `api-docs-${artifact.id}`, 'html');
          exportCount++;
        }
      }

      if (session.implementationGuide && (format === 'all' || format === 'markdown')) {
        exporter.exportImplementationGuide(session.implementationGuide, 'implementation-guide');
        exportCount++;
      }

      console.log(`✅ Exported ${exportCount} artifacts`);
      console.log('\n🎉 Delegation complete!');
    } catch (error) {
      if (error instanceof Error && error.message.includes('Path traversal')) {
        console.error('❌ Security Error:', error.message);
        console.error('💡 Tip: Output directory must be within current directory');
      } else {
        console.error('❌ Error exporting artifacts:', error instanceof Error ? error.message : error);
      }
      process.exit(1);
    }
  });
