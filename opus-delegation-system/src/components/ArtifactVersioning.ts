/**
 * Artifact Versioning Component
 * Implements Task 16 - Artifact Versioning and Comparison
 * Requirements: 15.1-15.7
 */

import { ParsedArtifact, ArtifactType, MermaidAST } from '../types/core.js';

/**
 * Artifact version metadata
 */
export interface ArtifactVersion {
  versionNumber: number;
  artifact: ParsedArtifact;
  timestamp: Date;
  roundNumber: number;
  sessionId: string;
  changesSummary?: string;
}

/**
 * Diff result for text-based artifacts
 */
export interface TextDiff {
  additions: Array<{ line: number; content: string }>;
  deletions: Array<{ line: number; content: string }>;
  modifications: Array<{ line: number; oldContent: string; newContent: string }>;
}

/**
 * Diff result for structural artifacts (diagrams)
 */
export interface StructuralDiff {
  nodesAdded: Array<{ id: string; label: string }>;
  nodesRemoved: Array<{ id: string; label: string }>;
  nodesModified: Array<{ id: string; oldLabel: string; newLabel: string }>;
  edgesAdded: Array<{ from: string; to: string; label?: string }>;
  edgesRemoved: Array<{ from: string; to: string; label?: string }>;
}

/**
 * Artifact Versioning Component
 * Manages artifact versions, comparison, and reversion
 */
export class ArtifactVersioning {
  private versions: Map<string, ArtifactVersion[]> = new Map();

  /**
   * Store artifact version
   * Requirement 15.1: Assign version numbers to artifacts in each round
   * Requirement 15.2: Store all versions with timestamps and metadata
   * Task 16.1: Create version management
   */
  public storeVersion(
    artifact: ParsedArtifact,
    sessionId: string,
    roundNumber: number
  ): ArtifactVersion {
    const artifactKey = this.getArtifactKey(artifact);
    const versions = this.versions.get(artifactKey) || [];

    const versionNumber = versions.length + 1;
    const version: ArtifactVersion = {
      versionNumber,
      artifact: { ...artifact },
      timestamp: new Date(),
      roundNumber,
      sessionId,
    };

    versions.push(version);
    this.versions.set(artifactKey, versions);

    return version;
  }

  /**
   * Get all versions of an artifact
   */
  public getVersions(artifactKey: string): ArtifactVersion[] {
    return this.versions.get(artifactKey) || [];
  }

  /**
   * Get specific version
   */
  public getVersion(artifactKey: string, versionNumber: number): ArtifactVersion | undefined {
    const versions = this.versions.get(artifactKey);
    return versions?.find((v) => v.versionNumber === versionNumber);
  }

  /**
   * Get latest version
   */
  public getLatestVersion(artifactKey: string): ArtifactVersion | undefined {
    const versions = this.versions.get(artifactKey);
    return versions?.[versions.length - 1];
  }

  /**
   * Compare two artifact versions
   * Requirement 15.3: Create text diff for markdown artifacts
   * Requirement 15.4: Create structural diff for diagrams
   * Task 16.2: Implement artifact comparison
   */
  public compareVersions(
    artifactKey: string,
    version1: number,
    version2: number
  ): TextDiff | StructuralDiff | null {
    const v1 = this.getVersion(artifactKey, version1);
    const v2 = this.getVersion(artifactKey, version2);

    if (!v1 || !v2) {
      return null;
    }

    // Determine comparison type based on artifact type
    if (v1.artifact.type === ArtifactType.MERMAID_DIAGRAM) {
      return this.compareStructural(v1.artifact, v2.artifact);
    } else {
      return this.compareText(v1.artifact.content, v2.artifact.content);
    }
  }

  /**
   * Compare text-based artifacts
   * Creates line-by-line diff
   */
  private compareText(oldContent: string, newContent: string): TextDiff {
    const oldLines = oldContent.split('\n');
    const newLines = newContent.split('\n');

    const additions: Array<{ line: number; content: string }> = [];
    const deletions: Array<{ line: number; content: string }> = [];
    const modifications: Array<{ line: number; oldContent: string; newContent: string }> = [];

    // Simple line-by-line comparison
    const maxLength = Math.max(oldLines.length, newLines.length);

    for (let i = 0; i < maxLength; i++) {
      const oldLine = oldLines[i];
      const newLine = newLines[i];

      if (oldLine === undefined && newLine !== undefined) {
        // Addition
        additions.push({ line: i + 1, content: newLine });
      } else if (oldLine !== undefined && newLine === undefined) {
        // Deletion
        deletions.push({ line: i + 1, content: oldLine });
      } else if (oldLine !== newLine) {
        // Modification
        modifications.push({ line: i + 1, oldContent: oldLine, newContent: newLine });
      }
    }

    return { additions, deletions, modifications };
  }

  /**
   * Compare structural artifacts (Mermaid diagrams)
   * Compares nodes and edges
   */
  private compareStructural(oldArtifact: ParsedArtifact, newArtifact: ParsedArtifact): StructuralDiff {
    const oldMermaid = oldArtifact.structured?.mermaid;
    const newMermaid = newArtifact.structured?.mermaid;

    const diff: StructuralDiff = {
      nodesAdded: [],
      nodesRemoved: [],
      nodesModified: [],
      edgesAdded: [],
      edgesRemoved: [],
    };

    if (!oldMermaid || !newMermaid) {
      return diff;
    }

    // Compare nodes
    const oldNodeIds = new Set(oldMermaid.nodes.map((n) => n.id));
    const newNodeIds = new Set(newMermaid.nodes.map((n) => n.id));

    // Find added nodes
    for (const node of newMermaid.nodes) {
      if (!oldNodeIds.has(node.id)) {
        diff.nodesAdded.push({ id: node.id, label: node.label });
      }
    }

    // Find removed nodes
    for (const node of oldMermaid.nodes) {
      if (!newNodeIds.has(node.id)) {
        diff.nodesRemoved.push({ id: node.id, label: node.label });
      }
    }

    // Find modified nodes
    for (const newNode of newMermaid.nodes) {
      const oldNode = oldMermaid.nodes.find((n) => n.id === newNode.id);
      if (oldNode && oldNode.label !== newNode.label) {
        diff.nodesModified.push({
          id: newNode.id,
          oldLabel: oldNode.label,
          newLabel: newNode.label,
        });
      }
    }

    // Compare edges
    const oldEdges = oldMermaid.edges.map((e) => `${e.from}->${e.to}`);
    const newEdges = newMermaid.edges.map((e) => `${e.from}->${e.to}`);

    // Find added edges
    for (const edge of newMermaid.edges) {
      const edgeKey = `${edge.from}->${edge.to}`;
      if (!oldEdges.includes(edgeKey)) {
        diff.edgesAdded.push({ from: edge.from, to: edge.to, label: edge.label });
      }
    }

    // Find removed edges
    for (const edge of oldMermaid.edges) {
      const edgeKey = `${edge.from}->${edge.to}`;
      if (!newEdges.includes(edgeKey)) {
        diff.edgesRemoved.push({ from: edge.from, to: edge.to, label: edge.label });
      }
    }

    return diff;
  }

  /**
   * Revert to previous version
   * Requirement 15.5: Support reverting to previous artifact versions
   * Task 16.3: Implement version reversion
   */
  public revertToVersion(artifactKey: string, versionNumber: number): ParsedArtifact | null {
    const version = this.getVersion(artifactKey, versionNumber);
    if (!version) {
      return null;
    }

    // Create new version with reverted content
    const revertedArtifact: ParsedArtifact = {
      ...version.artifact,
      id: this.generateNewId(),
      metadata: {
        ...version.artifact.metadata,
        extractedAt: new Date(),
        parseWarnings: [
          ...version.artifact.metadata.parseWarnings,
          `Reverted to version ${versionNumber}`,
        ],
      },
    };

    return revertedArtifact;
  }

  /**
   * Generate change summary between versions
   * Requirement 15.6: Generate change summaries describing what evolved
   */
  public generateChangeSummary(artifactKey: string, fromVersion: number, toVersion: number): string {
    const diff = this.compareVersions(artifactKey, fromVersion, toVersion);
    if (!diff) {
      return 'Unable to generate change summary: versions not found';
    }

    if ('additions' in diff) {
      // Text diff
      const textDiff = diff as TextDiff;
      const parts: string[] = [];

      if (textDiff.additions.length > 0) {
        parts.push(`${textDiff.additions.length} line(s) added`);
      }

      if (textDiff.deletions.length > 0) {
        parts.push(`${textDiff.deletions.length} line(s) deleted`);
      }

      if (textDiff.modifications.length > 0) {
        parts.push(`${textDiff.modifications.length} line(s) modified`);
      }

      return parts.length > 0 ? parts.join(', ') : 'No changes detected';
    } else {
      // Structural diff
      const structDiff = diff as StructuralDiff;
      const parts: string[] = [];

      if (structDiff.nodesAdded.length > 0) {
        parts.push(`${structDiff.nodesAdded.length} node(s) added`);
      }

      if (structDiff.nodesRemoved.length > 0) {
        parts.push(`${structDiff.nodesRemoved.length} node(s) removed`);
      }

      if (structDiff.nodesModified.length > 0) {
        parts.push(`${structDiff.nodesModified.length} node(s) modified`);
      }

      if (structDiff.edgesAdded.length > 0) {
        parts.push(`${structDiff.edgesAdded.length} edge(s) added`);
      }

      if (structDiff.edgesRemoved.length > 0) {
        parts.push(`${structDiff.edgesRemoved.length} edge(s) removed`);
      }

      return parts.length > 0 ? parts.join(', ') : 'No changes detected';
    }
  }

  /**
   * Get version history for an artifact
   */
  public getVersionHistory(artifactKey: string): Array<{
    versionNumber: number;
    timestamp: Date;
    roundNumber: number;
    changesSummary: string;
  }> {
    const versions = this.versions.get(artifactKey);
    if (!versions || versions.length === 0) {
      return [];
    }

    const history: Array<{
      versionNumber: number;
      timestamp: Date;
      roundNumber: number;
      changesSummary: string;
    }> = [];

    for (let i = 0; i < versions.length; i++) {
      const version = versions[i];
      let changesSummary = 'Initial version';

      if (i > 0) {
        changesSummary = this.generateChangeSummary(artifactKey, i, i + 1);
      }

      history.push({
        versionNumber: version.versionNumber,
        timestamp: version.timestamp,
        roundNumber: version.roundNumber,
        changesSummary,
      });
    }

    return history;
  }

  /**
   * Export version comparison as markdown
   */
  public exportComparisonAsMarkdown(
    artifactKey: string,
    version1: number,
    version2: number
  ): string {
    const diff = this.compareVersions(artifactKey, version1, version2);
    if (!diff) {
      return '# Comparison Failed\n\nUnable to compare versions.';
    }

    let md = `# Version Comparison: ${artifactKey}\n\n`;
    md += `**From Version:** ${version1}\n`;
    md += `**To Version:** ${version2}\n\n`;

    if ('additions' in diff) {
      // Text diff
      const textDiff = diff as TextDiff;

      if (textDiff.additions.length > 0) {
        md += '## Additions\n\n';
        for (const add of textDiff.additions) {
          md += `**Line ${add.line}:** \`${add.content}\`\n`;
        }
        md += '\n';
      }

      if (textDiff.deletions.length > 0) {
        md += '## Deletions\n\n';
        for (const del of textDiff.deletions) {
          md += `**Line ${del.line}:** ~~\`${del.content}\`~~\n`;
        }
        md += '\n';
      }

      if (textDiff.modifications.length > 0) {
        md += '## Modifications\n\n';
        for (const mod of textDiff.modifications) {
          md += `**Line ${mod.line}:**\n`;
          md += `- Old: ~~\`${mod.oldContent}\`~~\n`;
          md += `- New: \`${mod.newContent}\`\n\n`;
        }
      }
    } else {
      // Structural diff
      const structDiff = diff as StructuralDiff;

      if (structDiff.nodesAdded.length > 0) {
        md += '## Nodes Added\n\n';
        for (const node of structDiff.nodesAdded) {
          md += `- **${node.id}**: ${node.label}\n`;
        }
        md += '\n';
      }

      if (structDiff.nodesRemoved.length > 0) {
        md += '## Nodes Removed\n\n';
        for (const node of structDiff.nodesRemoved) {
          md += `- ~~**${node.id}**: ${node.label}~~\n`;
        }
        md += '\n';
      }

      if (structDiff.nodesModified.length > 0) {
        md += '## Nodes Modified\n\n';
        for (const node of structDiff.nodesModified) {
          md += `- **${node.id}**:\n`;
          md += `  - Old: ~~${node.oldLabel}~~\n`;
          md += `  - New: ${node.newLabel}\n`;
        }
        md += '\n';
      }

      if (structDiff.edgesAdded.length > 0) {
        md += '## Edges Added\n\n';
        for (const edge of structDiff.edgesAdded) {
          md += `- ${edge.from} → ${edge.to}${edge.label ? ` (${edge.label})` : ''}\n`;
        }
        md += '\n';
      }

      if (structDiff.edgesRemoved.length > 0) {
        md += '## Edges Removed\n\n';
        for (const edge of structDiff.edgesRemoved) {
          md += `- ~~${edge.from} → ${edge.to}${edge.label ? ` (${edge.label})` : ''}~~\n`;
        }
        md += '\n';
      }
    }

    return md;
  }

  /**
   * Clear all versions (for testing)
   */
  public clearVersions(): void {
    this.versions.clear();
  }

  /**
   * Get artifact key for versioning
   */
  private getArtifactKey(artifact: ParsedArtifact): string {
    // Use artifact type and a hash of content as key
    const contentHash = this.simpleHash(artifact.content);
    return `${artifact.type}-${contentHash.substring(0, 8)}`;
  }

  /**
   * Simple hash function for content
   */
  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Generate new artifact ID
   */
  private generateNewId(): string {
    return `artifact-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get statistics about versioning
   */
  public getStatistics(): {
    totalArtifacts: number;
    totalVersions: number;
    averageVersionsPerArtifact: number;
  } {
    const totalArtifacts = this.versions.size;
    let totalVersions = 0;

    for (const versions of this.versions.values()) {
      totalVersions += versions.length;
    }

    const averageVersionsPerArtifact =
      totalArtifacts > 0 ? Math.round((totalVersions / totalArtifacts) * 10) / 10 : 0;

    return {
      totalArtifacts,
      totalVersions,
      averageVersionsPerArtifact,
    };
  }
}
