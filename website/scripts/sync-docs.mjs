import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const websiteRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(websiteRoot, '..');
const sourceDocsRoot = path.join(repoRoot, 'docs');
const outputDocsRoot = path.join(websiteRoot, 'docs');

const curatedDocs = [
  {
    source: 'DOCS_INDEX.md',
    title: 'Documentation Index',
    description:
      'Navigation guide for the current audited computational pathology research record.',
  },
  {
    source: 'CURRENT_STATUS.md',
    title: 'Current Status',
    description:
      'Promoted evidence, active exploratory studies, explicit non-claims, and next steps.',
  },
  {
    source: 'PLATFORM_OVERVIEW.md',
    title: 'Repository Overview',
    description:
      'Research scope and the boundary between validated evidence, exploratory work, and historical software modules.',
  },
  {
    source: 'PORTFOLIO_SUMMARY.md',
    title: 'Research Engineering Portfolio',
    description:
      'Neural methods, experimental design, reproducibility engineering, scientific corrections, and limitations.',
  },
  {
    source: 'PRESENTATION_ABSTRACT.md',
    title: 'Presentation Abstract',
    description:
      'Bounded abstract for Paired-Acquisition Neural Factorization and its audited evidence.',
  },
  {
    source: 'DATA_PROVENANCE.md',
    title: 'Data Provenance',
    description:
      'Canonical dataset provenance and the separation between scientific data and software-test fixtures.',
  },
  {
    source: 'REPOSITORY_OVERVIEW.md',
    title: 'Repository Structure',
    description:
      'Project organization and navigation guide for the computational pathology research repository.',
  },
  {
    source: 'PCAM_REAL_RESULTS.md',
    title: 'PCam Result Record',
    description:
      'Single-split PatchCamelyon engineering result with its current public claim boundary.',
  },
  {
    source: 'TESTING.md',
    title: 'Testing',
    description:
      'Testing strategy, suite organization, and quality-assurance workflows.',
  },
  {
    source: 'FOUNDATION_MODELS.md',
    title: 'Foundation Models',
    description:
      'Foundation-model integration references and implementation boundaries.',
  },
  {
    source: 'MODEL_INTERPRETABILITY.md',
    title: 'Model Interpretability',
    description:
      'Grad-CAM, attention visualization, and explainability tooling references.',
  },
  {
    source: 'QUICK_REFERENCE.md',
    title: 'Quick Reference',
    description:
      'Fast command and workflow reference for repository development tasks.',
  },
  {
    source: 'TROUBLESHOOTING.md',
    title: 'Troubleshooting',
    description:
      'Troubleshooting guide for installation, runtime, and development workflows.',
  },
];

const curatedDocIds = new Set(
  curatedDocs.map(({source}) => path.basename(source, path.extname(source))),
);

const retainedOutputFiles = new Set([
  'intro.md',
  ...curatedDocs.map(({source}) => source.replace(/\\/g, '/')),
]);

function stripFrontmatter(content) {
  if (!content.startsWith('---')) {
    return content;
  }

  const end = content.indexOf('\n---', 3);
  if (end === -1) {
    return content;
  }

  return content.slice(end + 4).replace(/^\r?\n/, '');
}

function toGithubBlobUrl(relativePath) {
  const normalized = relativePath.replace(/\\/g, '/').replace(/^\.\.\//, '');
  return `https://github.com/matthewvaishnav/computational-pathology-research/blob/main/${normalized}`;
}

function toGithubTreeUrl(relativePath) {
  const normalized = relativePath
    .replace(/\\/g, '/')
    .replace(/^\.\.\//, '')
    .replace(/\/$/, '');
  return `https://github.com/matthewvaishnav/computational-pathology-research/tree/main/${normalized}`;
}

function getRepoTargetInfo(pathname, sourceDir) {
  const candidates = [
    path.resolve(sourceDir, pathname),
    path.resolve(sourceDocsRoot, pathname),
    path.resolve(repoRoot, pathname),
  ];

  for (const candidate of candidates) {
    if (!fsSync.existsSync(candidate)) {
      continue;
    }

    const stats = fsSync.statSync(candidate);
    return {
      isDirectory: stats.isDirectory(),
      relativePath: path.relative(repoRoot, candidate).replace(/\\/g, '/'),
    };
  }

  return null;
}

function toSiteDocUrl(docId, suffix) {
  return `/docs/${docId}${suffix}`;
}

function rewriteLinkTarget(target, sourceDir) {
  if (
    target.startsWith('http://') ||
    target.startsWith('https://') ||
    target.startsWith('mailto:') ||
    target.startsWith('#')
  ) {
    return target;
  }

  if (target === '.' || target === './') {
    return '/docs/';
  }

  const [pathname, hash = ''] = target.split('#');
  const suffix = hash ? `#${hash}` : '';

  if (pathname.endsWith('.html')) {
    const noExtension = pathname.slice(0, -'.html'.length);
    const docId = path.basename(noExtension);
    if (curatedDocIds.has(docId)) {
      return toSiteDocUrl(docId, suffix);
    }

    if (noExtension.startsWith('../')) {
      return `${toGithubBlobUrl(`${noExtension}.md`)}${suffix}`;
    }

    const repoTarget = getRepoTargetInfo(`${noExtension}.md`, sourceDir);
    if (repoTarget) {
      return `${toGithubBlobUrl(repoTarget.relativePath)}${suffix}`;
    }

    return null;
  }

  if (pathname.endsWith('.md')) {
    const docId = path.basename(pathname, '.md');
    if (curatedDocIds.has(docId)) {
      return toSiteDocUrl(docId, suffix);
    }

    const repoTarget = getRepoTargetInfo(pathname, sourceDir);
    if (repoTarget) {
      return repoTarget.isDirectory
        ? `${toGithubTreeUrl(repoTarget.relativePath)}${suffix}`
        : `${toGithubBlobUrl(repoTarget.relativePath)}${suffix}`;
    }

    return null;
  }

  const docId = path.basename(pathname);
  if (curatedDocIds.has(docId)) {
    return toSiteDocUrl(docId, suffix);
  }

  const repoTarget =
    getRepoTargetInfo(pathname, sourceDir) ?? getRepoTargetInfo(`${pathname}.md`, sourceDir);
  if (repoTarget) {
    return repoTarget.isDirectory
      ? `${toGithubTreeUrl(repoTarget.relativePath)}${suffix}`
      : `${toGithubBlobUrl(repoTarget.relativePath)}${suffix}`;
  }

  return null;
}

function sanitizeNonFenceLine(line, sourceDir) {
  const imageMatch = line.match(/^!\[[^\]]*]\(([^)]+)\)\s*$/);
  if (imageMatch) {
    const candidate = imageMatch[1].split('#')[0];
    if (candidate.startsWith('assets/')) {
      return Promise.resolve(line.replace(candidate, `/img/${path.basename(candidate)}`));
    }
    if (
      !candidate.startsWith('http://') &&
      !candidate.startsWith('https://') &&
      !candidate.startsWith('pathname://')
    ) {
      const imagePath = path.resolve(sourceDir, candidate);
      return fs
        .access(imagePath)
        .then(() => line)
        .catch(
          () =>
            '> Note: An image referenced in the source docs is omitted from the live site build because the asset is not packaged with the website.',
        );
    }
  }

  const rewritten = line
    .replace(/\[([^\]]+)]\(([^)]+)\)/g, (_match, label, target) => {
      const rewrittenTarget = rewriteLinkTarget(target, sourceDir);
      return rewrittenTarget ? `[${label}](${rewrittenTarget})` : label;
    })
    .replace(/<(?=[A-Za-z0-9/])/g, '&lt;')
    .replace(/(?<=[A-Za-z0-9)])>/g, '&gt;');

  return Promise.resolve(rewritten);
}

async function sanitizeMarkdown(content, sourceDir) {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const sanitizedLines = [];
  let inFence = false;

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      inFence = !inFence;
      sanitizedLines.push(line);
      continue;
    }

    if (inFence) {
      sanitizedLines.push(line);
      continue;
    }

    sanitizedLines.push(await sanitizeNonFenceLine(line, sourceDir));
  }

  return sanitizedLines.join('\n').trimEnd() + '\n';
}

async function writeDoc(doc) {
  const sourcePath = path.join(sourceDocsRoot, doc.source);
  const outputPath = path.join(outputDocsRoot, doc.source);
  const sourceDir = path.dirname(sourcePath);
  const raw = await fs.readFile(sourcePath, 'utf8');
  const stripped = stripFrontmatter(raw);
  const sanitized = await sanitizeMarkdown(stripped, sourceDir);
  const frontmatter =
    `---\n` +
    `title: ${doc.title}\n` +
    `description: ${doc.description}\n` +
    `---\n\n`;

  await fs.mkdir(path.dirname(outputPath), {recursive: true});
  await fs.writeFile(outputPath, frontmatter + sanitized, 'utf8');
}

async function removeStaleGeneratedDocs() {
  const entries = await fs.readdir(outputDocsRoot, {withFileTypes: true});

  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.md')) {
      continue;
    }

    if (retainedOutputFiles.has(entry.name)) {
      continue;
    }

    await fs.rm(path.join(outputDocsRoot, entry.name));
  }
}

async function main() {
  await fs.mkdir(outputDocsRoot, {recursive: true});
  await removeStaleGeneratedDocs();

  for (const doc of curatedDocs) {
    await writeDoc(doc);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
