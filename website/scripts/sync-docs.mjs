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
      'Navigation guide for HistoCore platform, modeling, deployment, and validation documentation.',
  },
  {
    source: 'GETTING_STARTED.md',
    title: 'Getting Started',
    description:
      'Installation, environment setup, and first-run guidance for the HistoCore platform.',
  },
  {
    source: 'FRAMEWORK_OVERVIEW.md',
    title: 'Framework Overview',
    description:
      'High-level orientation to the HistoCore platform architecture and major workflows.',
  },
  {
    source: 'ARCHITECTURE.md',
    title: 'Architecture',
    description:
      'Detailed system architecture for the computational pathology platform and pipeline.',
  },
  {
    source: 'API_REFERENCE.md',
    title: 'API Reference',
    description:
      'Reference guide for the primary APIs, modules, and commands exposed by HistoCore.',
  },
  {
    source: 'FOUNDATION_MODELS.md',
    title: 'Foundation Models',
    description:
      'Guide to foundation-model support, tradeoffs, and integration patterns inside HistoCore.',
  },
  {
    source: 'INFERENCE_OPTIMIZATION.md',
    title: 'Inference Optimization',
    description:
      'Production inference guidance covering TorchScript, batching, and large-bag workflows.',
  },
  {
    source: 'BENCHMARK_SYSTEM.md',
    title: 'Benchmark System',
    description:
      'Benchmarking workflow, performance measurement, and comparison methodology for HistoCore.',
  },
  {
    source: 'PERFORMANCE_COMPARISON.md',
    title: 'Performance Comparison',
    description:
      'Comparative performance positioning for the platform and key optimization layers.',
  },
  {
    source: 'DEPLOYMENT.md',
    title: 'Deployment',
    description:
      'Deployment guide for packaging, serving, and operating HistoCore in production-style environments.',
  },
  {
    source: 'SECURITY_HARDENING.md',
    title: 'Security Hardening',
    description:
      'Operational hardening guidance for deploying the platform with stronger security posture.',
  },
  {
    source: 'CLINICAL_VALIDATION.md',
    title: 'Clinical Validation',
    description:
      'Validation framing, evidence expectations, and clinical-readiness guidance for HistoCore.',
  },
  {
    source: 'TESTING.md',
    title: 'Testing',
    description:
      'Testing strategy, suite organization, and quality assurance workflows for HistoCore.',
  },
  {
    source: 'QUICK_REFERENCE.md',
    title: 'Quick Reference',
    description:
      'Fast command and workflow reference for common HistoCore development and operations tasks.',
  },
  {
    source: 'TROUBLESHOOTING.md',
    title: 'Troubleshooting',
    description:
      'Troubleshooting guide for common install, runtime, and workflow issues across the platform.',
  },
];

const curatedDocIds = new Set(
  curatedDocs.map(({source}) => path.basename(source, path.extname(source))),
);

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
  const normalized = relativePath.replace(/\\/g, '/').replace(/^\.\.\//, '').replace(/\/$/, '');
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
            `> Note: An image referenced in the source docs is omitted from the live site build because the asset is not packaged with the website.`,
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

  await fs.writeFile(outputPath, frontmatter + sanitized, 'utf8');
}

async function main() {
  await fs.mkdir(outputDocsRoot, {recursive: true});

  for (const doc of curatedDocs) {
    await writeDoc(doc);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
