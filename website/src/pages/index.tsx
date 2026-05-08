import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const capabilityCards = [
  {
    title: 'Clinical-scale inference',
    body:
      'Large-bag sliding-window inference, uncertainty aggregation, and deployment-facing performance guidance for real-world whole-slide workflows.',
    href: '/docs/INFERENCE_OPTIMIZATION',
  },
  {
    title: 'Foundation model integration',
    body:
      'UNI, Phikon, GigaPath, and adapter-based workflows with documented projection, freezing, and compatibility behavior.',
    href: '/docs/FOUNDATION_MODELS',
  },
  {
    title: 'Deployment & operations',
    body:
      'Docker, monitoring, distributed tracing, and clinical integration guides organized like a real platform, not a research dump.',
    href: '/docs/DEPLOYMENT',
  },
  {
    title: 'Security & compliance',
    body:
      'Security hardening, HIPAA-adjacent controls, PACS integration, and regulatory material surfaced in one coherent delivery path.',
    href: '/docs/SECURITY_HARDENING',
  },
  {
    title: 'Benchmarkable engineering',
    body:
      'Performance comparisons, testing strategy, and benchmark-system docs that help teams verify claims instead of taking marketing at face value.',
    href: '/docs/BENCHMARK_SYSTEM',
  },
  {
    title: 'Research-to-product docs',
    body:
      'Curated documentation architecture for onboarding researchers, ML engineers, platform teams, and deployment stakeholders.',
    href: '/docs/DOCS_INDEX',
  },
];

const workflowTracks = [
  {
    eyebrow: 'Research',
    title: 'Ship experiments without losing the plot',
    body:
      'Move from dataset prep to models, training, interpretability, and benchmark reporting with a cleaner documentation spine.',
    href: '/docs/BENCHMARK_SYSTEM',
  },
  {
    eyebrow: 'Engineering',
    title: 'Turn promising pipelines into systems',
    body:
      'Use the architecture, inference, security, and deployment sections as a proper implementation map for production work.',
    href: '/docs/ARCHITECTURE',
  },
  {
    eyebrow: 'Clinical',
    title: 'Review readiness before integration',
    body:
      'Surface clinical validation, PACS, regulatory, and operational guidance in one path instead of scattered markdown islands.',
    href: '/docs/CLINICAL_VALIDATION',
  },
];

const statItems = [
  {label: 'Focus', value: 'Docs + product'},
  {label: 'Navigation', value: 'Structured'},
  {label: 'Search', value: 'Local index'},
  {label: 'Stack', value: 'Docusaurus'},
];

export default function Home(): ReactNode {
  const architectureImage = useBaseUrl('/img/architecture-overview.png');

  return (
    <Layout
      title="Production-grade computational pathology website"
      description="HistoCore is a computational pathology platform with structured docs for research, engineering, deployment, and clinical-scale workflows.">
      <main className={styles.page}>
        <section className={styles.hero}>
          <div className={styles.heroCopy}>
            <p className={styles.eyebrow}>Computational pathology platform</p>
            <Heading as="h1" className={styles.title}>
              A real product website for a serious technical system.
            </Heading>
            <p className={styles.subtitle}>
              HistoCore now has a production-grade docs framework, clearer
              navigation, and a platform-facing information architecture built
              for researchers, ML engineers, and deployment teams.
            </p>
            <div className={styles.heroActions}>
              <Link className={clsx('button button--primary', styles.primaryCta)} to="/docs/GETTING_STARTED">
                Start with the docs
              </Link>
              <Link className={clsx('button button--secondary', styles.secondaryCta)} to="/docs/ARCHITECTURE">
                Explore the architecture
              </Link>
            </div>
            <div className={styles.statRow}>
              {statItems.map((item) => (
                <div key={item.label} className={styles.statCard}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className={styles.heroPanel}>
            <div className={styles.panelChrome}>
              <span />
              <span />
              <span />
            </div>
            <img
              className={styles.archImage}
              src={architectureImage}
              alt="HistoCore architecture overview"
            />
            <div className={styles.panelSummary}>
              <p>Platform pillars</p>
              <ul>
                <li>Research and benchmark workflows</li>
                <li>Inference and deployment guidance</li>
                <li>Clinical integration and security material</li>
              </ul>
            </div>
          </div>
        </section>

        <section className={styles.workflowSection}>
          <div className={styles.sectionHeading}>
            <p>Three entry paths</p>
            <Heading as="h2">Different audiences can find the right surface fast</Heading>
          </div>
          <div className={styles.workflowGrid}>
            {workflowTracks.map((track) => (
              <Link key={track.title} className={styles.workflowCard} to={track.href}>
                <span>{track.eyebrow}</span>
                <h3>{track.title}</h3>
                <p>{track.body}</p>
              </Link>
            ))}
          </div>
        </section>

        <section className={styles.capabilitySection}>
          <div className={styles.sectionHeading}>
            <p>What the site now does better</p>
            <Heading as="h2">Production-grade structure instead of a themed markdown dump</Heading>
          </div>
          <div className={styles.capabilityGrid}>
            {capabilityCards.map((card) => (
              <Link key={card.title} className={styles.capabilityCard} to={card.href}>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
                <span>Open section</span>
              </Link>
            ))}
          </div>
        </section>

        <section className={styles.bottomBand}>
          <div>
            <p>Next step</p>
            <Heading as="h2">Use the docs like a platform map</Heading>
            <p>
              Start with the documentation index, then move through models,
              inference, deployment, and validation without fighting the site.
            </p>
          </div>
          <div className={styles.bottomActions}>
            <Link className={clsx('button button--primary', styles.primaryCta)} to="/docs/DOCS_INDEX">
              Open documentation index
            </Link>
            <Link className={clsx('button button--secondary', styles.secondaryCta)} to="https://github.com/matthewvaishnav/computational-pathology-research">
              View repository
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  );
}
