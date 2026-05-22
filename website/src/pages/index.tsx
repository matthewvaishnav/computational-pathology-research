import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

export default function Home(): ReactNode {
  return (
    <Layout
      title="Computational Pathology Research Platform"
      description="A comprehensive computational pathology research platform with foundation model integration, clinical deployment capabilities, and rigorous benchmarking systems.">

      <main className={styles.main}>
        <div className={styles.paperContainer}>
          <header className={styles.paperHeader}>
            <Heading as="h1" className={styles.paperTitle}>
              Computational Pathology Research Platform
            </Heading>

            <p className={styles.heroLead}>
              Production-grade research framework for multiple-instance learning, whole-slide pathology AI, and privacy-preserving federated learning.
            </p>

            <div className={styles.authors}>
              <span className={styles.author}>Matthew Vaishnav</span>
            </div>

            <div className={styles.paperMeta}>
              <span className={styles.metaItem}>Research Platform</span>
              <span className={styles.metaItem}>•</span>
              <span className={styles.metaItem}>Version 2.0</span>
              <span className={styles.metaItem}>•</span>
              <span className={styles.metaItem}>2026</span>
            </div>
          </header>

          <section className={styles.quickNav}>
            <Link to="/docs/PERFORMANCE_COMPARISON" className={styles.quickNavCard}>
              <strong>Results</strong>
              <span>Performance comparisons, benchmark context, and model metrics</span>
            </Link>
            <Link to="/docs/FAIR_WEIGHTS_HYBRID_PROTOCOL" className={styles.quickNavCard}>
              <strong>FAIR-WEIGHTS-H</strong>
              <span>Hybrid institutional weighting protocol and validation plan</span>
            </Link>
            <Link to="/docs/GETTING_STARTED" className={styles.quickNavCard}>
              <strong>Start Here</strong>
              <span>Install, run, and explore the platform</span>
            </Link>
            <Link to="https://github.com/matthewvaishnav/computational-pathology-research" className={styles.quickNavCard}>
              <strong>Source Code</strong>
              <span>Repository, tests, experiments, and implementation</span>
            </Link>
          </section>

          <section className={styles.abstract}>
            <h2 className={styles.sectionTitle}>What this is</h2>
            <p className={styles.abstractText}>
              This repository combines computational pathology model development, WSI processing, MIL architectures,
              federated learning infrastructure, clinical-integration adapters, and benchmark tooling. Key results
              include 93.94% AUC on PCam, 85.26% accuracy on the PCam test set, and 5,071+ automated tests. The
              platform includes PathologyFL for domain-specific federated learning and FAIR-WEIGHTS-H for auditable,
              evidence-based institutional weighting research.
            </p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Project Map</h2>
            <div className={styles.linksGrid}>
              <Link to="/docs/FRAMEWORK_OVERVIEW" className={styles.docLink}>
                <h3>Framework Overview</h3>
                <p>High-level map of the platform, modules, and research direction.</p>
              </Link>
              <Link to="/docs/ARCHITECTURE" className={styles.docLink}>
                <h3>Architecture</h3>
                <p>System architecture, pipeline structure, and major components.</p>
              </Link>
              <Link to="/docs/PERFORMANCE_COMPARISON" className={styles.docLink}>
                <h3>Performance</h3>
                <p>Benchmark results, model metrics, and comparative evaluation context.</p>
              </Link>
              <Link to="/docs/FAIR_WEIGHTS_HYBRID_PROTOCOL" className={styles.docLink}>
                <h3>FAIR-WEIGHTS-H</h3>
                <p>Hybrid institutional weighting with contribution, useful uniqueness, and subgroup constraints.</p>
              </Link>
              <Link to="/docs/BENCHMARK_SYSTEM" className={styles.docLink}>
                <h3>Benchmarking</h3>
                <p>Evaluation infrastructure, statistical validation, and experiment design.</p>
              </Link>
              <Link to="/docs/SECURITY_HARDENING" className={styles.docLink}>
                <h3>Security</h3>
                <p>Privacy, audit logging, hardening, and compliance-oriented infrastructure.</p>
              </Link>
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Verified Results & Infrastructure</h2>
            <div className={styles.metricsGrid}>
              <div className={styles.metric}>
                <div className={styles.metricValue}>5,071+</div>
                <div className={styles.metricLabel}>Automated Tests</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>93.94%</div>
                <div className={styles.metricLabel}>PCam AUC</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>85.26%</div>
                <div className={styles.metricLabel}>PCam Accuracy</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>12.2M</div>
                <div className={styles.metricLabel}>Model Parameters</div>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Core Research Threads</h2>
            <div className={styles.researchGrid}>
              <div className={styles.researchCard}>
                <h3>MIL & WSI Modeling</h3>
                <p>AttentionMIL, CLAM, TransMIL/TransnnMIL-style modeling, feature extraction, and whole-slide processing.</p>
              </div>
              <div className={styles.researchCard}>
                <h3>PathologyFL</h3>
                <p>Federated learning infrastructure for multi-site pathology training without sharing patient-level data.</p>
              </div>
              <div className={styles.researchCard}>
                <h3>FAIR-WEIGHTS-H</h3>
                <p>Hybrid institutional weighting replacing fixed prestige multipliers with contribution, quality, useful uniqueness, uncertainty, and subgroup-safety constraints.</p>
              </div>
              <div className={styles.researchCard}>
                <h3>Clinical Infrastructure</h3>
                <p>PACS/DICOM, FHIR adapters, audit logging, privacy infrastructure, and deployment-oriented APIs.</p>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>FAIR-WEIGHTS-H Institutional Weighting</h2>
            <p className={styles.sectionText}>
              DMI no longer needs to rely on fixed hospital-type prestige multipliers. FAIR-WEIGHTS-H models
              institutional influence as a constrained optimization problem over contribution, quality, useful
              uniqueness, uncertainty, and subgroup safety.
            </p>
            <div className={styles.equationBox}>
              <code>
                w_t = argmax Σᵢ wᵢ(φᵢ^Owen + λ_D Dᵢ^useful + λ_F Fᵢ + λ_Q Qᵢ − λ_S Sᵢ)
              </code>
            </div>
            <ul className={styles.architectureList}>
              <li><strong>Counterfactual contribution:</strong> grouped/Owen-style marginal utility estimates.</li>
              <li><strong>Useful uniqueness:</strong> distributional difference only helps when paired with quality and subgroup utility.</li>
              <li><strong>Subgroup constraints:</strong> representation and performance constraints are treated as binding validation requirements.</li>
              <li><strong>Implemented scaffold:</strong> weighting engine, explicit weighted aggregator, synthetic federation benchmark, perturbation suite, and markdown report generation.</li>
            </ul>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Federated Learning & Privacy</h2>
            <ul className={styles.architectureList}>
              <li><strong>Pathology-aware aggregation:</strong> cancer-type strategies, slide quality weighting, and attention-aware aggregation.</li>
              <li><strong>Explicit weighted aggregation:</strong> modular adapter for externally computed institutional weights.</li>
              <li><strong>Differential privacy:</strong> gradient clipping, calibrated noise, and privacy budget tracking.</li>
              <li><strong>Secure aggregation:</strong> encrypted gradient aggregation infrastructure.</li>
              <li><strong>Byzantine robustness:</strong> malicious-client detection via robust aggregation methods.</li>
            </ul>
          </section>
        </div>
      </main>
    </Layout>
  );
}
