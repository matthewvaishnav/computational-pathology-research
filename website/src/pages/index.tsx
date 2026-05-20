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
          {/* Medical Paper Header */}
          <header className={styles.paperHeader}>
            <Heading as="h1" className={styles.paperTitle}>
              Computational Pathology Research Platform: Production-Grade Framework for Clinical AI Deployment
            </Heading>

            <div className={styles.authors}>
              <span className={styles.author}>Matthew Vaishnav</span>
            </div>

            <div className={styles.affiliation}>
              Computational Pathology Research Laboratory
            </div>

            <div className={styles.paperMeta}>
              <span className={styles.metaItem}>Research Platform</span>
              <span className={styles.metaItem}>•</span>
              <span className={styles.metaItem}>Version 2.0</span>
              <span className={styles.metaItem}>•</span>
              <span className={styles.metaItem}>2026</span>
            </div>
          </header>

          {/* Abstract */}
          <section className={styles.abstract}>
            <h2 className={styles.sectionTitle}>Abstract</h2>
            <p className={styles.abstractText}>
              This platform provides a comprehensive computational pathology framework designed for clinical-scale
              deployment with integrated foundation models, security compliance, and production-ready inference
              capabilities. The system addresses critical challenges in digital pathology including whole slide
              image (WSI) processing, model interpretability, federated learning, and regulatory compliance.
              Key achievements include 93.94% AUC on PCam (327K patches, #1 vs 10 published baselines), 85.26%
              accuracy on 32,768-sample test set, and 5,071+ automated tests with comprehensive coverage. The
              platform features a hybrid architecture with clean separation of concerns, HIPAA-compliant deployment
              with clinical PACS integration, advanced federated learning with pathology-specific aggregation
              strategies (PathologyFL), and a novel Distributed Medical Intelligence (DMI) system enabling
              multi-institutional collaboration without compromising patient privacy.
            </p>
          </section>

          {/* Key Contributions */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Key Contributions</h2>
            <ul className={styles.contributionsList}>
              <li>Production-ready WSI processing pipeline with optimized batch inference and uncertainty quantification</li>
              <li>Seamless integration with foundation models (UNI, Phikon, GigaPath) and custom architectures</li>
              <li>Comprehensive security framework with HIPAA compliance and clinical workflow integration</li>
              <li>Rigorous benchmarking system with statistical validation and comparative analysis</li>
              <li>Direct PACS integration with DICOM handling for pathology departments</li>
              <li>Advanced federated learning integration with pathology-specific aggregation strategies</li>
              <li>Distributed Medical Intelligence (DMI) system for multi-institutional collaboration</li>
            </ul>
          </section>

          {/* Performance Metrics */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Performance Metrics</h2>
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
                <div className={styles.metricValue}>Hybrid</div>
                <div className={styles.metricLabel}>Architecture</div>
              </div>
            </div>
          </section>

          {/* System Architecture */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>System Architecture</h2>
            <p className={styles.sectionText}>
              The HistoCore platform is built on a modular architecture supporting multiple deployment scenarios
              from research environments to clinical production systems. The core components include:
            </p>
            <ul className={styles.architectureList}>
              <li><strong>Foundation Model Integration:</strong> Support for UNI, Phikon, GigaPath, and custom architectures</li>
              <li><strong>Clinical Inference Engine:</strong> Optimized batch processing with real-time performance monitoring</li>
              <li><strong>Security Layer:</strong> HIPAA-compliant data handling with comprehensive audit logging</li>
              <li><strong>Benchmarking Framework:</strong> Statistical validation and comparative analysis tools</li>
              <li><strong>PACS Integration:</strong> Direct clinical system integration with DICOM support</li>
              <li><strong>Federated Learning Module:</strong> Pathology-aware FL integration with Flower framework</li>
              <li><strong>Distributed Medical Intelligence:</strong> Multi-institutional knowledge collaboration system</li>
            </ul>
          </section>

          {/* Federated Learning & DMI */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Federated Learning Integration & Distributed Medical Intelligence</h2>
            <p className={styles.sectionText}>
              HistoCore features advanced federated learning capabilities that enable multi-hospital AI training
              without sharing patient data, combined with a novel Distributed Medical Intelligence (DMI) system
              for medical knowledge collaboration.
            </p>

            <h3 className={styles.subsectionTitle}>PathologyFL: Expertise-Weighted Aggregation</h3>
            <ul className={styles.architectureList}>
              <li><strong>Hospital Expertise Weighting:</strong> Cancer centers receive higher weights than community hospitals</li>
              <li><strong>Cancer-Type Specific Strategies:</strong> Specialized aggregation for breast, lung, prostate, and colorectal cancers</li>
              <li><strong>Slide Quality Assessment:</strong> Automatic weighting based on image sharpness, stain consistency, and label confidence</li>
              <li><strong>Attention-Aware Aggregation:</strong> Different strategies for attention layers vs. standard parameters</li>
            </ul>

            <h3 className={styles.subsectionTitle}>Production Security & Privacy</h3>
            <ul className={styles.architectureList}>
              <li><strong>Differential Privacy (DP-SGD):</strong> Gradient clipping and calibrated noise with privacy budget tracking</li>
              <li><strong>Secure Aggregation:</strong> Homomorphic encryption using TenSEAL for encrypted gradient aggregation</li>
              <li><strong>Byzantine Robustness:</strong> Krum algorithm and coordinate-wise median for malicious client detection</li>
              <li><strong>HIPAA Compliance:</strong> Tamper-evident audit logging and regulatory compliance</li>
            </ul>

            <h3 className={styles.subsectionTitle}>Distributed Medical Intelligence (DMI)</h3>
            <ul className={styles.architectureList}>
              <li><strong>Medical Expertise Calculation:</strong> Weights based on board certifications, publications, and diagnostic accuracy</li>
              <li><strong>Collective Knowledge Synthesis:</strong> Aggregates medical insights across institutions without data sharing</li>
              <li><strong>Specialization Matching:</strong> Routes cases to hospitals with relevant expertise</li>
              <li><strong>Multi-Institutional Collaboration:</strong> Enables knowledge sharing while preserving institutional autonomy</li>
            </ul>
          </section>

          {/* Documentation Links */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Documentation & Resources</h2>
            <div className={styles.linksGrid}>
              <Link to="/docs/GETTING_STARTED" className={styles.docLink}>
                <h3>Getting Started</h3>
                <p>Installation, setup, and basic usage instructions</p>
              </Link>
              <Link to="/docs/FOUNDATION_MODELS" className={styles.docLink}>
                <h3>Foundation Models</h3>
                <p>Integration guide for UNI, Phikon, and GigaPath models</p>
              </Link>
              <Link to="/docs/DEPLOYMENT" className={styles.docLink}>
                <h3>Clinical Deployment</h3>
                <p>Production deployment and PACS integration guide</p>
              </Link>
              <Link to="/docs/BENCHMARK_SYSTEM" className={styles.docLink}>
                <h3>Benchmarking</h3>
                <p>Performance evaluation and validation frameworks</p>
              </Link>
              <Link to="/docs/SECURITY_HARDENING" className={styles.docLink}>
                <h3>Security & Compliance</h3>
                <p>HIPAA compliance and security implementation</p>
              </Link>
              <Link to="https://github.com/matthewvaishnav/computational-pathology-research" className={styles.docLink}>
                <h3>Source Code</h3>
                <p>Complete implementation and research codebase</p>
              </Link>
            </div>
          </section>

          {/* Citation */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Citation</h2>
            <div className={styles.citation}>
              <pre>{`@software{vaishnav2026computational_pathology,
  title={Computational Pathology Research Platform: Production-Grade Framework for Clinical AI Deployment},
  author={Vaishnav, Matthew},
  year={2026},
  url={https://github.com/matthewvaishnav/computational-pathology-research},
  note={Research Platform v2.0 with PathologyFL and DMI}
}`}</pre>
            </div>
          </section>
        </div>
      </main>
    </Layout>
  );
}
