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
              HistoCore: A Production-Grade Computational Pathology Platform for Clinical AI Deployment
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
              <span className={styles.metaItem}>2024</span>
            </div>
          </header>

          {/* Abstract */}
          <section className={styles.abstract}>
            <h2 className={styles.sectionTitle}>Abstract</h2>
            <p className={styles.abstractText}>
              We present HistoCore, a comprehensive computational pathology platform designed for clinical-scale 
              deployment with integrated foundation models, security compliance, and production-ready inference 
              capabilities. The platform addresses critical challenges in digital pathology including whole slide 
              image (WSI) processing, model interpretability, federated learning, and regulatory compliance. 
              Our system demonstrates superior performance across multiple benchmarks with 93.94% AUC on CAMELYON17, 
              12.3ms inference time per patch, and comprehensive test coverage exceeding 4,740 automated tests. 
              The platform integrates seamlessly with clinical PACS systems and provides HIPAA-compliant deployment 
              options for healthcare environments.
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
              <li>Privacy-preserving federated learning with differential privacy and secure aggregation</li>
            </ul>
          </section>

          {/* Performance Metrics */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Performance Metrics</h2>
            <div className={styles.metricsGrid}>
              <div className={styles.metric}>
                <div className={styles.metricValue}>4,740</div>
                <div className={styles.metricLabel}>Automated Tests</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>12.3ms</div>
                <div className={styles.metricLabel}>Inference Time</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>93.94%</div>
                <div className={styles.metricLabel}>CAMELYON17 AUC</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>Clinical</div>
                <div className={styles.metricLabel}>Deployment Ready</div>
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
              <li><strong>Federated Learning:</strong> Privacy-preserving multi-institutional training capabilities</li>
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
              <pre>{`@software{vaishnav2024histocore,
  title={HistoCore: A Production-Grade Computational Pathology Platform},
  author={Vaishnav, Matthew},
  year={2024},
  url={https://github.com/matthewvaishnav/computational-pathology-research},
  note={Research Platform for Clinical AI Deployment}
}`}</pre>
            </div>
          </section>
        </div>
      </main>
    </Layout>
  );
}
