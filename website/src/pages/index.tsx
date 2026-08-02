import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const repositoryUrl =
  'https://github.com/matthewvaishnav/computational-pathology-research';

export default function Home(): ReactNode {
  return (
    <Layout
      title="Computational Pathology Research"
      description="Independent neural-network research on paired acquisition, representation auditing, and reproducible computational pathology experiments.">
      <main className={styles.main}>
        <div className={styles.paperContainer}>
          <header className={styles.paperHeader}>
            <Heading as="h1" className={styles.paperTitle}>
              Computational Pathology Research
            </Heading>

            <p className={styles.heroLead}>
              Independent neural-network research on paired acquisition,
              scanner-associated representation structure, and reproducible
              computational pathology experiments.
            </p>

            <div className={styles.authors}>
              <span className={styles.author}>Matthew Vaishnav</span>
            </div>

            <div className={styles.paperMeta}>
              <span className={styles.metaItem}>Research Engineering</span>
              <span className={styles.metaItem}>•</span>
              <span className={styles.metaItem}>Audited Evidence</span>
              <span className={styles.metaItem}>•</span>
              <span className={styles.metaItem}>Updated August 2026</span>
            </div>
          </header>

          <section className={styles.quickNav}>
            <Link to="/docs/CURRENT_STATUS" className={styles.quickNavCard}>
              <strong>Current Status</strong>
              <span>Promoted evidence, exploratory work, and next steps</span>
            </Link>
            <Link
              to={`${repositoryUrl}/blob/main/CLAIM_BOUNDARY.md`}
              className={styles.quickNavCard}>
              <strong>Claim Boundary</strong>
              <span>Authoritative public interpretation and explicit non-claims</span>
            </Link>
            <Link to="/docs/PORTFOLIO_SUMMARY" className={styles.quickNavCard}>
              <strong>Research Portfolio</strong>
              <span>Neural methods, experiment design, and reproducibility engineering</span>
            </Link>
            <Link to={repositoryUrl} className={styles.quickNavCard}>
              <strong>Source Code</strong>
              <span>Repository, experiments, tests, and evidence packages</span>
            </Link>
          </section>

          <section className={styles.abstract}>
            <h2 className={styles.sectionTitle}>What this is</h2>
            <p className={styles.abstractText}>
              The central research line is Paired-Acquisition Neural
              Factorization: a study of whether matched scans of the same tissue
              regions can support partial separation of tissue-associated and
              scanner-associated information in frozen pathology embeddings.
              The repository also documents whole-slide modeling, source-effect
              studies, provenance audits, and experiment infrastructure.
            </p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Current Promoted Evidence</h2>
            <div className={styles.metricsGrid}>
              <div className={styles.metric}>
                <div className={styles.metricValue}>175/175</div>
                <div className={styles.metricLabel}>SCORPION Capacity Fits</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>450/450</div>
                <div className={styles.metricLabel}>Canine Factorial Cells</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>48</div>
                <div className={styles.metricLabel}>SCORPION Slide Blocks</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue}>5</div>
                <div className={styles.metricLabel}>Matched Scanner Views</div>
              </div>
            </div>
            <p className={styles.sectionText}>
              The bounded result is partial structured separation under the
              tested protocols: lower linearly recoverable scanner identity in a
              tissue-oriented branch, preserved same-region retrieval, and strong
              scanner information in an explicit acquisition branch. This does
              not prove pure biological factors, complete disentanglement, or
              clinical value.
            </p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Research Threads</h2>
            <div className={styles.researchGrid}>
              <div className={styles.researchCard}>
                <h3>Paired Acquisition</h3>
                <p>
                  Same-region multi-scanner representation learning, capacity
                  controls, leakage audits, retrieval, and explicit acquisition
                  modeling.
                </p>
              </div>
              <div className={styles.researchCard}>
                <h3>Whole-Slide Modeling</h3>
                <p>
                  TransnnMIL repair and controlled evaluation. Historical fusion
                  and topology scores remain withdrawn pending matched reruns.
                </p>
              </div>
              <div className={styles.researchCard}>
                <h3>Reproducibility</h3>
                <p>
                  Deterministic run identities, frozen hashes, append-only
                  ledgers, fail-closed resume, and forward-valid evidence
                  releases.
                </p>
              </div>
              <div className={styles.researchCard}>
                <h3>Scientific Audit</h3>
                <p>
                  Leakage, pseudoreplication, confounding, capacity mismatch,
                  provenance, and claim-boundary review with negative results
                  retained.
                </p>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Active but Unpromoted</h2>
            <ul className={styles.architectureList}>
              <li>
                <strong>Paired affine comparison:</strong> prospective translation,
                Procrustes, affine, and ridge-affine evaluation.
              </li>
              <li>
                <strong>Crossed-target diagnostics:</strong> synthetic scanner-prototype
                and identity-disjoint intervention studies.
              </li>
              <li>
                <strong>TransnnMIL reruns:</strong> matched comparisons of repaired
                fusion against standalone and controlled baselines.
              </li>
            </ul>
            <p className={styles.sectionText}>
              Draft PRs and smoke runs are engineering or exploratory records,
              not promoted pathology-domain evidence.
            </p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Explicit Boundaries</h2>
            <ul className={styles.architectureList}>
              <li>No novelty, priority, or patentability claim.</li>
              <li>No pure-biological-factor or complete-invariance claim.</li>
              <li>No clinical, regulatory, hospital, PACS, privacy, or deployment validation.</li>
              <li>No state-of-the-art or universal-superiority claim.</li>
              <li>No promotion of unvalidated smoke, draft-PR, or historical results.</li>
            </ul>
          </section>
        </div>
      </main>
    </Layout>
  );
}
