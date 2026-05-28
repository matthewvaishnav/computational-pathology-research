import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Computational Pathology AI Research Framework',
  description: 'Whole-slide pathology AI, PANDA slide-level MIL benchmarking, TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, and reproducible computational pathology research infrastructure.',
  base: '/computational-pathology-research/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,
  srcExclude: [
    'ABOUT.md',
    'ANALYSIS_SYSTEM.md',
    'API_REFERENCE.md',
    'ARCHITECTURE.md',
    'ARCHITECTURE_MIGRATION.md',
    'AWS_AZURE_BAA_GUIDE.md',
    'BENCHMARK_SYSTEM.md',
    'CHANGELOG.md',
    'DOCS_INDEX.md',
    'EXPERIMENTS.md',
    'FAILURE_ANALYSIS.md',
    'FAIR_WEIGHTS_HYBRID_PROTOCOL.md',
    'FAIR_WEIGHTS_H_IMPLEMENTATION_STATUS.md',
    'FAIR_WEIGHTS_H_PCAM_FEDERATED_SMOKE_REPORT.md',
    'FAIR_WEIGHTS_H_SYNTHETIC_CAMELYON17_SMOKE_REPORT.md',
    'FAIR_WEIGHTS_H_SYNTHETIC_REPORT.md',
    'GETTING_STARTED.md',
    'PCAM_BENCHMARK_RESULTS.md',
    'PCAM_COMPARISON_GUIDE.md',
    'PCAM_CROSS_VALIDATION.md',
    'PCAM_FAILURE_ANALYSIS.md',
    'PCAM_FULLSCALE_GUIDE.md',
    'PCAM_REAL_RESULTS.md',
    'PERFORMANCE_COMPARISON.md',
    'PERFORMANCE.md',
    'PLATFORM_OVERVIEW.md',
    'REPOSITORY_OVERVIEW.md',
    'SECURITY_HARDENING.md',
    'TESTING_SUMMARY.md',
    'TRANSNNMIL_V2_ARCHITECTURE.md',
    'TRANSNNMIL_V2_STATUS.md',
    'TRANSNNMIL_V2_TRAINING.md',
    'archive/**',
    'api/**',
    'demo/**',
    'deployment/**',
    'diagrams/**',
    'federated_learning/**',
    'operations/**',
    'regulatory/**',
    'security/**',
    'training/**'
  ],
  themeConfig: {
    siteTitle: 'Computational Pathology AI Research',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Overview', link: '/overview/' },
      { text: 'Research', link: '/research/literature-positioning' },
      { text: 'Models', link: '/models/' },
      { text: 'Federated', link: '/federated/pathologyfl' },
      { text: 'Validation', link: '/validation/' },
      { text: 'Results', link: '/results/pcam-results' },
      { text: 'Roadmap', link: '/roadmap/' }
    ],
    sidebar: [
      { text: 'Overview', items: [
        { text: 'Home', link: '/' },
        { text: 'What this project is', link: '/overview/' },
        { text: 'Repository structure', link: '/repository-overview' },
        { text: 'Platform overview', link: '/platform-overview' },
        { text: 'Claim status', link: '/overview/claim-status' }
      ]},
      { text: 'Research', items: [
        { text: 'Literature positioning', link: '/research/literature-positioning' },
        { text: 'FAIR-WEIGHTS-H stress result', link: '/research/fair-weights-h-stress-result' }
      ]},
      { text: 'Quickstart', items: [
        { text: 'Getting started', link: '/getting-started' },
        { text: 'Reproducing experiments', link: '/quickstart/reproducing-experiments' },
        { text: 'Running benchmarks', link: '/quickstart/running-benchmarks' }
      ]},
      { text: 'Models', items: [
        { text: 'Models overview', link: '/models/' },
        { text: 'TransnnMIL v2.0', link: '/models/transnnmil-v2' },
        { text: 'Foundation encoders', link: '/models/foundation-encoders' },
        { text: 'Model cards', link: '/models/model-cards' }
      ]},
      { text: 'Federated Learning', items: [
        { text: 'PathologyFL', link: '/federated/pathologyfl' },
        { text: 'FAIR-WEIGHTS-H', link: '/theory/fair-weights-h' },
        { text: 'Implementation status', link: '/theory/implementation-status' },
        { text: 'Privacy and DP', link: '/federated/privacy-dp' },
        { text: 'Secure aggregation', link: '/federated/secure-aggregation' },
        { text: 'Byzantine robustness', link: '/federated/byzantine-robustness' }
      ]},
      { text: 'Validation', items: [
        { text: 'Validation overview', link: '/validation/' },
        { text: 'Synthetic validation', link: '/validation/synthetic-report' },
        { text: 'PCam smoke report', link: '/validation/pcam-smoke-report' },
        { text: 'PCam benchmark plan', link: '/validation/PCAM_FEDERATED_BENCHMARK_PLAN' },
        { text: 'Camelyon17 plan', link: '/validation/camelyon17-plan' }
      ]},
      { text: 'Results', items: [
        { text: 'PCam results', link: '/results/pcam-results' },
        { text: 'Performance comparison', link: '/results/performance-comparison' }
      ]},
      { text: 'Engineering', items: [
        { text: 'Architecture', link: '/engineering/architecture' },
        { text: 'Testing status', link: '/engineering/testing-status' },
        { text: 'Benchmark system', link: '/engineering/benchmark-system' },
        { text: 'Security hardening', link: '/engineering/security-hardening' },
        { text: 'Deployment', link: '/engineering/deployment' }
      ]},
      { text: 'Roadmap', items: [
        { text: 'Roadmap', link: '/roadmap/' },
        { text: 'Limitations', link: '/roadmap/limitations' },
        { text: 'Changelog', link: '/changelog' }
      ]}
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/matthewvaishnav/computational-pathology-research' }
    ],
    search: { provider: 'local' },
    footer: {
      message: 'Research documentation. Not clinical validation or regulatory clearance.',
      copyright: 'Computational pathology, federated oncology learning, and mathematical validation infrastructure.'
    }
  },
  markdown: { math: true }
})