#!/usr/bin/env python3
"""
Generate architecture diagrams for HistoCore documentation.

This script creates various architecture diagrams in different formats:
- Mermaid diagrams for documentation
- PNG/SVG exports for presentations
- Interactive HTML for exploration

Usage:
    python scripts/generate_architecture_diagrams.py [--format mermaid|png|svg|html]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys


class ArchitectureDiagramGenerator:
    """Generate architecture diagrams for HistoCore."""
    
    def __init__(self, output_dir: Path = Path("docs/diagrams")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_system_overview(self) -> str:
        """Generate high-level system overview diagram."""
        return '''
graph TB
    subgraph "Input Layer"
        WSI[🔬 Whole-Slide Images<br/>96×96 patches<br/>N patches per slide]
        GEN[🧬 Genomic Data<br/>2000 gene expressions<br/>Continuous values]
        CLI[📋 Clinical Text<br/>Medical notes<br/>Variable length]
    end
    
    subgraph "Feature Extraction"
        WSI_FE[ResNet-50 Feature Extractor<br/>1024-dim features per patch]
        GEN_NORM[Gene Expression Normalization<br/>Log-transform + Z-score]
        CLI_TOK[Clinical Text Tokenization<br/>WordPiece tokenizer]
    end
    
    subgraph "Modality Encoders"
        WSI_ENC[🎯 WSI Encoder<br/>Transformer + Attention Pooling<br/>8.5M parameters]
        GEN_ENC[🧮 Genomic Encoder<br/>Deep MLP + BatchNorm<br/>2.1M parameters]
        CLI_ENC[📝 Clinical Encoder<br/>Transformer + CLS token<br/>12.3M parameters]
    end
    
    subgraph "Cross-Modal Fusion"
        FUSION[🔗 Cross-Modal Attention<br/>All-to-all modality attention<br/>3.2M parameters]
        MISSING[❓ Missing Modality Handler<br/>Graceful degradation<br/>Zero-masking]
    end
    
    subgraph "Task-Specific Heads"
        CLASS[🎯 Classification Head<br/>Multi-class disease prediction<br/>1.5M parameters]
        SURV[📊 Survival Prediction<br/>Cox proportional hazards<br/>0.8M parameters]
    end
    
    subgraph "Output"
        PRED[📈 Predictions<br/>Disease probabilities<br/>Survival curves<br/>Attention maps]
    end
    
    WSI --> WSI_FE --> WSI_ENC
    GEN --> GEN_NORM --> GEN_ENC
    CLI --> CLI_TOK --> CLI_ENC
    
    WSI_ENC --> FUSION
    GEN_ENC --> FUSION
    CLI_ENC --> FUSION
    
    FUSION --> MISSING
    MISSING --> CLASS
    MISSING --> SURV
    
    CLASS --> PRED
    SURV --> PRED
    
    style WSI fill:#e1f5fe
    style GEN fill:#f3e5f5
    style CLI fill:#e8f5e8
    style FUSION fill:#fff3e0
    style PRED fill:#f1f8e9
'''

    def generate_cross_modal_attention(self) -> str:
        """Generate detailed cross-modal attention diagram."""
        return '''
graph LR
    subgraph "Input Modalities"
        WSI_EMB[WSI Embedding<br/>256-dim]
        GEN_EMB[Genomic Embedding<br/>256-dim]
        CLI_EMB[Clinical Embedding<br/>256-dim]
    end
    
    subgraph "Pairwise Attention"
        WSI_TO_GEN[WSI → Genomic<br/>Q: WSI, K,V: Genomic]
        WSI_TO_CLI[WSI → Clinical<br/>Q: WSI, K,V: Clinical]
        GEN_TO_WSI[Genomic → WSI<br/>Q: Genomic, K,V: WSI]
        GEN_TO_CLI[Genomic → Clinical<br/>Q: Genomic, K,V: Clinical]
        CLI_TO_WSI[Clinical → WSI<br/>Q: Clinical, K,V: WSI]
        CLI_TO_GEN[Clinical → Genomic<br/>Q: Clinical, K,V: Genomic]
    end
    
    subgraph "Fusion"
        CONCAT[Concatenate All Outputs<br/>[WSI', GEN', CLI']<br/>768-dim total]
        PROJ[Linear Projection<br/>768 → 256 dim<br/>+ Layer Normalization]
    end
    
    WSI_EMB --> WSI_TO_GEN
    WSI_EMB --> WSI_TO_CLI
    GEN_EMB --> GEN_TO_WSI
    GEN_EMB --> GEN_TO_CLI
    CLI_EMB --> CLI_TO_WSI
    CLI_EMB --> CLI_TO_GEN
    
    WSI_TO_GEN --> CONCAT
    WSI_TO_CLI --> CONCAT
    GEN_TO_WSI --> CONCAT
    GEN_TO_CLI --> CONCAT
    CLI_TO_WSI --> CONCAT
    CLI_TO_GEN --> CONCAT
    
    CONCAT --> PROJ
    PROJ --> FUSED[Fused Representation<br/>256-dim]
    
    style WSI_EMB fill:#e1f5fe
    style GEN_EMB fill:#f3e5f5
    style CLI_EMB fill:#e8f5e8
    style FUSED fill:#f1f8e9
'''

    def generate_training_pipeline(self) -> str:
        """Generate training pipeline diagram."""
        return '''
graph TB
    subgraph "Data Loading"
        BATCH[📦 Batch Loader<br/>Multi-modal batches<br/>Dynamic padding<br/>Missing data handling]
        AUG[🔄 Data Augmentation<br/>WSI: rotation, flip, color<br/>Genomic: noise injection<br/>Clinical: synonym replacement]
    end
    
    subgraph "Forward Pass"
        ENCODE[🧠 Modality Encoding<br/>Parallel processing<br/>GPU acceleration<br/>Mixed precision (FP16)]
        FUSE[🔗 Cross-Modal Fusion<br/>Attention computation<br/>Gradient checkpointing<br/>Memory optimization]
        PREDICT[🎯 Task Prediction<br/>Multi-task heads<br/>Shared representations<br/>Task-specific losses]
    end
    
    subgraph "Optimization"
        BACKWARD[⬅️ Backward Pass<br/>Gradient computation<br/>Automatic differentiation<br/>Memory efficient]
        CLIP[✂️ Gradient Clipping<br/>Max norm = 1.0<br/>Prevent exploding gradients<br/>Stable training]
        UPDATE[🔄 Parameter Update<br/>AdamW optimizer<br/>Learning rate scheduling<br/>Weight decay]
    end
    
    BATCH --> AUG
    AUG --> ENCODE
    ENCODE --> FUSE
    FUSE --> PREDICT
    PREDICT --> BACKWARD
    BACKWARD --> CLIP
    CLIP --> UPDATE
    UPDATE --> BATCH
    
    style BATCH fill:#e3f2fd
    style FUSE fill:#fff3e0
    style UPDATE fill:#e8f5e8
'''

    def generate_clinical_deployment(self) -> str:
        """Generate clinical deployment architecture."""
        return '''
graph TB
    subgraph "Clinical Data Sources"
        PACS[🏥 PACS System<br/>DICOM WSI files<br/>Medical imaging<br/>Metadata]
        EHR[📋 Electronic Health Records<br/>HL7 FHIR format<br/>Clinical notes<br/>Patient history]
        LAB[🧪 Laboratory Systems<br/>Genomic sequencing<br/>Biomarker data<br/>Test results]
    end
    
    subgraph "AI Processing Pipeline"
        PREPROCESS[⚙️ Preprocessing Service<br/>Data validation<br/>Format standardization<br/>Quality control]
        INFERENCE[🧠 AI Inference Engine<br/>Model serving<br/>Batch processing<br/>Real-time prediction]
        POSTPROCESS[📊 Post-processing<br/>Result validation<br/>Confidence scoring<br/>Uncertainty quantification]
    end
    
    subgraph "Clinical Decision Support"
        INTERPRET[🔍 Result Interpretation<br/>Attention visualization<br/>Explainable AI<br/>Clinical relevance]
        REPORT[📄 Report Generation<br/>Structured reports<br/>PDF generation<br/>DICOM SR creation]
        ALERT[🚨 Clinical Alerts<br/>Critical findings<br/>Notification system<br/>Workflow integration]
    end
    
    PACS --> PREPROCESS
    EHR --> PREPROCESS
    LAB --> PREPROCESS
    
    PREPROCESS --> INFERENCE
    INFERENCE --> POSTPROCESS
    POSTPROCESS --> INTERPRET
    INTERPRET --> REPORT
    REPORT --> ALERT
    
    style PACS fill:#e3f2fd
    style EHR fill:#f3e5f5
    style LAB fill:#e8f5e8
    style INFERENCE fill:#fff3e0
    style REPORT fill:#f1f8e9
'''

    def save_mermaid_diagram(self, name: str, content: str) -> Path:
        """Save Mermaid diagram to file."""
        output_file = self.output_dir / f"{name}.mmd"
        
        mermaid_content = f"""```mermaid
{content.strip()}
```"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
            
        print(f"✅ Generated {output_file}")
        return output_file

    def export_to_png(self, mermaid_file: Path) -> Optional[Path]:
        """Export Mermaid diagram to PNG using mermaid-cli."""
        try:
            png_file = mermaid_file.with_suffix('.png')
            cmd = ['mmdc', '-i', str(mermaid_file), '-o', str(png_file), '-t', 'neutral', '-b', 'white']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Exported {png_file}")
                return png_file
            else:
                print(f"❌ Failed to export {png_file}: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("❌ mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli")
            return None

    def generate_all_diagrams(self, export_format: str = "mermaid") -> Dict[str, Path]:
        """Generate all architecture diagrams."""
        diagrams = {
            "system_overview": self.generate_system_overview(),
            "cross_modal_attention": self.generate_cross_modal_attention(),
            "training_pipeline": self.generate_training_pipeline(),
            "clinical_deployment": self.generate_clinical_deployment(),
        }
        
        generated_files = {}
        
        for name, content in diagrams.items():
            # Always generate Mermaid files
            mermaid_file = self.save_mermaid_diagram(name, content)
            generated_files[name] = mermaid_file
            
            # Export to other formats if requested
            if export_format == "png":
                png_file = self.export_to_png(mermaid_file)
                if png_file:
                    generated_files[f"{name}_png"] = png_file
                    
        return generated_files

    def generate_diagram_index(self, generated_files: Dict[str, Path]) -> Path:
        """Generate an index file listing all diagrams."""
        index_file = self.output_dir / "README.md"
        
        content = """# Architecture Diagrams

This directory contains generated architecture diagrams for HistoCore.

## Available Diagrams

### System Overview
- **File**: `system_overview.mmd`
- **Description**: High-level system architecture showing data flow from input modalities through processing to predictions
- **Components**: Input layer, encoders, fusion, task heads

### Cross-Modal Attention
- **File**: `cross_modal_attention.mmd`  
- **Description**: Detailed view of the cross-modal attention mechanism
- **Components**: Pairwise attention, fusion, projection

### Training Pipeline
- **File**: `training_pipeline.mmd`
- **Description**: Complete training workflow from data loading to optimization
- **Components**: Data loading, forward pass, loss computation, optimization

### Clinical Deployment
- **File**: `clinical_deployment.mmd`
- **Description**: Production deployment architecture for clinical environments
- **Components**: Data sources, processing pipeline, decision support

## Usage

### Viewing Diagrams
1. **GitHub/GitLab**: Diagrams render automatically in markdown files
2. **VS Code**: Install Mermaid Preview extension
3. **Online**: Copy content to [mermaid.live](https://mermaid.live)

### Exporting to Images
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export to PNG
mmdc -i system_overview.mmd -o system_overview.png -t neutral -b white

# Export to SVG  
mmdc -i system_overview.mmd -o system_overview.svg -t neutral -b white
```

### Regenerating Diagrams
```bash
# Generate all diagrams
python scripts/generate_architecture_diagrams.py

# Generate with PNG export
python scripts/generate_architecture_diagrams.py --format png
```

## Generated Files

"""
        
        for name, file_path in generated_files.items():
            if not name.endswith('_png'):
                content += f"- `{file_path.name}`\n"
        
        content += f"""
---

**Last Generated**: {Path(__file__).stat().st_mtime}  
**Generator**: `scripts/generate_architecture_diagrams.py`  
**Status**: Auto-generated - do not edit manually ⚠️
"""
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Generated diagram index: {index_file}")
        return index_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate HistoCore architecture diagrams")
    parser.add_argument(
        "--format", 
        choices=["mermaid", "png", "svg"], 
        default="mermaid",
        help="Output format (default: mermaid)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/diagrams"),
        help="Output directory (default: docs/diagrams)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ArchitectureDiagramGenerator(args.output_dir)
    
    # Generate diagrams
    print(f"🎨 Generating architecture diagrams in {args.output_dir}")
    generated_files = generator.generate_all_diagrams(args.format)
    
    # Generate index
    generator.generate_diagram_index(generated_files)
    
    print(f"\n✅ Generated {len(generated_files)} diagram files")
    print(f"📁 Output directory: {args.output_dir}")
    
    if args.format == "png":
        print("\n💡 Tip: Install mermaid-cli for PNG export:")
        print("   npm install -g @mermaid-js/mermaid-cli")


if __name__ == "__main__":
    main()