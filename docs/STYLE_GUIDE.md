# Documentation Style Guide

This style guide defines the documentation standards for the Computational Pathology Research Framework. Following these guidelines ensures consistency, clarity, and professionalism across all project documentation.

## Table of Contents

- [Voice and Tone](#voice-and-tone)
- [Rationale for Voice Choice](#rationale-for-voice-choice)
- [Voice Guidelines](#voice-guidelines)
- [Examples and Usage](#examples-and-usage)
- [Special Cases](#special-cases)
- [Writing Best Practices](#writing-best-practices)
- [Markdown Conventions](#markdown-conventions)

## Voice and Tone

### Primary Voice: Singular First-Person

All documentation uses **singular first-person voice** ("I/my") rather than plural first-person ("we/our"). This accurately reflects that this is an individual's research project and portfolio work.

**Key Principle**: Distinguish between **author actions** (use "I") and **system capabilities** (use "the system/platform/model").

### Tone

- **Professional**: Maintain academic and technical rigor
- **Clear**: Use precise, unambiguous language
- **Accessible**: Explain complex concepts clearly
- **Confident**: Present work with appropriate confidence
- **Honest**: Acknowledge limitations and areas for improvement

## Rationale for Voice Choice

### Why Singular First-Person?

1. **Accuracy**: This is an individual's research project, not a team effort
2. **Portfolio Context**: The project serves as a demonstration of individual capabilities
3. **Academic Convention**: Individual research papers often use "I" for single-author work
4. **Clarity**: Clearly distinguishes author decisions from system behaviors
5. **Authenticity**: Honestly represents the project's authorship

### Historical Context

The project originally used plural first-person ("we/our") following common software documentation conventions. However, this created ambiguity about authorship and didn't accurately reflect the individual nature of the work. The voice was updated to singular first-person to better represent the project's context as individual research and portfolio work.

## Voice Guidelines

### Rule 1: Author Actions → "I"

When describing actions taken by the author (design decisions, implementation choices, research directions):

**Pattern**: `I [verb]`

✅ **Correct Examples**:
- "I built this system to address..."
- "I designed the architecture with modularity in mind"
- "I implemented cross-modal attention mechanisms"
- "I chose PyTorch for its flexibility"
- "I plan to benchmark against additional baselines"
- "I'm seeking collaborators for clinical validation"
- "I provide support for DICOM and WSI formats"

❌ **Incorrect Examples**:
- "We built this system to address..."
- "We designed the architecture with modularity in mind"
- "Our implementation uses cross-modal attention"

### Rule 2: System Capabilities → "the system/platform/model"

When describing what the system, platform, or model does (technical capabilities, automated processes, performance characteristics):

**Pattern**: `the [component] [verb]` or `[component] [verb]`

✅ **Correct Examples**:
- "The system processes whole-slide images in under 30 seconds"
- "The model achieves 95% accuracy on the PatchCamelyon dataset"
- "The platform supports DICOM, TIFF, and SVS formats"
- "The architecture handles missing modalities gracefully"
- "The inference pipeline maintains HIPAA compliance"
- "the platform integrates with existing PACS systems"

❌ **Incorrect Examples**:
- "I process whole-slide images in under 30 seconds" (system capability, not author action)
- "I achieve 95% accuracy" (model performance, not author action)
- "I support DICOM formats" (system capability, not author action)

### Rule 3: Possessive Adjectives

**Author's Work**: Use "my"
- "my approach to multimodal fusion"
- "my implementation of temporal reasoning"
- "my research focuses on..."
- "my design philosophy emphasizes..."

**System Components**: Use "the"
- "the testing strategy validates..."
- "the PACS integration supports..."
- "the model architecture consists of..."
- "the data pipeline handles..."

### Rule 4: Collaborative Contexts

When discussing partnerships, collaborations, or seeking contributors:

✅ **Correct Examples**:
- "I'm looking for clinical partners to validate the system"
- "I welcome contributions from researchers and developers"
- "I offer flexible deployment options for hospital partners"
- "I can work with your existing infrastructure"

❌ **Incorrect Examples**:
- "We're looking for clinical partners"
- "We welcome contributions"
- "We offer flexible deployment options"

### Rule 5: Comparative Contexts

When comparing with other systems or baselines:

✅ **Correct Examples**:
- "the platform achieves 94.2% accuracy compared to..."
- "The system outperforms baseline approaches"
- "My implementation shows improvements over..."

❌ **Incorrect Examples**:
- "the platform (Ours) achieves 94.2% accuracy"
- "Our system outperforms..."

## Examples and Usage

### Example 1: README Introduction

✅ **Correct**:
```markdown
# Computational Pathology Research Framework

I built this framework to advance multimodal fusion architectures in computational 
pathology. The system integrates whole-slide images, genomic profiles, and clinical 
text to improve diagnostic accuracy.

My research focuses on attention-based fusion mechanisms that handle missing 
modalities gracefully. The platform achieves state-of-the-art performance on 
benchmark datasets while maintaining clinical deployment requirements.
```

❌ **Incorrect**:
```markdown
# Computational Pathology Research Framework

We built this framework to advance multimodal fusion architectures in computational 
pathology. Our system integrates whole-slide images, genomic profiles, and clinical 
text to improve diagnostic accuracy.

Our research focuses on attention-based fusion mechanisms that handle missing 
modalities gracefully. We achieve state-of-the-art performance on benchmark 
datasets while maintaining clinical deployment requirements.
```

### Example 2: Feature Description

✅ **Correct**:
```markdown
## PACS Integration

I designed the PACS integration to work with existing hospital infrastructure. 
The system supports multiple vendor protocols (DICOM C-FIND, C-MOVE, WADO) and 
handles network failures gracefully.

My implementation includes:
- Automatic retry with exponential backoff
- Connection pooling for high throughput
- Comprehensive audit logging

The integration maintains HIPAA compliance and supports both on-premise and 
cloud deployments.
```

❌ **Incorrect**:
```markdown
## PACS Integration

We designed the PACS integration to work with existing hospital infrastructure. 
We support multiple vendor protocols (DICOM C-FIND, C-MOVE, WADO) and handle 
network failures gracefully.

Our implementation includes:
- Automatic retry with exponential backoff
- Connection pooling for high throughput
- Comprehensive audit logging

We maintain HIPAA compliance and support both on-premise and cloud deployments.
```

### Example 3: Performance Claims

✅ **Correct**:
```markdown
## Performance

The system achieves the following performance metrics on the PatchCamelyon dataset:

- Accuracy: 94.2%
- Inference time: 28ms per slide
- Memory usage: 2.1GB

I benchmarked the system against three baseline approaches and observed 
consistent improvements across all metrics. The model maintains accuracy 
even with missing modalities.
```

❌ **Incorrect**:
```markdown
## Performance

We achieve the following performance metrics on the PatchCamelyon dataset:

- Accuracy: 94.2%
- Inference time: 28ms per slide
- Memory usage: 2.1GB

We benchmarked against three baseline approaches and observed consistent 
improvements across all metrics. We maintain accuracy even with missing modalities.
```

### Example 4: Future Work

✅ **Correct**:
```markdown
## Future Directions

I plan to extend this work in several directions:

1. **Clinical Validation**: I'm seeking hospital partners to validate the system 
   in real-world clinical workflows
2. **Additional Modalities**: I plan to integrate radiology images and laboratory 
   results
3. **Federated Learning**: The system will support federated training across 
   multiple institutions

I welcome collaborations and contributions in these areas.
```

❌ **Incorrect**:
```markdown
## Future Directions

We plan to extend this work in several directions:

1. **Clinical Validation**: We're seeking hospital partners to validate our system 
   in real-world clinical workflows
2. **Additional Modalities**: We plan to integrate radiology images and laboratory 
   results
3. **Federated Learning**: Our system will support federated training across 
   multiple institutions

We welcome collaborations and contributions in these areas.
```

## Special Cases

### Case 1: Quotes and Examples

**Rule**: Leave quotes unchanged, even if they contain "we/our"

✅ **Correct**:
```markdown
**Q: "We're already working with [Competitor]"**

**A**: I understand you have existing relationships. The system is designed to 
complement rather than replace existing tools.
```

### Case 2: Code Comments

**Rule**: Leave code comments unchanged (they follow different conventions)

✅ **Correct**:
```python
# Move study to our server
def transfer_study(study_id: str) -> bool:
    """Transfer a study to the local server.
    
    I implemented this to support offline analysis workflows.
    """
    pass
```

### Case 3: Third-Party References

**Rule**: Leave references to other projects/teams unchanged

✅ **Correct**:
```markdown
PathML provides their own implementation of stain normalization. I chose to 
implement a custom approach that better fits my use case.
```

### Case 4: Passive Voice

**Rule**: Consider rewriting passive voice to active voice with "I"

✅ **Better**:
```markdown
I designed the approach to minimize computational overhead.
```

❌ **Acceptable but less clear**:
```markdown
The approach was designed to minimize computational overhead.
```

### Case 5: Ambiguous Contexts

**Rule**: When in doubt, prefer "the system" for technical capabilities

✅ **Correct**:
```markdown
The system achieves 95% accuracy on validation data.
```

❌ **Incorrect**:
```markdown
I achieve 95% accuracy on validation data.
```

**Reasoning**: The model/system achieves the accuracy, not the author personally.

## Writing Best Practices

### Clarity and Precision

- Use specific, concrete language
- Avoid vague terms like "very", "quite", "fairly"
- Define technical terms on first use
- Use consistent terminology throughout

### Active Voice

Prefer active voice over passive voice:

✅ **Better**: "I implemented the feature"
❌ **Weaker**: "The feature was implemented"

✅ **Better**: "The system processes images"
❌ **Weaker**: "Images are processed by the system"

### Conciseness

- Remove unnecessary words
- Use simple sentence structures
- Break complex sentences into multiple sentences
- Use bullet points for lists

### Technical Accuracy

- Verify all performance claims
- Include units for measurements
- Specify hardware/software versions
- Provide reproducible examples

### Accessibility

- Explain acronyms on first use
- Provide context for domain-specific terms
- Include examples for complex concepts
- Link to additional resources

## Markdown Conventions

### Headers

Use ATX-style headers with proper hierarchy:

```markdown
# Top-Level Header (Document Title)

## Major Section

### Subsection

#### Minor Subsection
```

### Code Blocks

Always specify the language for syntax highlighting:

```markdown
```python
def example():
    pass
```
```

### Links

Use descriptive link text:

✅ **Good**: See the [installation guide](docs/INSTALLATION.md) for details.
❌ **Bad**: Click [here](docs/INSTALLATION.md) for installation.

### Lists

Use consistent list formatting:

**Unordered lists**:
```markdown
- First item
- Second item
- Third item
```

**Ordered lists**:
```markdown
1. First step
2. Second step
3. Third step
```

### Emphasis

- Use **bold** for important terms and UI elements
- Use *italics* for emphasis and technical terms
- Use `code` for inline code, commands, and file names

### Tables

Use tables for structured data:

```markdown
| Metric | Value | Unit |
|--------|-------|------|
| Accuracy | 94.2% | % |
| Latency | 28 | ms |
```

## Enforcement and Review

### Pre-Commit Checks

Consider adding linting rules to catch "we/our" usage in new documentation:

```bash
# Example: Check for plural first-person in markdown files
grep -r "we\|our\|ours\|We\|Our" docs/ --include="*.md"
```

### Pull Request Review

Reviewers should verify:
- [ ] Documentation uses singular first-person voice correctly
- [ ] Author actions use "I"
- [ ] System capabilities use "the system/platform/model"
- [ ] Possessive adjectives are correct ("my" vs "the")
- [ ] No instances of "we/our/ours" in author-voice contexts

### Exceptions

Document any intentional exceptions to these guidelines:

```markdown
<!-- STYLE_EXCEPTION: Quote from user feedback -->
**User feedback**: "We love the system's performance"
```

## Questions and Clarifications

### When to Use "I" vs "the system"?

**Ask yourself**: "Am I describing something I did/decided, or something the system does?"

- **I did/decided** → Use "I"
  - "I chose this architecture because..."
  - "I implemented this feature to..."
  
- **System does** → Use "the system"
  - "The system processes images at..."
  - "The model achieves accuracy of..."

### What About Future Work?

Use "I" for your plans and intentions:

✅ **Correct**:
- "I plan to add support for..."
- "I'm working on implementing..."
- "I will benchmark against..."

### What About Contributions?

Use "I" when welcoming contributions:

✅ **Correct**:
- "I welcome contributions from..."
- "I'm looking for collaborators to..."
- "I appreciate feedback on..."

## Summary

**Key Takeaways**:

1. Use **"I"** for author actions, decisions, and plans
2. Use **"the system/platform/model"** for technical capabilities
3. Use **"my"** for author's work, **"the"** for system components
4. Maintain technical accuracy by not attributing system capabilities to the author
5. Keep documentation clear, professional, and accessible

**When in doubt**: Ask "Who/what is performing this action?" and choose the voice accordingly.

---

For questions about these guidelines or specific cases, please open an issue or refer to [CONTRIBUTING.md](CONTRIBUTING.md).

**Last Updated**: 2026
**Version**: 1.0
