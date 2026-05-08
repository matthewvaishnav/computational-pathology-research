"""
Scientific hypothesis generation via LLM API.

Takes latent factor loadings, subtype survival curves, and TME composition
as context and generates structured, testable biological hypotheses.

Outputs hypotheses in a structured schema with:
  - hypothesis_text: plain-language statement
  - mechanism: proposed biological mechanism
  - testable_prediction: what experiment would confirm/refute
  - affected_genes: list of relevant genes (for validation against data)
  - confidence: HIGH/MEDIUM/LOW
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeneratedHypothesis:
    hypothesis_text: str
    mechanism: str
    testable_prediction: str
    affected_genes: List[str]
    affected_pathways: List[str]
    confidence: str  # HIGH / MEDIUM / LOW
    subtype_context: Optional[str] = None
    factor_context: Optional[str] = None
    raw_response: Optional[str] = None


def _summarise_factor_loadings(
    loadings: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
) -> str:
    """Return top-k genes by absolute loading for a single factor."""
    if len(feature_names) != loadings.shape[0]:
        return "(feature names mismatch)"
    top_idx = np.argsort(np.abs(loadings))[::-1][:top_k]
    items = [(feature_names[i], float(loadings[i])) for i in top_idx]
    return ", ".join(f"{g}({v:+.3f})" for g, v in items)


class HypothesisGenerator:
    """
    Generates scientific hypotheses from computational findings using any LLM API.

    Args:
        llm_fn: Callable that takes (system_prompt, user_prompt) and returns text response
        cancer_type: tissue/cancer type context (e.g. "lung adenocarcinoma")
        max_hypotheses_per_call: number of hypotheses to request per API call
    """

    SYSTEM_PROMPT = """You are an expert computational oncologist with deep knowledge of
tumour biology, immunology, and molecular pathology. You analyse computational pathology
findings and generate novel, testable scientific hypotheses.

For each hypothesis, respond ONLY with a valid JSON array. Each element must have exactly
these keys: hypothesis_text, mechanism, testable_prediction, affected_genes (list),
affected_pathways (list), confidence (HIGH/MEDIUM/LOW).

Be specific. Cite known biology. Avoid generic statements. Focus on clinically actionable
or mechanistically novel hypotheses."""

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        cancer_type: str = "cancer",
        max_hypotheses_per_call: int = 3,
    ):
        self.llm_fn = llm_fn
        self.cancer_type = cancer_type
        self.max_per_call = max_hypotheses_per_call

    def _call_api(self, user_prompt: str) -> str:
        """Call LLM API via provided function."""
        return self.llm_fn(self.SYSTEM_PROMPT, user_prompt)

    def _parse_response(self, text: str, context: dict) -> List[GeneratedHypothesis]:
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            # Handle markdown code blocks robustly
            parts = text.split("```")
            if len(parts) >= 3:
                # Standard case: ```json\n...\n```
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
            elif len(parts) == 2:
                # Edge case: only opening fence
                text = parts[1]
        text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse hypothesis JSON: %s\nRaw: %s", e, text[:500])
            return []

        hypotheses = []
        for item in items[: self.max_per_call]:
            hypotheses.append(
                GeneratedHypothesis(
                    hypothesis_text=item.get("hypothesis_text", ""),
                    mechanism=item.get("mechanism", ""),
                    testable_prediction=item.get("testable_prediction", ""),
                    affected_genes=item.get("affected_genes", []),
                    affected_pathways=item.get("affected_pathways", []),
                    confidence=item.get("confidence", "MEDIUM"),
                    subtype_context=context.get("subtype"),
                    factor_context=context.get("factor"),
                    raw_response=text,
                )
            )
        return hypotheses

    def from_subtypes(
        self,
        subtype_labels: np.ndarray,
        survival_pvalue: float,
        tme_compositions: Optional[List[Dict[str, Any]]] = None,
        additional_context: str = "",
    ) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses from subtype analysis.

        Args:
            subtype_labels: (N,) array of subtype assignments
            survival_pvalue: log-rank p-value between subtypes
            tme_compositions: list of per-subtype TME dicts
            additional_context: any additional findings to include
        """
        n_subtypes = len(np.unique(subtype_labels))
        subtype_sizes = {
            int(s): int((subtype_labels == s).sum()) for s in np.unique(subtype_labels)
        }

        tme_parts = []
        if tme_compositions:
            for i, tme in enumerate(tme_compositions):
                tme_parts.append(
                    f"\n  Subtype {i}: TIL density={tme.get('til_density', 'N/A'):.2%}, "
                    f"phenotype={tme.get('immune_phenotype', 'N/A')}"
                )
        tme_summary = "".join(tme_parts)

        prompt = f"""Cancer type: {self.cancer_type}

Computational findings from unsupervised subtype discovery:
- Number of subtypes: {n_subtypes}
- Subtype sizes: {subtype_sizes}
- Kaplan-Meier log-rank p-value: {survival_pvalue:.4f}
- TME composition by subtype:{tme_summary}
{additional_context}

Generate {self.max_per_call} mechanistic hypotheses explaining:
1. Why these subtypes have different survival outcomes
2. The role of the tumour immune microenvironment in each subtype
3. Potential therapeutic targets specific to each subtype

Return a JSON array."""

        try:
            raw = self._call_api(prompt)
            return self._parse_response(raw, {"subtype": f"{n_subtypes}-subtype discovery"})
        except Exception as e:
            logger.error("Hypothesis generation failed: %s", e)
            return []

    def from_factors(
        self,
        factor_idx: int,
        loadings: np.ndarray,
        feature_names: List[str],
        variance_explained: float,
        modality: str = "RNA",
        additional_context: str = "",
    ) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses from a MOFA latent factor.

        Args:
            factor_idx: which factor number
            loadings: (P,) loading vector for this factor
            feature_names: gene/protein names corresponding to loadings
            variance_explained: R² of this factor
            modality: data modality name
            additional_context: e.g. correlation with survival
        """
        top_pos = _summarise_factor_loadings(loadings, feature_names, top_k=15)
        top_neg = _summarise_factor_loadings(-loadings, feature_names, top_k=15)

        prompt = f"""Cancer type: {self.cancer_type}
Multi-omics latent factor analysis (MOFA) results:

Factor {factor_idx + 1} ({modality}, variance explained: {variance_explained:.1%}):
  Top positively loaded features: {top_pos}
  Top negatively loaded features: {top_neg}
{additional_context}

Generate {self.max_per_call} hypotheses about the biological process captured by this factor.
Focus on: pathway activation/repression, cell state transitions, clinical relevance.

Return a JSON array."""

        try:
            raw = self._call_api(prompt)
            return self._parse_response(raw, {"factor": f"Factor {factor_idx + 1}"})
        except Exception as e:
            logger.error("Factor hypothesis generation failed: %s", e)
            return []

    def from_spatial_pattern(
        self,
        gene: str,
        morans_i: float,
        pearson_r: float,
        description: str = "",
    ) -> List[GeneratedHypothesis]:
        """Generate hypotheses from spatially variable gene patterns."""
        prompt = f"""Cancer type: {self.cancer_type}
Spatial transcriptomics finding:
  Gene: {gene}
  Moran's I (spatial autocorrelation): {morans_i:.3f}
  H&E-to-expression Pearson r: {pearson_r:.3f}
  {description}

Generate {self.max_per_call} hypotheses explaining the spatial expression pattern
of {gene} and its relationship to tumour morphology.

Return a JSON array."""

        try:
            raw = self._call_api(prompt)
            return self._parse_response(raw, {"factor": f"spatial_{gene}"})
        except Exception as e:
            logger.error("Spatial hypothesis generation failed: %s", e)
            return []


# Helper functions for common LLM providers

def create_openai_llm(api_key: Optional[str] = None, model: str = "gpt-4") -> Callable[[str, str], str]:
    """Create OpenAI LLM function."""
    import openai
    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def llm_fn(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=2048
        )
        return response.choices[0].message.content
    return llm_fn


def create_anthropic_llm(api_key: Optional[str] = None, model: str = "claude-opus-4-7") -> Callable[[str, str], str]:
    """Create Anthropic Claude LLM function."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    
    def llm_fn(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=60.0
        )
        return msg.content[0].text
    return llm_fn


def create_ollama_llm(model: str = "llama3", base_url: str = "http://localhost:11434") -> Callable[[str, str], str]:
    """Create Ollama local LLM function."""
    import requests
    
    def llm_fn(system: str, user: str) -> str:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "stream": False
            }
        )
        return response.json()["message"]["content"]
    return llm_fn
