"""
Causal graph utilities for pathology domain knowledge encoding.

Encodes domain knowledge as a DAG: morphology → biomarkers → survival.
Supports d-separation testing and backdoor criterion checking.
"""

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.warning("networkx not installed. pip install networkx>=3.0")


class CausalDAG:
    """
    Directed Acyclic Graph for encoding causal assumptions.

    Example — pathology causal structure:
        morphology_features → tumor_grade
        morphology_features → treatment_response
        patient_age → treatment_assignment
        patient_age → survival
        tumor_grade → survival
        treatment_assignment → treatment_response
        treatment_response → survival
    """

    def __init__(self):
        if not HAS_NX:
            raise ImportError("networkx required. pip install networkx>=3.0")
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_node(self, name: str, node_type: str = "observed") -> None:
        """
        Args:
            name: Variable name
            node_type: "observed", "latent", "treatment", "outcome"
        """
        self.graph.add_node(name, node_type=node_type)

    def add_edge(self, cause: str, effect: str) -> None:
        if cause not in self.graph:
            self.add_node(cause)
        if effect not in self.graph:
            self.add_node(effect)
        self.graph.add_edge(cause, effect)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(cause, effect)
            raise ValueError(f"Adding edge {cause}→{effect} creates a cycle")

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        for cause, effect in edges:
            self.add_edge(cause, effect)

    def parents(self, node: str) -> Set[str]:
        return set(self.graph.predecessors(node))

    def children(self, node: str) -> Set[str]:
        return set(self.graph.successors(node))

    def ancestors(self, node: str) -> Set[str]:
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> Set[str]:
        return nx.descendants(self.graph, node)

    def _moral_graph(self, nodes: Set[str]) -> "nx.Graph":
        """Moral graph of ancestors of nodes."""
        an = set()
        for n in nodes:
            an |= self.ancestors(n)
        an |= nodes
        subgraph = self.graph.subgraph(an).copy()
        # Marry parents (add undirected edges between co-parents)
        moral = subgraph.to_undirected()
        for node in an:
            pars = list(subgraph.predecessors(node))
            for i in range(len(pars)):
                for j in range(i + 1, len(pars)):
                    moral.add_edge(pars[i], pars[j])
        return moral

    def d_separated(
        self,
        X: Set[str],
        Y: Set[str],
        Z: Set[str],
    ) -> bool:
        """
        Test d-separation: X ⊥ Y | Z via Bayes Ball / moralization algorithm.

        Returns True if X and Y are d-separated given Z (i.e., conditionally
        independent given Z under every distribution compatible with the DAG).
        """
        all_nodes = X | Y | Z
        ancestors_of_all = set()
        for n in all_nodes:
            ancestors_of_all |= self.ancestors(n)
        ancestors_of_all |= all_nodes

        # Build moral graph on ancestors
        subgraph = self.graph.subgraph(ancestors_of_all).copy()
        moral = subgraph.to_undirected()
        for node in ancestors_of_all:
            pars = list(subgraph.predecessors(node))
            for i in range(len(pars)):
                for j in range(i + 1, len(pars)):
                    moral.add_edge(pars[i], pars[j])

        # Remove Z nodes (conditioning)
        moral.remove_nodes_from(Z)

        # Check if X and Y are in different connected components
        for x in X:
            if x not in moral:
                continue
            for y in Y:
                if y not in moral:
                    continue
                if nx.has_path(moral, x, y):
                    return False
        return True

    def is_valid(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def summary(self) -> Dict:
        return {
            "nodes": list(self.graph.nodes),
            "edges": list(self.graph.edges),
            "is_dag": self.is_valid(),
        }


def check_backdoor_criterion(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    adjustment_set: Set[str],
) -> bool:
    """
    Check if adjustment_set satisfies the backdoor criterion for
    identifying the causal effect of treatment on outcome.

    Backdoor criterion (Pearl 1993):
    1. No node in Z is a descendant of treatment.
    2. Z blocks every backdoor path from treatment to outcome
       (paths that start with an arrow INTO treatment).

    Returns True if the criterion is satisfied.
    """
    # Condition 1: no Z node is a descendant of treatment
    desc_treatment = dag.descendants(treatment)
    if adjustment_set & desc_treatment:
        logger.warning(
            f"Backdoor criterion violated: {adjustment_set & desc_treatment} "
            f"are descendants of {treatment}"
        )
        return False

    # Condition 2: Z d-separates treatment from outcome in the graph
    # where all edges OUT of treatment are removed (backdoor graph)
    if not HAS_NX:
        raise ImportError("networkx required")
    backdoor_graph_edges = [
        (u, v) for u, v in dag.graph.edges if u != treatment
    ]
    backdoor_dag = CausalDAG()
    for node in dag.graph.nodes:
        backdoor_dag.add_node(node, **dag.graph.nodes[node])
    for u, v in backdoor_graph_edges:
        try:
            backdoor_dag.add_edge(u, v)
        except ValueError:
            pass

    separated = backdoor_dag.d_separated({treatment}, {outcome}, adjustment_set)
    if separated:
        logger.info(
            f"Backdoor criterion satisfied: {adjustment_set} adjusts for "
            f"{treatment} → {outcome}"
        )
    else:
        logger.warning(
            f"Backdoor criterion NOT satisfied for adjustment set {adjustment_set}"
        )
    return separated


def build_pathology_dag() -> CausalDAG:
    """
    Construct default causal DAG for computational pathology.

    Encodes domain knowledge about relationships between:
    - Morphological features (WSI-derived)
    - Molecular biomarkers
    - Patient demographics
    - Treatment assignment (observational confounding)
    - Clinical outcomes
    """
    dag = CausalDAG()
    dag.add_edges([
        # Demographics → confounders
        ("patient_age", "treatment_assignment"),
        ("patient_age", "survival"),
        ("tumor_stage", "treatment_assignment"),
        ("tumor_stage", "survival"),
        # Morphology → downstream
        ("morphology_features", "tumor_grade"),
        ("morphology_features", "treatment_response"),
        ("morphology_features", "biomarker_expression"),
        # Molecular → outcomes
        ("biomarker_expression", "treatment_response"),
        ("biomarker_expression", "survival"),
        # Tumor biology
        ("tumor_grade", "survival"),
        ("tumor_grade", "treatment_assignment"),
        # Treatment pathway
        ("treatment_assignment", "treatment_response"),
        ("treatment_response", "survival"),
    ])
    return dag
