"""
GO-term propagation for mDeepFRI predictions.

Propagates GO annotations up the ontology DAG using the true-path rule
(is_a and part_of relations). Automatically downloads the go-basic.obo
file if not present.

Dependencies: obonet, networkx
"""

from __future__ import annotations

import csv
import logging
import urllib.request
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set, Tuple

import networkx as nx
import obonet

logger = logging.getLogger(__name__)

GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"

# Root GO term IDs (BP, MF, CC)
ROOT_IDS: frozenset = frozenset({"GO:0008150", "GO:0003674", "GO:0005575"})


def download_obo(obo_path: Path) -> Path:
    """Download go-basic.obo from the Gene Ontology if not already present.

    Parameters
    ----------
    obo_path : Path
        Destination path for the OBO file.

    Returns
    -------
    Path
        Path to the downloaded OBO file.
    """
    if obo_path.exists():
        logger.info("OBO file already exists: %s", obo_path)
        return obo_path

    obo_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading go-basic.obo from %s", GO_OBO_URL)
    req = urllib.request.Request(GO_OBO_URL, headers={"User-Agent": "mDeepFRI"})
    with urllib.request.urlopen(req) as response, open(obo_path, "wb") as out:
        out.write(response.read())
    logger.info("Downloaded go-basic.obo to %s", obo_path)
    return obo_path


def load_obo(obo_path: Path) -> nx.MultiDiGraph:
    """Load GO OBO file into a networkx MultiDiGraph.

    obonet convention: edges go from child -> parent (is_a, part_of).
    """
    logger.info("Loading OBO from %s", obo_path)
    graph = obonet.read_obo(str(obo_path))
    return graph


def get_ancestors_filtered(
    graph: nx.MultiDiGraph,
    term: str,
    relations: Tuple[str, ...] = ("is_a", "part_of"),
) -> Set[str]:
    """All ancestors of a term traversing only the given relation types.

    In obonet's MultiDiGraph, edges go child -> parent and edge keys are the
    relationship type (e.g. 'is_a', 'part_of').
    """
    relations_set = set(relations)
    visited: Set[str] = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node not in graph:
            continue
        for parent, edge_key_dict in graph[node].items():
            for rel_type in edge_key_dict:
                if rel_type in relations_set and parent not in visited:
                    stack.append(parent)
                    break
    visited.discard(term)
    return visited


def propagate_results(
    results_path: Path,
    output_path: Path,
    obo_path: Path,
    relations: Tuple[str, ...] = ("is_a", "part_of"),
    exclude_roots: bool = True,
) -> Path:
    """Propagate GO terms from a results tsv file.

    Reads the results file, propagates each protein's GO terms up the
    ontology DAG (true-path rule), and writes an expanded results file
    with a ``propagated`` column.

    Propagated ancestor terms inherit the maximum score among their
    descendant terms that were directly predicted.

    Parameters
    ----------
    results_path : Path
        Path to the mDeepFRI ``results.tsv`` file.
    output_path : Path
        Path to write the propagated results TSV.
    obo_path : Path
        Path to the ``go-basic.obo`` file.
    relations : tuple of str
        Edge types to traverse; defaults to ``('is_a', 'part_of')``.
    exclude_roots : bool
        If True, drop the three ontology root terms from output.

    Returns
    -------
    Path
        Path to the propagated results file.
    """
    graph = load_obo(obo_path)

    # Cache ancestor lookups
    @lru_cache(maxsize=None)
    def _ancestors(term_id: str) -> frozenset:
        node = graph.nodes.get(term_id, {})
        if node.get("is_obsolete", False):
            return frozenset()
        ancs = get_ancestors_filtered(graph, term_id, relations)
        if exclude_roots:
            ancs -= ROOT_IDS
        return frozenset(ancs)

    # Read input results
    # Expected header:
    # protein, network_type, prediction_mode, go_term, score,
    # go_name, aligned, target_id, db_name, query_identity,
    # query_coverage, target_coverage
    rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            rows.append(row)

    # Build a mapping: (protein, prediction_mode) -> list of (go_term, score, row_data)
    # We group by protein AND prediction_mode because different modes
    # (bp, mf, cc) have different ontology namespaces.
    protein_mode_terms: Dict[Tuple[str, str], Dict[str, Tuple[float, list]]] = defaultdict(dict)

    for row in rows:
        protein = row[0]
        prediction_mode = row[2]
        go_term = row[3]
        try:
            score = float(row[4])
        except (ValueError, IndexError):
            score = 0.0

        key = (protein, prediction_mode)
        # Keep the row with the highest score for each (protein, mode, term)
        if go_term not in protein_mode_terms[key] or score > protein_mode_terms[key][go_term][0]:
            protein_mode_terms[key][go_term] = (score, row)

    # Propagate
    output_rows = []
    for (protein, prediction_mode), term_data in protein_mode_terms.items():
        # Track max score per term and if it's propagated
        term_scores: Dict[str, float] = {}
        term_propagated: Dict[str, bool] = {}
        term_row_template: Dict[str, list] = {}

        # First pass: add original terms
        for go_term, (score, row) in term_data.items():
            if not go_term.startswith("GO:"):
                # Non-GO terms (EC numbers) are kept as-is
                term_scores[go_term] = score
                term_propagated[go_term] = False
                term_row_template[go_term] = row
                continue

            term_scores[go_term] = score
            term_propagated[go_term] = False
            term_row_template[go_term] = row

            # Propagate to ancestors.
            # is_a and part_of edges in GO do not cross namespaces,
            # so ancestors will always be in the same namespace.
            ancestors = _ancestors(go_term)
            for anc in ancestors:
                if anc not in term_scores or score > term_scores[anc]:
                    term_scores[anc] = score
                if anc not in term_propagated:
                    term_propagated[anc] = True

        # Build output rows
        for term, score in term_scores.items():
            is_propagated = term_propagated.get(term, True)

            if term in term_row_template:
                # Original term - use its original row data
                row = list(term_row_template[term])
            else:
                # Propagated term - construct row from graph metadata
                node_data = graph.nodes.get(term, {})
                go_name = node_data.get("name", "")
                # Use a template row from the same protein/mode for alignment info
                template = next(iter(term_data.values()))[1]
                row = [
                    protein,
                    template[1],    # network_type
                    template[2],    # prediction_mode
                    term,
                    f"{score:.4f}",
                    go_name,
                    template[6] if len(template) > 6 else "",    # aligned
                    template[7] if len(template) > 7 else "",    # target_id
                    template[8] if len(template) > 8 else "",    # db_name
                    template[9] if len(template) > 9 else "",    # query_identity
                    template[10] if len(template) > 10 else "",  # query_coverage
                    template[11] if len(template) > 11 else "",  # target_coverage
                ]

            # Ensure score is formatted
            try:
                row[4] = f"{float(row[4]):.4f}"
            except (ValueError, IndexError):
                pass

            row.append("True" if is_propagated else "False")
            output_rows.append(row)

    # Sort: by protein, then propagated (original first), then score desc
    def sort_key(row):
        try:
            score = -float(row[4])
        except (ValueError, IndexError):
            score = 0
        propagated = row[-1] == "True"
        return (row[0], propagated, score, row[3])

    output_rows.sort(key=sort_key)

    # Write output
    output_header = header + ["propagated"]
    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(output_header)
        for row in output_rows:
            writer.writerow(row)

    n_original = sum(1 for r in output_rows if r[-1] == "False")
    n_propagated = sum(1 for r in output_rows if r[-1] == "True")
    logger.info(
        "GO propagation complete: %d original + %d propagated = %d total rows",
        n_original, n_propagated, len(output_rows),
    )

    return output_path
