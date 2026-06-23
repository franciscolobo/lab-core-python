from __future__ import annotations

import numpy as np
import pandas as pd
from gprofiler import GProfiler

# Map common names to gprofiler's internal organism IDs
GPROFILER_ORGANISMS = {
    "human": "hsapiens",
    "mouse": "mmusculus",
    "chicken": "ggallus",
}


def run_ora(
    gene_list: list[str],
    species: str,
    sources: list[str] = ["GO:BP"],
    background_genes: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run Over-Representation Analysis (ORA) using g:Profiler.

    Args:
        gene_list: Genes to test for enrichment.
        species: Common species name. Supported values:
            ``"human"``, ``"mouse"``, ``"chicken"``.
        sources: g:Profiler data sources to query.
            Defaults to ``["GO:BP"]``.
        background_genes: Custom background gene set. When ``None``
            g:Profiler uses its default background. Defaults to
            ``None``.
        **kwargs: Forwarded to ``GProfiler.profile()``.

    Returns:
        DataFrame of enriched terms sorted by adjusted p-value, with
        columns ``Term``, ``Description``, ``Gene_set``,
        ``Adjusted P-value``, ``Overlap_Size``, ``Combined Score``,
        and ``Genes``. Returns an empty DataFrame when no terms are
        enriched.

    Raises:
        ValueError: If ``species`` is not in the supported organisms
            dict.
    """
    if species.lower() not in GPROFILER_ORGANISMS:
        raise ValueError(
            f"Species '{species}' not supported. "
            f"Choose from: {list(GPROFILER_ORGANISMS.keys())}"
        )

    organism_id = GPROFILER_ORGANISMS[species.lower()]
    cleaned_genes = sorted(set(str(g).upper() for g in gene_list))

    print(
        f"Running ORA with g:Profiler for {species} ({organism_id}) "
        f"on {len(cleaned_genes)} genes — sources: {sources}"
    )

    gp = GProfiler(return_dataframe=True)

    if background_genes is not None:
        print(f"Using a custom background of {len(background_genes)} genes.")

    results_df = gp.profile(
        organism=organism_id,
        query=cleaned_genes,
        sources=sources,
        background=background_genes,
        no_evidences=False,
        **kwargs,
    )

    if results_df.empty:
        print("  -> No enrichment results found.")
        return pd.DataFrame()

    results_df = results_df.rename(columns={
        "native":            "Term",
        "p_value":           "Adjusted P-value",
        "intersection_size": "Overlap_Size",
        "source":            "Gene_set",
        "name":              "Description",
    })

    results_df["Combined Score"] = (
        results_df["Overlap_Size"]
        / results_df["term_size"]
        * -np.log10(results_df["Adjusted P-value"])
    )
    results_df["Genes"] = results_df["intersections"].apply(";".join)

    print(f"  -> Found {len(results_df)} enriched terms.")
    return results_df.sort_values("Adjusted P-value")
