from gprofiler import GProfiler
import pandas as pd

ORGANISM_ALIASES = {
    "human":   "hsapiens",
    "mouse":   "mmusculus",
    "chicken": "ggallus",
    # Add other species as needed
}


def get_orthologs(
    gene_list: list[str],
    target_species: str,
    source_species: str = "human",
    include_paralogs: bool = False,
) -> dict[str, str]:
    """Find one-to-one orthologs (and optionally paralogs) for a gene list.

    Args:
        gene_list: Source gene symbols to query.
        target_species: Target species key (must be in ORGANISM_ALIASES).
        source_species: Source species key. Defaults to "human".
        include_paralogs: When ``True``, the one-to-one filter is relaxed
            and all hits are returned, including one-to-many and many-to-one
            mappings (paralogs). When multiple target hits exist for the same
            source gene, only the first hit is kept. Defaults to ``False``.

    Returns:
        A dict mapping source gene symbol → target gene symbol. When
        ``include_paralogs=False`` only unambiguous one-to-one relationships
        are returned. When ``True``, all hits are included (first hit wins
        for duplicated source genes).

    Raises:
        ValueError: If either species key is not in ORGANISM_ALIASES.
        KeyError: If g:Profiler response is missing expected columns.
    """
    if target_species not in ORGANISM_ALIASES:
        raise ValueError(
            f"Target species '{target_species}' not recognized. "
            f"Available aliases: {list(ORGANISM_ALIASES.keys())}"
        )
    if source_species not in ORGANISM_ALIASES:
        raise ValueError(
            f"Source species '{source_species}' not recognized. "
            f"Available aliases: {list(ORGANISM_ALIASES.keys())}"
        )

    source_organism = ORGANISM_ALIASES[source_species]
    target_organism = ORGANISM_ALIASES[target_species]

    print(
        f"Finding orthologs in '{target_species}' for "
        f"{len(gene_list)} genes from '{source_species}'..."
    )

    gp = GProfiler(return_dataframe=True)
    ortholog_df = gp.orth(
        organism=source_organism,
        query=gene_list,
        target=target_organism,
    )

    if ortholog_df.empty:
        print("Warning: g:Profiler returned no orthologs.")
        return {}

    required_cols = {"incoming", "name"}
    if not required_cols.issubset(ortholog_df.columns):
        print("g:Profiler response columns:", ortholog_df.columns.tolist())
        print(ortholog_df.head())
        raise KeyError(
            f"g:Profiler did not return expected columns: "
            f"{required_cols - set(ortholog_df.columns)}"
        )

    # Drop rows where the target ortholog is missing (no hit)
    df = ortholog_df.dropna(subset=["name"]).copy()

    if include_paralogs:
        # Relax the one-to-one constraint: keep all hits, first hit wins
        # when a source gene maps to multiple targets.
        df = df.drop_duplicates(subset=["incoming"])
        ortholog_map = df.set_index("incoming")["name"].to_dict()
        print(f"Found {len(ortholog_map)} hits (paralogs included).")
    else:
        # One-to-one filtering:
        #   - drop source genes that map to more than one target (one-to-many)
        #   - drop target genes that are hit by more than one source (many-to-one)
        # Count occurrences on each side BEFORE deduplication.
        source_counts = df["incoming"].value_counts()
        target_counts = df["name"].value_counts()

        one_to_one_sources = source_counts[source_counts == 1].index
        one_to_one_targets = target_counts[target_counts == 1].index

        df = df[
            df["incoming"].isin(one_to_one_sources) &
            df["name"].isin(one_to_one_targets)
        ]
        ortholog_map = df.set_index("incoming")["name"].to_dict()
        print(f"Found {len(ortholog_map)} one-to-one orthologs.")

    return ortholog_map
