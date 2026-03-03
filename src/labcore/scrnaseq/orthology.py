from gprofiler import GProfiler
import pandas as pd

# (ORGANISM_ALIASES dictionary is unchanged)

def get_orthologs(
    gene_list: list[str],
    target_species: str,
    source_species: str = "human",
) -> dict[str, str]:
    """
    Finds one-to-one orthologs for a given gene list in a target species.
    ... (docstring is unchanged) ...
    """
    if target_species not in ORGANISM_ALIASES:
        raise ValueError(f"Target species '{target_species}' not recognized. "
                         f"Available aliases: {list(ORGANISM_ALIASES.keys())}")
    if source_species not in ORGANISM_ALIASES:
        raise ValueError(f"Source species '{source_species}' not recognized.")

    source_organism = ORGANISM_ALIASES[source_species]
    target_organism = ORGANISM_ALIASES[target_species]
    
    print(f"Finding orthologs in '{target_species}' for {len(gene_list)} genes from '{source_species}'...")

    gp = GProfiler(return_dataframe=True)
    
    ortholog_df = gp.orth(
        organism=source_organism,
        query=gene_list,
        target=target_organism
    )

    if ortholog_df.empty:
        print("Warning: g:Profiler returned no orthologs.")
        return {}

    # --- THIS IS THE FIX ---
    # The column for the ortholog's name is 'name', not 'ortholog_name'.
    # We will also add a check to make sure the expected columns exist.
    
    required_cols = {'incoming_gene_name', 'name', 'n_incoming'}
    if not required_cols.issubset(ortholog_df.columns):
        print("g:Profiler response is missing expected columns. Columns found:", ortholog_df.columns.tolist())
        print("DataFrame head:\n", ortholog_df.head())
        raise KeyError(f"g:Profiler did not return the expected columns: {required_cols - set(ortholog_df.columns)}")

    # Filter for best one-to-one orthologs and create the mapping dictionary
    ortholog_map = (
        ortholog_df
        .dropna(subset=['name'])  # <-- Corrected column name
        .loc[ortholog_df['n_incoming'] == 1]
        .drop_duplicates(subset=['incoming_gene_name'])
        .set_index('incoming_gene_name')['name']  # <-- Corrected column name
        .to_dict()
    )
    
    print(f"Found {len(ortholog_map)} one-to-one orthologs.")
    
    return ortholog_map
