from gprofiler import GProfiler
import pandas as pd

# Standard organism aliases used by gProfiler
ORGANISM_ALIASES = {
    "human": "hsapiens",
    "mouse": "mmusculus",
    "chicken": "ggallus",
    # Add other species as needed
}

def get_orthologs(
    gene_list: list[str],
    target_species: str,
    source_species: str = "human",
) -> dict[str, str]:
    """
    Finds one-to-one orthologs for a given gene list in a target species.

    This function uses the g:Profiler API to map gene symbols from a source
    species (default: human) to a target species.

    Args:
        gene_list: A list of gene symbols from the source species.
        target_species: The common name of the target species (e.g., 'mouse', 'chicken').
        source_species: The common name of the source species (default: 'human').

    Returns:
        A dictionary where keys are the original gene symbols and values are the
        corresponding one-to-one ortholog symbols in the target species. Genes
        without a 1-to-1 ortholog are excluded.
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
    
    # Perform the orthology query
    ortholog_df = gp.orth(
        organism=source_organism,
        query=gene_list,
        target=target_organism
    )

    if ortholog_df.empty:
        print("Warning: g:Profiler returned no orthologs.")
        return {}

    # Filter for best one-to-one orthologs and create the mapping dictionary
    # 'ortholog_ensg' is the Ensembl ID, 'ortholog_name' is the symbol
    ortholog_map = (
        ortholog_df
        .dropna(subset=['ortholog_name'])
        .loc[ortholog_df['n_incoming'] == 1] # Ensure it's a 1-to-1 mapping
        .drop_duplicates(subset=['incoming_gene_name']) # Take the first if there are multiple hits
        .set_index('incoming_gene_name')['ortholog_name']
        .to_dict()
    )
    
    print(f"Found {len(ortholog_map)} one-to-one orthologs.")
    
    return ortholog_map
