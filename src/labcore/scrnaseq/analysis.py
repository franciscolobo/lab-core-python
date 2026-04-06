# In src/labcore/scrnaseq/analysis.py
import pandas as pd
import numpy as np
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
    sources: list[str] = ['GO:BP'],
    background_genes: list[str] | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Performs Over-Representation Analysis (ORA) using gprofiler.
    """
    if species.lower() not in GPROFILER_ORGANISMS:
        raise ValueError(f"Species '{species}' not supported. Use one of: {list(GPROFILER_ORGANISMS.keys())}")

    organism_id = GPROFILER_ORGANISMS[species.lower()]
    cleaned_gene_list = sorted(list(set(str(g).upper() for g in gene_list)))

    print(f"Running ORA with gprofiler for {species} ({organism_id}) on {len(cleaned_gene_list)} genes...")
    print(f"Data sources: {sources}")

    gp_instance = GProfiler(return_dataframe=True)

    background = background_genes if background_genes is not None else None
    if background:
        print(f"Using a custom background of {len(background)} genes.")

    # --- THIS IS THE FIX ---
    # 1. Add `no_evidences=False` to ensure the gene list column is returned.
    # 2. This is the only change needed in this section.
    results_df = gp_instance.profile(
        organism=organism_id,
        query=cleaned_gene_list,
        sources=sources,
        background=background,
        no_evidences=False, # <-- CRITICAL: This requests the gene list evidence.
        **kwargs
    )

    if results_df.empty:
        print("  -> No enrichment results found.")
        return pd.DataFrame()

    # Now that we have the 'intersections' column, we can harmonize.
    results_df = results_df.rename(columns={
        'native': 'Term',
        'p_value': 'Adjusted P-value',
        'intersection_size': 'Overlap_Size',
        'source': 'Gene_set',
        'name': 'Description'
    })

    results_df['Combined Score'] = results_df['Overlap_Size'] / results_df['term_size'] * -np.log10(results_df['Adjusted P-value'])

    # Use the correct column name 'intersections' (plural).
    results_df['Genes'] = results_df['intersections'].apply(lambda x: ';'.join(x))

    print(f"  -> Found {len(results_df)} enriched terms.")
    return results_df.sort_values(by='Adjusted P-value')

