import pandas as pd
import os

def load_chromosome_genes_from_biomart(
    filepath: str,
    chromosomes: list[str]
) -> dict[str, list[str]]:
    """
    Parses a downloaded BioMart file to get protein-coding genes for specific chromosomes.

    Args:
        filepath: The full path to the tab-separated file downloaded from BioMart.
                  The file must contain the columns: "ensembl_gene_id", 
                  "external_gene_name", "chromosome_name", "gene_biotype".
        chromosomes: A list of chromosome names to extract genes for (e.g., ['Z', 'W']).

    Returns:
        A dictionary where keys are the requested chromosome names and values are the
        lists of protein-coding gene symbols found on that chromosome.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BioMart file not found at: {filepath}")

    print(f"Loading and parsing BioMart file: {filepath}")
    
    # Read the tab-separated file
    df = pd.read_csv(filepath, sep='\t')
    
    # --- 1. Filter for Protein-Coding Genes ---
    protein_coding_df = df[df['gene_biotype'] == 'protein_coding'].copy()
    
    # --- 2. Filter for the specified chromosomes ---
    # The chromosome names in the file might be strings or numbers, so we convert to string for safety
    protein_coding_df['chromosome_name'] = protein_coding_df['chromosome_name'].astype(str)
    chromosomes_str = [str(c) for c in chromosomes]
    sex_chrom_df = protein_coding_df[protein_coding_df['chromosome_name'].isin(chromosomes_str)]
    
    # --- 3. Group by chromosome and create the dictionary ---
    # Drop any genes that don't have an external_gene_name
    sex_chrom_df = sex_chrom_df.dropna(subset=['external_gene_name'])
    
    # Group the dataframe by chromosome and collect the gene names into lists
    gene_lists = (
        sex_chrom_df.groupby('chromosome_name')['external_gene_name']
        .apply(list)
        .to_dict()
    )

    # Ensure all requested chromosomes are in the output dict, even if empty
    for chrom in chromosomes_str:
        if chrom not in gene_lists:
            gene_lists[chrom] = []
        print(f"  -> Found {len(gene_lists[chrom])} protein-coding genes for chromosome '{chrom}'.")
        
    return gene_lists

