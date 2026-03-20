# In src/labcore/scrnaseq/annotation.py

from pyensembl import EnsemblRelease
import pandas as pd
import os

# Define standard Ensembl releases for reproducibility
ENSEMBL_RELEASES = {
    "human": 102,
    "mouse": 102,  # GRCm38
    "chicken": 109, # bGalGal1.mat.broiler.GRCg7b
}

def get_chromosome_genes(
    species: str,
    chromosomes: list[str],
    output_dir: str = "gene_lists",
) -> dict[str, list[str]]:
    """
    Fetches and saves lists of protein-coding genes for specified chromosomes.
    ... (docstring is unchanged) ...
    """
    if species not in ENSEMBL_RELEASES:
        raise ValueError(f"Species '{species}' not configured. Please add it to ENSEMBL_RELEASES.")
    
    os.makedirs(output_dir, exist_ok=True)
    release = ENSEMBL_RELEASES[species]
    
    all_genes = {}

    print(f"Loading Ensembl release {release} for {species}...")
    try:
        ensembl_data = EnsemblRelease(release, species=species)
    except Exception as e:
        print(f"Could not load Ensembl data for {species} release {release}.")
        print("You may need to run `pyensembl install --release {release} --species {species}`")
        raise e

    for chrom in chromosomes:
        file_path = os.path.join(output_dir, f"{species}_release{release}_chrom_{chrom}_genes.tsv")
        
        if os.path.exists(file_path):
            print(f"Loading genes for chromosome '{chrom}' from cached file: {file_path}")
            df = pd.read_csv(file_path, sep="\t")
            all_genes[chrom] = df['gene_symbol'].tolist()
        else:
            print(f"Fetching all genes for chromosome '{chrom}'...")
            
            # --- THIS IS THE FIX ---
            # 1. Get ALL genes on the chromosome first.
            all_genes_on_chrom = ensembl_data.genes(contig=chrom)
            
            # 2. Now, filter this list in Python.
            print(f"Filtering for protein-coding genes...")
            protein_coding_genes = [
                g for g in all_genes_on_chrom 
                if g.biotype == 'protein_coding' and g.gene_name is not None
            ]
            
            gene_symbols = [g.gene_name for g in protein_coding_genes]
            
            print(f"Found {len(gene_symbols)} protein-coding genes. Saving to file: {file_path}")
            pd.DataFrame({'gene_symbol': gene_symbols}).to_csv(file_path, sep="\t", index=False)
            all_genes[chrom] = gene_symbols
            
    return all_genes
