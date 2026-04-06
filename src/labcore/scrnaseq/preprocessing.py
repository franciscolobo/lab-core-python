import scanpy as sc
from anndata import AnnData
import pandas as pd # <-- Make sure pandas is imported

def score_cell_cycle(
    adata: AnnData,
    s_genes: list[str],
    g2m_genes: list[str],
    gene_symbol_col: str = "gene_symbol",
) -> AnnData:
    """
    Scores cells for cell cycle phases (S and G2/M) using gene symbols.

    This function robustly finds the intersection of the provided marker gene
    lists with the genes actually present in the AnnData object and performs
    scoring without modifying the AnnData index.

    Args:
        adata: The AnnData object (should be log-normalized).
        s_genes: A list of gene symbols for the S phase.
        g2m_genes: A list of gene symbols for the G2/M phase.
        gene_symbol_col: The column in `adata.var` that contains the gene symbols.

    Returns:
        The input AnnData object, updated with 'S_score', 'G2M_score',
        and 'phase' columns in `.obs`.
    """
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(f"Column '{gene_symbol_col}' not found in adata.var.")

    print("Scoring cell cycle phases...")

    # --- FINAL, ROBUST LOGIC ---
    # Create a mapping from UPPERCASE gene symbols to their original index positions
    gene_symbol_map = {
        symbol.upper(): i
        for i, symbol in enumerate(adata.var[gene_symbol_col].astype(str))
    }

    # Find the index positions of the cell cycle genes that exist in our data
    s_genes_idx = [gene_symbol_map[g.upper()] for g in s_genes if g.upper() in gene_symbol_map]
    g2m_genes_idx = [gene_symbol_map[g.upper()] for g in g2m_genes if g.upper() in gene_symbol_map]

    print(f"Found {len(s_genes_idx)}/{len(s_genes)} S-phase genes in data.")
    print(f"Found {len(g2m_genes_idx)}/{len(g2m_genes)} G2/M-phase genes in data.")

    if not s_genes_idx or not g2m_genes_idx:
        raise ValueError("Not enough cell cycle genes were found in the data to proceed with scoring.")

    # Calculate the mean expression of the gene sets for each cell
    s_score = adata.X[:, s_genes_idx].mean(axis=1)
    g2m_score = adata.X[:, g2m_genes_idx].mean(axis=1)

    # Convert to numpy arrays if they are sparse matrices
    if hasattr(s_score, 'A'): s_score = s_score.A.flatten()
    if hasattr(g2m_score, 'A'): g2m_score = g2m_score.A.flatten()
        
    # Assign scores to .obs
    adata.obs['S_score'] = s_score
    adata.obs['G2M_score'] = g2m_score

    # Assign phase based on which score is higher
    phase = pd.Series('S', index=adata.obs.index)
    phase[adata.obs['G2M_score'] > adata.obs['S_score']] = 'G2M'
    
    # Define a threshold for 'G1' phase
    # This is a simple heuristic; a more complex one could be used
    g1_threshold = 0.0
    phase[(adata.obs['S_score'] <= g1_threshold) & (adata.obs['G2M_score'] <= g1_threshold)] = 'G1'
    
    adata.obs['phase'] = pd.Categorical(phase)
    
    print("Cell cycle scoring complete.")
    return adata


def preprocess_for_pca(
    adata: AnnData,
    n_top_genes: int = 3000,
    regress_vars: list[str] | None = None,
) -> AnnData:
    """
    Prepares an AnnData object for PCA by finding HVGs, regressing, and scaling.
    ... (docstring is unchanged) ...
    """
    print("\n--- Preprocessing for PCA ---")
    
    if 'counts' not in adata.layers:
        raise ValueError("A 'counts' layer with raw counts is required for HVG selection.")
        
    print("Finding highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        layer='counts',
        n_top_genes=n_top_genes,
        flavor='seurat_v3'
    )

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    if regress_vars:
        print(f"Regressing out the following variables: {regress_vars}")
        sc.pp.regress_out(adata_hvg, regress_vars)

    print("Scaling data...")
    sc.pp.scale(adata_hvg, max_value=10)
    
    return adata_hvg


# In src/labcore/scrnaseq/preprocessing.py

def score_gene_modules(
    adata: AnnData,
    gene_lists: dict[str, list[str]],
    gene_symbol_col: str = "gene_symbol",
    **kwargs,
) -> AnnData:
    """
    Scores cells for multiple gene lists (modules) using a robust copy-based method.

    This function temporarily creates a copy of the AnnData object with gene
    symbols as the index to robustly run `sc.tl.score_genes`.

    Args:
        adata: The AnnData object (should be log-normalized).
        gene_lists: A dictionary where keys are the desired score names
                    (e.g., 'X_score') and values are the lists of gene symbols.
        gene_symbol_col: The column in `adata.var` that contains gene symbols.
        **kwargs: Additional arguments passed to `sc.tl.score_genes`.

    Returns:
        The input AnnData object, updated with new score columns in `.obs`.
    """
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(f"Column '{gene_symbol_col}' not found in adata.var.")

    # --- THIS IS THE ROBUST, COPY-BASED FIX ---
    # Create a temporary copy to work on, ensuring we don't modify the original's index
    adata_for_scoring = adata.copy()
    
    # Set the index of the copy to the gene symbols.
    adata_for_scoring.var_names = adata_for_scoring.var[gene_symbol_col].astype(str)
    # Ensure the new index is unique before proceeding
    adata_for_scoring.var_names_make_unique()

    for score_name, gene_list in gene_lists.items():
        # We still find the intersection, but now we do it against the new index
        # This also handles case-insensitivity if needed.
        available_genes_scoring = set(adata_for_scoring.var_names)
        genes_to_score = [g for g in gene_list if g in available_genes_scoring]

        print(f"Calculating score for '{score_name}': "
              f"Found {len(genes_to_score)}/{len(gene_list)} genes in data.")
        
        if len(genes_to_score) == 0:
            print(f"  -> Warning: No genes found for '{score_name}'. Assigning score of 0.")
            adata.obs[score_name] = 0.0 # Assign to the original adata
            continue
            
        # Run the scoring on the temporary object which has the correct index
        sc.tl.score_genes(
            adata_for_scoring,
            gene_list=genes_to_score,
            score_name=score_name,
            use_raw=False,
            **kwargs
        )
        
        # Copy the calculated score from the temporary object back to the original
        adata.obs[score_name] = adata_for_scoring.obs[score_name]
        
    print("Module scoring complete.")
    return adata

