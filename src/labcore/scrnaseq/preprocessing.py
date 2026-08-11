import scanpy as sc
from anndata import AnnData
import pandas as pd # <-- Make sure pandas is imported

def score_cell_cycle(
    adata: AnnData,
    s_genes: list[str],
    g2m_genes: list[str],
    gene_symbol_col: str = "gene_symbol",
    **kwargs,
) -> AnnData:
    """
    Scores cells for cell cycle phases (S and G2/M) using gene symbols.

    This is the scanpy equivalent of Seurat's `CellCycleScoring()`. Seurat's
    function computes S.Score and G2M.Score via `AddModuleScore()`, which
    scores a gene set relative to randomly sampled control genes matched on
    expression bins -- not a simple mean. Here we use `sc.tl.score_genes()`,
    which implements the same control-gene-subtraction approach, instead of
    a raw average of expression.

    Phase assignment mirrors Seurat's logic: a cell is "S" if its S_score
    exceeds its G2M_score and is > 0; "G2M" if the reverse; otherwise "G1".

    Args:
        adata: The AnnData object (should be log-normalized).
        s_genes: A list of gene symbols for the S phase.
        g2m_genes: A list of gene symbols for the G2/M phase.
        gene_symbol_col: The column in `adata.var` that contains the gene symbols.
        **kwargs: Additional arguments forwarded to `sc.tl.score_genes`
            (e.g. `ctrl_size`, `n_bins`, `random_state`).

    Returns:
        The input AnnData object, updated with 'S_score', 'G2M_score',
        and 'phase' columns in `.obs`.
    """
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(f"Column '{gene_symbol_col}' not found in adata.var.")

    print("Scoring cell cycle phases with sc.tl.score_genes (AddModuleScore-equivalent)...")

    # Work on a copy with gene symbols as var_names so sc.tl.score_genes can
    # match on symbol directly (case-insensitively, like the original logic).
    adata_for_scoring = adata.copy()
    adata_for_scoring.var_names = adata_for_scoring.var[gene_symbol_col].astype(str)
    adata_for_scoring.var_names_make_unique()

    sym_map = {v.upper(): v for v in adata_for_scoring.var_names}
    s_genes_found = [sym_map[g.upper()] for g in s_genes if g.upper() in sym_map]
    g2m_genes_found = [sym_map[g.upper()] for g in g2m_genes if g.upper() in sym_map]

    print(f"Found {len(s_genes_found)}/{len(s_genes)} S-phase genes in data.")
    print(f"Found {len(g2m_genes_found)}/{len(g2m_genes)} G2/M-phase genes in data.")

    if not s_genes_found or not g2m_genes_found:
        raise ValueError("Not enough cell cycle genes were found in the data to proceed with scoring.")

    # Score each gene set against its own set of expression-matched control genes,
    # exactly as AddModuleScore does for each list in Seurat.
    sc.tl.score_genes(
        adata_for_scoring,
        gene_list=s_genes_found,
        score_name="S_score",
        use_raw=False,
        **kwargs,
    )
    sc.tl.score_genes(
        adata_for_scoring,
        gene_list=g2m_genes_found,
        score_name="G2M_score",
        use_raw=False,
        **kwargs,
    )

    # Copy scores back onto the original object
    adata.obs["S_score"] = adata_for_scoring.obs["S_score"]
    adata.obs["G2M_score"] = adata_for_scoring.obs["G2M_score"]

    # Assign phase using Seurat's rule: whichever score is higher AND > 0 wins;
    # if neither score clears 0, the cell is classified as G1.
    phase = pd.Series("G1", index=adata.obs.index)
    is_s = (adata.obs["S_score"] > adata.obs["G2M_score"]) & (adata.obs["S_score"] > 0)
    is_g2m = (adata.obs["G2M_score"] > adata.obs["S_score"]) & (adata.obs["G2M_score"] > 0)
    phase[is_s] = "S"
    phase[is_g2m] = "G2M"

    adata.obs["phase"] = pd.Categorical(phase, categories=["G1", "S", "G2M"])

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

