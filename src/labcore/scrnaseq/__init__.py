# src/labcore/scrnaseq/__init__.py

# Promote key functions to the scrnaseq sub-package level for easy access

# From analysis.py
from .analysis import run_ora

# From io.py
from .io import read_cellbender_matrix_h5, build_gene_map_from_h5_paths

# From qc.py
from .qc import preprocess_sample, filter_outlier_cells, is_outlier

# From plotting.py
from .plotting import (
        split_umap,
        plot_umap_markers_per_celltype,
        plot_grouped_violin,
        plot_qc_metrics,
        plot_umap_grid,
        plot_proportions,
        plot_ora_results
)

from .annotation import load_chromosome_genes_from_biomart

# From utils.py
from .utils import (
    attach_gene_symbols,
    map_markers_to_ids,
    markers_df_to_dict,
    adata_status,
    rank_genes_groups_df,
    get_top_markers,
    print_top_markers
)

# From workflows.py
from .workflows import load_and_preprocess_from_manifest, run_downstream_analysis

# From preprocessing.py
from .preprocessing import score_cell_cycle, preprocess_for_pca, score_gene_modules

from .orthology import get_orthologs

from .utils import describe_adata

# Define what 'from labcore.scrnaseq import *' will import
__all__ = [
    "read_cellbender_matrix_h5",
    "build_gene_map_from_h5_paths",
    "preprocess_sample",
    "filter_outlier_cells",
    "is_outlier",
    "split_umap",
    "plot_umap_markers_per_celltype",
    "plot_qc_metrics",
    "plot_umap_grid",
    "load_chromosome_genes_from_biomart",
    "score_cell_cycle",
    "score_gene_modules",
    "plot_grouped_violin",
    "plot_proportions",
    "plot_ora_results",
    "preprocess_for_pca",
    "attach_gene_symbols",
    "rank_genes_groups_df",
    "get_top_markers",
    "print_top_markers",
    "map_markers_to_ids",
    "markers_df_to_dict",
    "adata_status",
    "describe_adata",
    "load_and_preprocess_from_manifest",
    "run_downstream_analysis",
    "get_orthologs",
    "run_ora",
]
