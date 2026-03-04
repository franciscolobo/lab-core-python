# src/labcore/scrnaseq/__init__.py

# Promote key functions to the scrnaseq sub-package level for easy access

# From io.py
from .io import read_cellbender_matrix_h5, build_gene_map_from_h5_paths

# From qc.py
from .qc import preprocess_sample, filter_outlier_cells, is_outlier

# From plotting.py
from .plotting import split_umap, plot_umap_markers_per_celltype, plot_qc_metrics, plot_umap_grid

# From utils.py
from .utils import (
    attach_gene_symbols,
    map_markers_to_ids,
    markers_df_to_dict,
    adata_status,
)

# From workflows.py
from .workflows import load_and_preprocess_from_manifest, run_downstream_analysis

# From preprocessing.py
from .preprocessing import score_cell_cycle, preprocess_for_pca

from .orthology import get_orthologs

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
    "score_cell_cycle",
    "preprocess_for_pca",
    "attach_gene_symbols",
    "map_markers_to_ids",
    "markers_df_to_dict",
    "adata_status",
    "load_and_preprocess_from_manifest",
    "run_downstream_analysis",
    "get_orthologs",

]
