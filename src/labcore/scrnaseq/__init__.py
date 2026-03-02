# src/labcore/scrnaseq/__init__.py

# Promote key functions to the scrnaseq sub-package level for easy access

# From io.py
from .io import read_cellbender_matrix_h5, build_gene_map_from_h5_paths

# From qc.py
from .qc import preprocess_sample, filter_outlier_cells, is_outlier

# From plotting.py
from .plotting import split_umap, plot_umap_markers_per_celltype

# From utils.py
from .utils import (
    attach_gene_symbols,
    map_markers_to_ids,
    markers_df_to_dict,
    adata_status,
)

# Define what 'from labcore.scrnaseq import *' will import
__all__ = [
    "read_cellbender_matrix_h5",
    "build_gene_map_from_h5_paths",
    "preprocess_sample",
    "filter_outlier_cells",
    "is_outlier",
    "split_umap",
    "plot_umap_markers_per_celltype",
    "attach_gene_symbols",
    "map_markers_to_ids",
    "markers_df_to_dict",
    "adata_status",
]
