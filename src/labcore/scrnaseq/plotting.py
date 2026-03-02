# In src/labcore/scrnaseq/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData  # <--- THIS IS THE FIX

# I've also added the 'AnnData' type hint to the other functions for consistency.

def split_umap(
    adata: AnnData, split_by: str, ncol: int = 2, nrow: int = None,
    panel_size: float = 4.0, title_height: float = 0.35,
    legend: str = "none", legend_kwargs: dict = None, legend_ncol: int = 1, **kwargs
):
    """Split UMAP into panels with truly square axes and optional global legend."""
    # ... (function body is unchanged) ...
    if legend_kwargs is None: legend_kwargs = {}
    s = adata.obs[split_by]
    categories = s.cat.categories if hasattr(s, "cat") else pd.Index(s.unique()).sort_values()
    if nrow is None: nrow = int(np.ceil(len(categories) / ncol))

    X = adata.obsm["X_umap"]
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)

    fig_w, fig_h = panel_size * ncol, (panel_size + title_height) * nrow
    fig, axs = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
    axs = np.atleast_1d(axs).flatten()

    for i, cat in enumerate(categories):
        ax = axs[i]
        sc.pl.umap(adata[adata.obs[split_by] == cat], ax=ax, show=False, legend_loc="none", **kwargs)
        if ax.get_legend() is not None: ax.get_legend().remove()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_box_aspect(1)
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.01, str(cat), ha="center", va="bottom")
        ax.set_title("")

    for j in range(len(categories), len(axs)): axs[j].axis("off")

    if legend == "global":
        import matplotlib.patches as mpatches
        color_key = kwargs.get("color")
        if not color_key or color_key not in adata.obs.columns:
            raise ValueError('legend="global" requires color="<obs_key>" in kwargs.')
        if not hasattr(adata.obs[color_key], "cat"): adata.obs[color_key] = adata.obs[color_key].astype("category")
        pal_key = f"{color_key}_colors"
        if pal_key not in adata.uns or not adata.uns[pal_key]: sc.pl.umap(adata, color=color_key, show=False)
        handles = [mpatches.Patch(color=c, label=str(l)) for l, c in zip(adata.obs[color_key].cat.categories, adata.uns[pal_key])]
        fig.legend(handles=handles, loc="center right", frameon=False, ncol=legend_ncol, **legend_kwargs)
        fig.subplots_adjust(right=0.85)

    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    return fig


def plot_umap_markers_per_celltype(
    adata: AnnData, markers_by_celltype: dict, basis: str = "umap", ncols: int = 4,
    point_size: float = 5, use_gene_symbols: bool = True, gene_symbol_col: str = "gene_symbol",
    cmap: str = "viridis", vmin=None, vmax=None, save_dir: str | None = None,
    fig_root_name: str | None = "markers", dpi: int = 150
):
    """For each cell type, plot feature plots for its markers on a fixed grid."""
    # ... (function body is unchanged) ...
    if use_gene_symbols and gene_symbol_col not in adata.var.columns:
        raise ValueError(f"adata.var lacks '{gene_symbol_col}'.")

    max_k = max(len(v) for v in markers_by_celltype.values()) if markers_by_celltype else 0
    if max_k == 0: return
    nrows = int(np.ceil(max_k / ncols))

    for cell_type in sorted(markers_by_celltype.keys()):
        genes = list(markers_by_celltype[cell_type])
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), constrained_layout=True)
        axs = np.array(axs).flatten()
        for i, ax in enumerate(axs):
            if i < len(genes):
                sc.pl.embedding(adata, basis=basis, color=genes[i], gene_symbols=gene_symbol_col if use_gene_symbols else None,
                                ax=ax, show=False, size=point_size, cmap=cmap, vmin=vmin, vmax=vmax, frameon=False)
            else:
                ax.axis("off")
        fig.suptitle(str(cell_type), fontsize=14)
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"{fig_root_name}.{cell_type}.{basis}.png")
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def plot_qc_metrics(
    adata: AnnData,
    save_path: str | None = None,
    dpi: int = 150
) -> None:
    """Generates and optionally saves a standard panel of QC plots."""
    # ... (function body is unchanged) ...
    print(f"Generating QC plots for AnnData object with {adata.n_obs} cells.")
    
    qc_metrics = [
        'n_genes_by_counts', 'total_counts', 'pct_counts_mt',
        'pct_counts_ribo', 'pct_counts_hb'
    ]
    available_metrics = [m for m in qc_metrics if m in adata.obs.columns]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Quality Control Metrics', fontsize=16)

    # Use a temporary AnnData view for the violin plot to avoid modifying the original
    temp_adata_view = adata[:, :0].copy() # Create a view with 0 vars but all obs
    temp_adata_view.obs = adata.obs
    
    sc.pl.violin(
        temp_adata_view,
        keys=available_metrics,
        jitter=0.4,
        multi_panel=True,
        show=False,
        ax=ax1
    )
    ax1.set_title("QC Metric Distributions")
    
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt', ax=ax2, show=False)
    ax2.set_title("Counts vs. Genes (color=MT%)")

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=ax3, show=False)
    ax3.set_title("Counts vs. MT%")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        print(f"Saving QC plot to: {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

