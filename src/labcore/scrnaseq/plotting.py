# src/labcore/scrnaseq/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import pandas as pd

def split_umap(
    adata, split_by, ncol=2, nrow=None, panel_size=4.0, title_height=0.35,
    legend="none", legend_kwargs=None, legend_ncol=1, **kwargs
):
    """Split UMAP into panels with truly square axes and optional global legend."""
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
    adata, markers_by_celltype: dict, basis: str = "umap", ncols: int = 4,
    point_size: float = 5, use_gene_symbols: bool = True, gene_symbol_col: str = "gene_symbol",
    cmap: str = "viridis", vmin=None, vmax=None, save_dir: str | None = None,
    fig_root_name: str | None = "markers", dpi: int = 150
):
    """For each cell type, plot feature plots for its markers on a fixed grid."""
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
