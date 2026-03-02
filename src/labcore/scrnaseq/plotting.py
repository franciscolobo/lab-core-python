# In src/labcore/scrnaseq/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
import os

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
    adata: AnnData,
    markers_by_celltype: dict,
    basis: str = "umap",
    ncols: int = 4,
    point_size: float = 5,
    use_gene_symbols: bool = True,
    gene_symbol_col: str = "gene_symbol",
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    save_dir: str | None = None,
    fig_root_name: str | None = "markers",
    dpi: int = 150,
    combine_figures: bool = False, # <-- NEW PARAMETER
) -> None:
    """
    For each cell type, plot UMAP feature plots for its markers.

    Can save plots as one image per cell type or a single combined image.

    Args:
        adata: The AnnData object.
        markers_by_celltype: Dictionary where keys are cell types and values are lists of marker genes.
        basis: The embedding to use (e.g., 'umap').
        ncols: Number of columns for the marker grid.
        point_size: Size of points in the scatter plots.
        use_gene_symbols: Whether to use gene symbols from `gene_symbol_col`.
        gene_symbol_col: Column in `adata.var` containing gene symbols.
        cmap: Colormap for expression.
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.
        save_dir: Directory to save the output files.
        fig_root_name: A prefix for the output filenames.
        dpi: Resolution for saved figures.
        combine_figures: If True, save all cell type plots into a single, tall image.
                         If False (default), save one image per cell type.
    """
    if use_gene_symbols and gene_symbol_col not in adata.var.columns:
        raise ValueError(f"adata.var lacks '{gene_symbol_col}'. Cannot use gene symbols.")
    if not markers_by_celltype:
        print("Warning: markers_by_celltype dictionary is empty. No plots will be generated.")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- NEW LOGIC FOR COMBINED FIGURE ---
    if combine_figures:
        total_rows = 0
        for cell_type in sorted(markers_by_celltype.keys()):
            num_markers = len(markers_by_celltype[cell_type])
            total_rows += int(np.ceil(num_markers / ncols)) if num_markers > 0 else 0

        if total_rows == 0:
            print("No markers to plot in combined figure.")
            return

        fig_w = 3.2 * ncols
        fig_h = 3.2 * total_rows  # Adjust height based on total rows
        fig, all_axs = plt.subplots(total_rows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
        all_axs = np.array(all_axs).flatten()
        
        current_ax_idx = 0
        for cell_type in sorted(markers_by_celltype.keys()):
            genes = list(markers_by_celltype[cell_type])
            if not genes:
                continue

            num_cell_rows = int(np.ceil(len(genes) / ncols))
            num_plots_for_cell = num_cell_rows * ncols
            
            # Add a title for this section
            first_ax_pos = all_axs[current_ax_idx].get_position()
            fig.text(
                0.5, first_ax_pos.y1 + 0.01 / fig.get_figheight(), str(cell_type), 
                ha='center', va='bottom', fontsize=16, weight='bold', transform=fig.transFigure
            )

            for i in range(num_plots_for_cell):
                ax = all_axs[current_ax_idx + i]
                if i < len(genes):
                    g = genes[i]
                    sc.pl.embedding(
                        adata, basis=basis, color=g,
                        gene_symbols=gene_symbol_col if use_gene_symbols else None,
                        ax=ax, show=False, size=point_size, cmap=cmap,
                        vmin=vmin, vmax=vmax, frameon=False, title=str(g)
                    )
                else:
                    ax.axis("off")
            current_ax_idx += num_plots_for_cell

        if save_dir:
            out_path = os.path.join(save_dir, f"{fig_root_name}.all_celltypes_combined.png")
            print(f"Saving combined figure to: {out_path}")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # --- ORIGINAL LOGIC FOR SEPARATE FIGURES ---
    else:
        for cell_type in sorted(markers_by_celltype.keys()):
            genes = list(markers_by_celltype[cell_type])
            if not genes:
                continue

            nrows = int(np.ceil(len(genes) / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), constrained_layout=True)
            axs = np.array(axs).flatten()

            for i, ax in enumerate(axs):
                if i < len(genes):
                    g = genes[i]
                    sc.pl.embedding(
                        adata, basis=basis, color=g,
                        gene_symbols=gene_symbol_col if use_gene_symbols else None,
                        ax=ax, show=False, size=point_size, cmap=cmap,
                        vmin=vmin, vmax=vmax, frameon=False, title=str(g)
                    )
                else:
                    ax.axis("off")

            fig.suptitle(str(cell_type), fontsize=14)
            
            if save_dir:
                out_path = os.path.join(save_dir, f"{fig_root_name}.{cell_type}.{basis}.png")
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

def plot_qc_metrics(
    adata: AnnData,
    save_prefix: str | None = None,
    dpi: int = 150
) -> None:
    """
    Generates and optionally saves a standard panel of QC plots as separate files.
    ... (docstring is unchanged) ...
    """
    print(f"Generating QC plots for AnnData object with {adata.n_obs} cells.")

    qc_metrics = [
        'n_genes_by_counts', 'total_counts', 'pct_counts_mt',
        'pct_counts_ribo', 'pct_counts_hb'
    ]
    available_metrics = [m for m in qc_metrics if m in adata.obs.columns]

    # --- 1. VIOLIN PLOTS ---
    print("Generating violin plots...")

    # Let scanpy create the plot. We do NOT pass 'return_fig' here.
    sc.pl.violin(
        adata,
        keys=available_metrics,
        jitter=0.4,
        multi_panel=True,
        show=False, # <-- Keep this to prevent premature display
    )

    # --- THIS IS THE FIX ---
    # After the plot is created, grab the current figure using matplotlib's standard function.
    fig_violin = plt.gcf()
    fig_violin.suptitle("QC Metric Distributions", y=1.02)

    if save_prefix:
        violin_path = f"{save_prefix}_violins.png"
        print(f"Saving violin plots to: {violin_path}")
        os.makedirs(os.path.dirname(violin_path) or '.', exist_ok=True)
        fig_violin.savefig(violin_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_violin) # Important: close the figure after saving
    else:
        plt.show() # Or show it if not saving

    # --- 2. SCATTER PLOTS ---
    # This part was already correct and remains unchanged.
    print("Generating scatter plots...")
    fig_scatter, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig_scatter.suptitle('QC Scatter Plots', fontsize=16)

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt', ax=ax1, show=False)
    ax1.set_title("Counts vs. Genes (color=MT%)")

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=ax2, show=False)
    ax2.set_title("Counts vs. MT%")

    fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_prefix:
        scatter_path = f"{save_prefix}_scatters.png"
        print(f"Saving scatter plots to: {scatter_path}")
        os.makedirs(os.path.dirname(scatter_path) or '.', exist_ok=True)
        fig_scatter.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_scatter)
    else:
        plt.show()

