import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
import os
import seaborn as sns
import gseapy as gp

def split_umap(
    adata: AnnData, split_by: str, ncol: int = 2, nrow: int = None,
    panel_size: float = 4.0, title_height: float = 0.35,
    legend: str = "none", legend_kwargs: dict = None, legend_ncol: int = 1, **kwargs
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
    combine_figures: bool = False,
    panel_size: float = 3.2,
    title_height: float = 0.35,
) -> None:
    """
    For each cell type, plot UMAP feature plots for its markers.

    If combine_figures=True, all cell types are saved into one tall figure.
    Each marker panel keeps the same size; cell types with more markers get more rows.
    """
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import scanpy as sc

    if use_gene_symbols and gene_symbol_col not in adata.var.columns:
        raise ValueError(f"adata.var lacks '{gene_symbol_col}'. Cannot use gene symbols.")
    if not markers_by_celltype:
        print("Warning: markers_by_celltype dictionary is empty. No plots will be generated.")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cell_types_to_plot = sorted([ct for ct, genes in markers_by_celltype.items() if genes])
    if not cell_types_to_plot:
        print("No markers to plot.")
        return

    if combine_figures:
        row_counts = {
            ct: int(np.ceil(len(markers_by_celltype[ct]) / ncols))
            for ct in cell_types_to_plot
        }

        total_rows = sum(row_counts.values())
        fig_w = panel_size * ncols
        fig_h = panel_size * total_rows + title_height * len(cell_types_to_plot)

        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

        height_ratios = [
            row_counts[ct] * panel_size + title_height
            for ct in cell_types_to_plot
        ]

        outer = fig.add_gridspec(
            nrows=len(cell_types_to_plot),
            ncols=1,
            height_ratios=height_ratios,
            hspace=0.35,
        )

        for i, cell_type in enumerate(cell_types_to_plot):
            genes = list(markers_by_celltype[cell_type])
            nrows = row_counts[cell_type]

            sub = outer[i].subgridspec(
                nrows=nrows + 1,
                ncols=ncols,
                height_ratios=[title_height] + [panel_size] * nrows,
                hspace=0.25,
                wspace=0.25,
            )

            title_ax = fig.add_subplot(sub[0, :])
            title_ax.axis("off")
            title_ax.text(
                0.5,
                0.5,
                str(cell_type),
                ha="center",
                va="center",
                fontsize=16,
                weight="bold",
                transform=title_ax.transAxes,
            )

            axs = []
            for r in range(nrows):
                for c in range(ncols):
                    axs.append(fig.add_subplot(sub[r + 1, c]))

            for j, ax in enumerate(axs):
                if j < len(genes):
                    g = genes[j]
                    sc.pl.embedding(
                        adata,
                        basis=basis,
                        color=g,
                        gene_symbols=gene_symbol_col if use_gene_symbols else None,
                        ax=ax,
                        show=False,
                        size=point_size,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        frameon=False,
                        title=str(g),
                    )
                    ax.set_box_aspect(1)
                else:
                    ax.axis("off")

        if save_dir:
            out_path = os.path.join(save_dir, f"{fig_root_name}.all_celltypes_combined.png")
            print(f"Saving combined figure to: {out_path}")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    else:
        for cell_type in cell_types_to_plot:
            genes = list(markers_by_celltype[cell_type])
            nrows = int(np.ceil(len(genes) / ncols))

            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(panel_size * ncols, panel_size * nrows + title_height),
                constrained_layout=True,
                squeeze=False,
            )
            axs = axs.ravel()

            for j, ax in enumerate(axs):
                if j < len(genes):
                    g = genes[j]
                    sc.pl.embedding(
                        adata,
                        basis=basis,
                        color=g,
                        gene_symbols=gene_symbol_col if use_gene_symbols else None,
                        ax=ax,
                        show=False,
                        size=point_size,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        frameon=False,
                        title=str(g),
                    )
                    ax.set_box_aspect(1)
                else:
                    ax.axis("off")

            fig.suptitle(str(cell_type), fontsize=14, weight="bold")

            if save_dir:
                safe_ct = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cell_type))
                out_path = os.path.join(save_dir, f"{fig_root_name}.{safe_ct}.{basis}.png")
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


def plot_umap_grid(
    adata: AnnData,
    color_keys: list[str],
    ncols: int = 2,
    point_size: float = 20,
    wspace: float = 0.4,
    **kwargs
) -> plt.Figure:
    """
    Plots multiple UMAPs in a clean, square-aspected grid.
    ... (docstring is unchanged) ...
    """
    n_plots = len(color_keys)
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axs = plt.subplots(
        nrows, ncols, 
        figsize=(6 * ncols, 6 * nrows), # Use a square-ish size per plot
    )
    
    axs = np.array(axs).flatten()

    # --- THIS IS THE NEW, ROBUST FIX FOR SQUARE PLOTS ---
    # 1. Get the global limits of the UMAP embedding
    umap_coords = adata.obsm['X_umap']
    x_min, y_min = umap_coords.min(axis=0)
    x_max, y_max = umap_coords.max(axis=0)

    # 2. Calculate the range and center, then find the max range
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # 3. Define the new, square limits centered on the data
    x_lim = [x_center - max_range * 0.55, x_center + max_range * 0.55]
    y_lim = [y_center - max_range * 0.55, y_center + max_range * 0.55]
    # The 0.55 factor adds a little padding around the edges

    for i, key in enumerate(color_keys):
        ax = axs[i]
        
        sc.pl.umap(
            adata,
            color=key,
            ax=ax,
            show=False,
            size=point_size,
            title=key,
            **kwargs
        )
        
        # 4. Forcefully apply the calculated square limits to each subplot
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        # This is a final enforcement, which should now work as intended
        ax.set_aspect('equal')

    # Turn off any unused axes
    for i in range(n_plots, len(axs)):
        axs[i].axis('off')

    fig.subplots_adjust(wspace=wspace)

    return fig


def plot_grouped_violin(
    adata: AnnData,
    metrics: list[str],
    group_by: str,
    stripplot: bool = True,
    show_median: bool = False,
    save_prefix: str | None = None,
    dpi: int = 150
) -> None:
    """
    Generates violin plots for multiple QC metrics, grouped by a category.

    This provides more customization than the default scanpy plot, including
    showing median values.

    Args:
        adata: An AnnData object with QC metrics calculated.
        metrics: List of metric columns in .obs to plot (e.g., ['pct_counts_mt']).
        group_by: Column in .obs to group by (e.g., 'SampleID').
        stripplot: If False, do not plot individual data points (the dots).
        show_median: If True, draw a line for the median and print its value.
        save_prefix: If provided, the plots will be saved with this prefix.
        dpi: The resolution for saved figures.
    """
    print("Generating grouped violin plots...")

    for metric in metrics:
        if metric not in adata.obs.columns:
            print(f"Warning: Metric '{metric}' not found in adata.obs. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(max(8, 0.5 * adata.obs[group_by].nunique()), 6))

        # Use seaborn directly for more control
        sns.violinplot(
            x=group_by,
            y=metric,
            data=adata.obs,
            ax=ax,
            inner=None, # We control the inner elements manually
            cut=0,
        )

        if stripplot:
            sns.stripplot(
                x=group_by, y=metric, data=adata.obs,
                ax=ax, jitter=0.4, color='black', size=1.5, alpha=0.3
            )

        if show_median:
            # Calculate and plot medians
            medians = adata.obs.groupby(group_by, observed=True)[metric].median()
            for i, cat in enumerate(medians.index):
                # Draw line for the median
                ax.hlines(medians[cat], i - 0.4, i + 0.4, color='red', lw=1)
                # Add text for the median value
                ax.text(i, medians[cat], f'{medians[cat]:.2f}',
                        ha='center', va='bottom', color='white',
                        bbox=dict(facecolor='red', alpha=0.9, boxstyle='round,pad=0.1'))

        ax.set_title(f'Distribution of {metric} by {group_by}')
        ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()

        if save_prefix:
            out_path = f"{save_prefix}_{metric}_by_{group_by}.png"
            print(f"Saving plot to: {out_path}")
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# In src/labcore/scrnaseq/plotting.py

def plot_proportions(
    adata: AnnData,
    group_by: str,
    category_to_plot: str,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Calculates and plots relative proportions, using scanpy's color palette
    for consistency with UMAP plots.

    ... (docstring) ...
    """
    print(f"Calculating proportions of '{category_to_plot}' within each '{group_by}' group...")

    counts_df = adata.obs.groupby([group_by, category_to_plot], observed=True).size().unstack(fill_value=0)
    proportions_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(proportions_df.index)), 6))

    # --- THIS IS THE NEW LOGIC FOR CONSISTENT COLORS ---
    # 1. Check if a color palette exists in adata.uns
    color_key = f"{category_to_plot}_colors"
    plot_colors = None # Default to None

    if color_key in adata.uns:
        # 2. Create a mapping from the category name to its official color
        category_names = adata.obs[category_to_plot].cat.categories
        color_map = {cat: color for cat, color in zip(category_names, adata.uns[color_key])}

        # 3. Order the colors according to the columns in our proportions_df
        plot_colors = [color_map.get(col) for col in proportions_df.columns]

    # 4. Use the specific list of colors if available, otherwise fall back to a cmap
    proportions_df.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=plot_colors, # <-- Use the specific color list
        cmap='tab20' if plot_colors is None else None # <-- Fallback if no colors found
    )

    # ... (the rest of the plotting code is unchanged) ...
    ax.set_title(f"Proportions of {category_to_plot} by {group_by}")
    ax.set_xlabel(group_by)
    ax.set_ylabel("Proportion of Cells")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend(title=category_to_plot, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    fig.subplots_adjust(right=0.8)

    if save_path:
        print(f"Saving plot to: {save_path}")
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_ora_results(
    ora_results: pd.DataFrame,
    top_n: int = 15,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Visualizes ORA results from gseapy.enrichr as a dot plot.

    Args:
        ora_results: The DataFrame of enrichment results from `run_ora`.
        top_n: The number of top terms (by Adjusted P-value) to display.
        save_path: If provided, the figure will be saved to this path.
        dpi: The resolution for the saved figure.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    if ora_results.empty:
        print("Input DataFrame is empty. Cannot generate plot.")
        # Return an empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Enrichment Results", ha='center', va='center')
        return fig

    # gseapy's dotplot function is excellent for this
    # It requires a gseapy.Enrichr object, so we create one from the results
    # We can also call the function directly with the dataframe
    print(f"Plotting top {top_n} enriched terms...")

    fig = gp.dotplot(
        ora_results,
        title=ora_results['Gene_set'].iloc[0], # Use the library name as title
        x='Combined Score',
        top_term=top_n,
        show_ring=True, # Shows p-value as ring color
        ofname=save_path, # gseapy can handle saving directly
        dpi=dpi,
        figsize=(6, 0.5 * top_n) # Adjust height based on number of terms
    )

    return fig
