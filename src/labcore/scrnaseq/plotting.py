import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
import os
import seaborn as sns
import gseapy as gp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PLOTLY_QUALITATIVE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _build_hover_text(adata, indices, color_key, color_values, hover_cols):
    """Construct one hover string per cell (barcode + color key + extra obs cols)."""
    barcodes = adata.obs_names[indices]
    lines = []
    extra_cols = [c for c in (hover_cols or []) if c in adata.obs.columns]
    for i, idx in enumerate(indices):
        parts = [f"Cell: {barcodes[i]}", f"{color_key}: {color_values[i]}"]
        for c in extra_cols:
            parts.append(f"{c}: {adata.obs[c].iloc[idx]}")
        lines.append("<br>".join(parts))
    return lines


def _get_color_values(adata, key):
    """Fetch a color vector for `key` from .obs (categorical/continuous) or gene expression."""
    if key in adata.obs.columns:
        col = adata.obs[key]
        is_categorical = hasattr(col, "cat") or col.dtype == object
        return col.to_numpy(), is_categorical
    elif key in adata.var_names:
        X = adata[:, key].X
        vals = X.toarray().flatten() if hasattr(X, "toarray") else np.asarray(X).flatten()
        return vals, False
    else:
        raise KeyError(f"'{key}' not found in adata.obs.columns or adata.var_names.")


def _square_limits(coords, pad_frac=0.05):
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * (1 + pad_frac)
    return [xc - half, xc + half], [yc - half, yc + half]


# ---------------------------------------------------------------------------
# Interactive UMAP grid
# ---------------------------------------------------------------------------

def plot_umap_grid_interactive(
    adata: AnnData,
    color_keys: list[str],
    ncols: int = 2,
    point_size: float = 4,
    hover_cols: list[str] | None = None,
    colorscale: str = "Viridis",
    height_per_panel: int = 450,
    width_per_panel: int = 450,
) -> go.Figure:
    """Interactive Plotly version of `plot_umap_grid`.

    Each point shows a hover tooltip with cell barcode, the panel's color
    value, and any extra `hover_cols` from `.obs` (e.g. sample ID, QC
    metrics). Panels share square, synchronized axis limits.

    Args:
        adata: AnnData with `X_umap` in `.obsm`.
        color_keys: obs columns or gene names to color/panel by.
        ncols: Number of columns in the subplot grid.
        point_size: Marker size.
        hover_cols: Extra `.obs` columns to include in every tooltip
            (e.g. `["sample_id", "total_counts", "pct_counts_mt"]`).
        colorscale: Plotly colorscale for continuous keys.
        height_per_panel: Panel height in px.
        width_per_panel: Panel width in px.

    Returns:
        A `plotly.graph_objects.Figure`. Call `.show()` or `.write_html(...)`.
    """
    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] not found. Run UMAP first.")

    coords = adata.obsm["X_umap"]
    x_lim, y_lim = _square_limits(coords)
    n = len(color_keys)
    nrows = int(np.ceil(n / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=color_keys,
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    seen_categories = set()
    for i, key in enumerate(color_keys):
        row, col = i // ncols + 1, i % ncols + 1
        values, is_categorical = _get_color_values(adata, key)

        if is_categorical:
            categories = pd.Index(values).unique()
            palette_key = f"{key}_colors"
            if palette_key in adata.uns and hasattr(adata.obs[key], "cat"):
                cat_order = list(adata.obs[key].cat.categories)
                color_map = dict(zip(cat_order, adata.uns[palette_key]))
            else:
                color_map = {
                    c: _PLOTLY_QUALITATIVE[j % len(_PLOTLY_QUALITATIVE)]
                    for j, c in enumerate(categories)
                }
            for cat in categories:
                mask = values == cat
                idx = np.where(mask)[0]
                fig.add_trace(
                    go.Scattergl(
                        x=coords[mask, 0], y=coords[mask, 1],
                        mode="markers",
                        marker=dict(size=point_size, color=color_map.get(cat, "gray")),
                        name=str(cat),
                        legendgroup=str(cat),
                        showlegend=str(cat) not in seen_categories,
                        text=_build_hover_text(adata, idx, key, values[mask], hover_cols),
                        hoverinfo="text",
                    ),
                    row=row, col=col,
                )
                seen_categories.add(str(cat))
        else:
            idx = np.arange(adata.n_obs)
            fig.add_trace(
                go.Scattergl(
                    x=coords[:, 0], y=coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=point_size, color=values, colorscale=colorscale,
                        colorbar=dict(len=0.9 / nrows, y=1 - (row - 0.5) / nrows, thickness=12),
                        showscale=True,
                    ),
                    showlegend=False,
                    text=_build_hover_text(adata, idx, key, values, hover_cols),
                    hoverinfo="text",
                ),
                row=row, col=col,
            )

        fig.update_xaxes(range=x_lim, showticklabels=False, row=row, col=col)
        fig.update_yaxes(range=y_lim, showticklabels=False, scaleanchor=f"x{i+1}" if i > 0 else "x", row=row, col=col)

    fig.update_layout(
        height=height_per_panel * nrows,
        width=width_per_panel * ncols,
        legend=dict(title="", itemsizing="constant"),
        margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Interactive split UMAP
# ---------------------------------------------------------------------------

def split_umap_interactive(
    adata: AnnData,
    split_by: str,
    color: str,
    ncols: int = 2,
    point_size: float = 4,
    hover_cols: list[str] | None = None,
    colorscale: str = "Viridis",
    height_per_panel: int = 400,
    width_per_panel: int = 400,
) -> go.Figure:
    """Interactive Plotly version of `split_umap`.

    Facets the UMAP by categories of `split_by`, coloring every panel by
    the single `color` key. Hover shows cell barcode, the color value, and
    any extra `hover_cols`.

    Args:
        adata: AnnData with `X_umap` in `.obsm`.
        split_by: obs column whose categories define the panels.
        color: obs column or gene name to color points by (shared across panels).
        ncols: Number of columns in the subplot grid.
        point_size: Marker size.
        hover_cols: Extra `.obs` columns to include in tooltips.
        colorscale: Plotly colorscale used when `color` is continuous.
        height_per_panel: Panel height in px.
        width_per_panel: Panel width in px.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] not found. Run UMAP first.")
    if split_by not in adata.obs.columns:
        raise KeyError(f"'{split_by}' not found in adata.obs.")

    s = adata.obs[split_by]
    categories = list(s.cat.categories) if hasattr(s, "cat") else sorted(s.unique())
    coords = adata.obsm["X_umap"]
    x_lim, y_lim = _square_limits(coords)
    values, is_categorical = _get_color_values(adata, color)

    nrows = int(np.ceil(len(categories) / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=[str(c) for c in categories],
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )

    # Precompute a shared color scheme if `color` is categorical
    if is_categorical:
        uniq = pd.Index(values).unique()
        palette_key = f"{color}_colors"
        if palette_key in adata.uns and hasattr(adata.obs[color], "cat"):
            cat_order = list(adata.obs[color].cat.categories)
            color_map = dict(zip(cat_order, adata.uns[palette_key]))
        else:
            color_map = {c: _PLOTLY_QUALITATIVE[j % len(_PLOTLY_QUALITATIVE)] for j, c in enumerate(uniq)}

    seen_categories = set()
    for i, cat in enumerate(categories):
        row, col = i // ncols + 1, i % ncols + 1
        panel_mask = (s == cat).to_numpy()
        idx = np.where(panel_mask)[0]

        if is_categorical:
            for sub_cat in pd.Index(values[panel_mask]).unique():
                sub_mask = panel_mask & (values == sub_cat)
                sub_idx = np.where(sub_mask)[0]
                fig.add_trace(
                    go.Scattergl(
                        x=coords[sub_mask, 0], y=coords[sub_mask, 1],
                        mode="markers",
                        marker=dict(size=point_size, color=color_map.get(sub_cat, "gray")),
                        name=str(sub_cat),
                        legendgroup=str(sub_cat),
                        showlegend=str(sub_cat) not in seen_categories,
                        text=_build_hover_text(adata, sub_idx, color, values[sub_mask], hover_cols),
                        hoverinfo="text",
                    ),
                    row=row, col=col,
                )
                seen_categories.add(str(sub_cat))
        else:
            fig.add_trace(
                go.Scattergl(
                    x=coords[panel_mask, 0], y=coords[panel_mask, 1],
                    mode="markers",
                    marker=dict(
                        size=point_size, color=values[panel_mask], colorscale=colorscale,
                        showscale=(i == 0),
                        colorbar=dict(thickness=12) if i == 0 else None,
                    ),
                    showlegend=False,
                    text=_build_hover_text(adata, idx, color, values[panel_mask], hover_cols),
                    hoverinfo="text",
                ),
                row=row, col=col,
            )

        fig.update_xaxes(range=x_lim, showticklabels=False, row=row, col=col)
        fig.update_yaxes(range=y_lim, showticklabels=False, scaleanchor=f"x{i+1}" if i > 0 else "x", row=row, col=col)

    fig.update_layout(
        height=height_per_panel * nrows,
        width=width_per_panel * ncols,
        legend=dict(title=color, itemsizing="constant"),
        margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Interactive stacked proportions
# ---------------------------------------------------------------------------

def plot_proportions_interactive(
    adata: AnnData,
    group_by: str,
    category_to_plot: str,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Interactive Plotly version of `plot_proportions`.

    Hovering over any bar segment shows the group, the category, its
    proportion, and its raw cell count. Uses the same `.uns` color
    palette as scanpy's UMAP plots when available, for visual consistency.

    Args:
        adata: AnnData object.
        group_by: obs column defining the x-axis groups (e.g. sample ID).
        category_to_plot: obs column whose category proportions are stacked
            within each group (e.g. cell type).
        height: Figure height in px.
        width: Figure width in px. Defaults to scaling with number of groups.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    counts_df = adata.obs.groupby([group_by, category_to_plot], observed=True).size().unstack(fill_value=0)
    proportions_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    color_key = f"{category_to_plot}_colors"
    if color_key in adata.uns and hasattr(adata.obs[category_to_plot], "cat"):
        cat_order = list(adata.obs[category_to_plot].cat.categories)
        color_map = dict(zip(cat_order, adata.uns[color_key]))
    else:
        color_map = {
            c: _PLOTLY_QUALITATIVE[j % len(_PLOTLY_QUALITATIVE)]
            for j, c in enumerate(proportions_df.columns)
        }

    fig = go.Figure()
    for cat in proportions_df.columns:
        fig.add_trace(
            go.Bar(
                x=proportions_df.index.astype(str),
                y=proportions_df[cat],
                name=str(cat),
                marker_color=color_map.get(cat, "gray"),
                customdata=counts_df[cat].to_numpy(),
                hovertemplate=(
                    f"Group: %{{x}}<br>{category_to_plot}: {cat}"
                    "<br>Proportion: %{y:.1%}<br>Count: %{customdata}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        height=height,
        width=width or max(500, 40 * len(proportions_df.index)),
        yaxis=dict(title="Proportion of Cells", range=[0, 1], tickformat=".0%"),
        xaxis=dict(title=group_by, tickangle=45),
        legend=dict(title=category_to_plot),
        margin=dict(t=40, r=20, b=80, l=60),
    )
    return fig

def plot_categorical_heatmap(
    adata: AnnData,
    cat_x: str,
    cat_y: str,
    cat_split: str | None = None,
    normalize: str | None = None,
    log_scale: bool = False,
    annotate: bool = True,
    annotate_fmt: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    annotate_fontsize: float = 8,
    annotate_color_threshold: float = 0.5,
    group_gap: float = 0.5,
    split_label_fontsize: float = 7,
    group_label_fontsize: float = 10,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plots a heatmap of cell counts (or proportions) between two categorical variables.

    Cross-tabulates `cat_x` and `cat_y` from `adata.obs` and renders the
    result as a heatmap, e.g. cluster vs. sample, cell type vs. condition,
    or any two categorical `.obs` columns.

    Optionally, a third categorical variable (`cat_split`) splits each
    `cat_x` category into side-by-side sub-columns -- e.g. `cat_x` = cell
    type, `cat_y` = time point, `cat_split` = genotype -- so that the two
    (or more) levels of `cat_split` can be visually compared next to each
    other, for every cell type, across every time point.

    Args:
        adata: AnnData object.
        cat_x: `.obs` column plotted along the x-axis (grouped columns of
            the cross-tab).
        cat_y: `.obs` column plotted along the y-axis (rows of the
            cross-tab).
        cat_split: Optional `.obs` column used to split each `cat_x`
            group into side-by-side sub-columns (e.g. genotype, condition).
            When provided, `cat_x` categories are rendered as visually
            separated groups, each containing one sub-column per
            `cat_split` category. Defaults to None (no splitting).
        normalize: How to normalize the cross-tabulated counts before
            plotting. One of:
                - None: plot raw cell counts (default).
                - "row": each row (per `cat_y` category) sums to 1. When
                  `cat_split` is provided, normalization is done
                  *separately within each `cat_split` level* -- i.e. each
                  genotype's cell-type composition sums to 1 on its own,
                  so genotypes with different total cell numbers remain
                  comparable side by side.
                - "col": each column (per `cat_x` / `cat_x`+`cat_split`
                  combination) sums to 1.
                - "all": the entire table sums to 1.
        log_scale: If True, color the heatmap using log1p-transformed
            values while still annotating cells with the original
            (non-log) values. Defaults to False.
        annotate: If True, print the value inside each cell. Defaults to
            True.
        annotate_fmt: Format spec used for the annotations (e.g. `"d"`
            for integers, `".1%"` for percentages, `".2f"` for floats).
            If None, defaults to `"d"` when `normalize is None` and
            `".1%"` otherwise.
        cmap: Matplotlib colormap name.
        figsize: Figure size `(width, height)`. Defaults to scaling with
            the number of categories.
        annotate_fontsize: Font size for in-cell annotations.
        annotate_color_threshold: Fraction (0-1) of the color scale above
            which annotation text switches from black to white, for
            legibility against dark cells. Defaults to 0.5.
        group_gap: Horizontal gap (in cell-width units) inserted between
            `cat_x` groups when `cat_split` is provided. Ignored
            otherwise. Defaults to 0.5.
        split_label_fontsize: Font size for the inner (`cat_split`) tick
            labels, shown under each sub-column. Only used when
            `cat_split` is provided.
        group_label_fontsize: Font size for the outer (`cat_x`) group
            labels, shown below the split labels. Only used when
            `cat_split` is provided.
        save_path: If provided, saves the figure to this path.
        dpi: Resolution for the saved figure.

    Returns:
        The matplotlib Figure object containing the heatmap.

    Raises:
        ValueError: If `cat_x`, `cat_y`, or `cat_split` are not found in
            `adata.obs`, or if `normalize` is not one of
            `{None, "row", "col", "all"}`.
    """
    for name in (cat_x, cat_y) + ((cat_split,) if cat_split else ()):
        if name not in adata.obs.columns:
            raise ValueError(f"'{name}' not found in adata.obs.")
    if normalize not in (None, "row", "col", "all"):
        raise ValueError(f"normalize must be one of {{None, 'row', 'col', 'all'}}, got '{normalize}'.")

    def _categories(col):
        s = adata.obs[col]
        return list(s.cat.categories) if hasattr(s, "cat") else sorted(s.unique())

    y_categories = _categories(cat_y)
    x_categories = _categories(cat_x)

    # --- Build the cross-tab (2-way or 3-way) ---------------------------
    if cat_split is not None:
        split_categories = _categories(cat_split)
        counts_df = pd.crosstab(adata.obs[cat_y], [adata.obs[cat_x], adata.obs[cat_split]])
        full_columns = pd.MultiIndex.from_product([x_categories, split_categories], names=[cat_x, cat_split])
        counts_df = counts_df.reindex(index=y_categories, columns=full_columns, fill_value=0)
    else:
        counts_df = pd.crosstab(adata.obs[cat_y], adata.obs[cat_x])
        counts_df = counts_df.reindex(index=y_categories, columns=x_categories, fill_value=0)

    # --- Normalize --------------------------------------------------------
    if normalize == "row":
        if cat_split is not None:
            plot_df = counts_df.astype(float).copy()
            for sv in split_categories:
                sub = counts_df.xs(sv, level=1, axis=1)
                sub_norm = sub.div(sub.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
                for xv in x_categories:
                    plot_df[(xv, sv)] = sub_norm[xv]
        else:
            plot_df = counts_df.div(counts_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    elif normalize == "col":
        plot_df = counts_df.div(counts_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    elif normalize == "all":
        total = counts_df.values.sum()
        plot_df = counts_df / total if total > 0 else counts_df.astype(float)
    else:
        plot_df = counts_df.astype(float)

    color_df = np.log10(plot_df + 1) if log_scale else plot_df
    annotate_df = plot_df

    if annotate_fmt is None:
        annotate_fmt = "d" if normalize is None else ".1%"

    n_rows = len(y_categories)

    # --- Lay out columns (inserting NaN gap columns between groups) ------
    if cat_split is not None:
        n_split = len(split_categories)
        col_values = []  # tuples (xv, sv), or None for a gap column
        for gi, xv in enumerate(x_categories):
            for sv in split_categories:
                col_values.append((xv, sv))
            if gi < len(x_categories) - 1:
                col_values.append(None)
    else:
        col_values = list(x_categories)

    n_total_cols = len(col_values)
    color_matrix = np.full((n_rows, n_total_cols), np.nan)
    annotate_matrix = np.full((n_rows, n_total_cols), np.nan)
    real_col_idx = []
    group_col_idx: dict = {}

    for ci, cv in enumerate(col_values):
        if cv is None:
            continue
        color_matrix[:, ci] = color_df[cv].to_numpy()
        annotate_matrix[:, ci] = annotate_df[cv].to_numpy()
        real_col_idx.append(ci)
        group_key = cv[0] if cat_split is not None else cv
        group_col_idx.setdefault(group_key, []).append(ci)

    # Non-uniform cell edges: normal width = 1.0, gap width = group_gap
    edges = [0.0]
    col_centers = np.full(n_total_cols, np.nan)
    cursor = 0.0
    for ci, cv in enumerate(col_values):
        width = group_gap if cv is None else 1.0
        col_centers[ci] = cursor + width / 2
        cursor += width
        edges.append(cursor)
    edges = np.array(edges)
    y_edges = np.arange(n_rows + 1, dtype=float)

    # --- Plot ---------------------------------------------------------------
    if figsize is None:
        fig_w = max(6, 0.45 * edges[-1] + 2)
        fig_h = max(4, 0.5 * n_rows + (3.2 if cat_split is not None else 2))
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)

    masked_vals = np.ma.masked_invalid(color_matrix)
    im = ax.pcolormesh(edges, y_edges, masked_vals, cmap=cmap, edgecolors="none", linewidth=0, antialiased=False)
    ax.invert_yaxis()
    ax.set_xlim(0, edges[-1])
    ax.grid(False)

    # y ticks
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(y_categories)
    ax.set_ylabel(cat_y)

    # x ticks
    if cat_split is not None:
        ax.set_xticks(col_centers[real_col_idx])
        split_labels = [col_values[i][1] for i in real_col_idx]
        ax.set_xticklabels(
            split_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=split_label_fontsize
        )
        trans = ax.get_xaxis_transform()
        for xv in x_categories:
            idxs = group_col_idx[xv]
            center = np.mean(col_centers[idxs])
            ax.text(
                center - 0.3, -0.30, str(xv), ha="right", va="top", rotation=45, rotation_mode="anchor",
                transform=trans, fontsize=group_label_fontsize, fontweight="bold",
            )
        ax.set_xlabel(cat_x, labelpad=90)
        fig.subplots_adjust(bottom=0.42)
    else:
        ax.set_xticks(col_centers)
        ax.set_xticklabels(x_categories, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel(cat_x)

    cbar_label = "log10(count+1)" if (log_scale and normalize is None) else \
                 "log10(proportion+1)" if log_scale else \
                 "Proportion" if normalize else "Count"
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    if annotate:
        vmin, vmax = np.nanmin(color_matrix), np.nanmax(color_matrix)
        color_range = (vmax - vmin) if vmax > vmin else 1.0
        for i in range(n_rows):
            for ci in real_col_idx:
                value = annotate_matrix[i, ci]
                if np.isnan(value):
                    continue
                norm_val = (color_matrix[i, ci] - vmin) / color_range
                text_color = "white" if norm_val > annotate_color_threshold else "black"
                if annotate_fmt.endswith("d"):
                    value = int(round(value))
                ax.text(
                    col_centers[ci], i + 0.5, format(value, annotate_fmt),
                    ha="center", va="center",
                    color=text_color, fontsize=annotate_fontsize,
                )

    title_bits = [f"{cat_y} vs. {cat_x}"]
    if cat_split is not None:
        title_bits.append(f"split by {cat_split}")
    if normalize:
        title_bits.append(f"(normalized by {normalize})")
    ax.set_title(" ".join(title_bits))

    if cat_split is None:
        fig.tight_layout()

    if save_path:
        print(f"Saving plot to: {save_path}")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
