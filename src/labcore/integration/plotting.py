"""
Plotting utilities for cross-species integration results.

Currently supports SAMap outputs (Sankey diagrams, UMAP scatter).
Holoviews and Bokeh are imported lazily — only required when the
Sankey functions are called.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# SAMap — Sankey
# ---------------------------------------------------------------------------

def sankey_plot_3(M, species_order=None, align_thr=0.1,
                  two_panel=True, show=False, save_to=None,
                  suppress_warnings=True, **params):
    """Build a Sankey diagram from a SAMap mapping table.

    If ``two_panel=True`` and there are exactly 3 species in
    ``species_order``, renders two synchronized Sankeys
    (species[0]↔species[1] and species[1]↔species[2]) so that the
    middle bars remain centered.

    Args:
        M: Mapping matrix (``pandas.DataFrame``) with index/columns
            labelled as ``"sp_Celltype"`` (e.g. ``"sm_Neuron"``).
        species_order: Left-to-right species order, e.g.
            ``['sm', 'ml', 'sc']``. Inferred from ``M`` when ``None``.
        align_thr: Drop edges whose value is below this threshold.
            Defaults to 0.1.
        two_panel: When ``True`` and exactly 3 species are present,
            build two side-by-side synchronized Sankeys. Defaults to
            ``True``.
        show: Explicitly render the plot via ``bokeh.io.show``.
            Defaults to ``False`` (useful in notebook contexts where
            the object is displayed on return).
        save_to: File path for an HTML export. Attempts two independent
            save routes for robustness. Skipped when ``None``.
        suppress_warnings: Suppress ``FutureWarning`` noise from
            holoviews and pandas. Defaults to ``True``.
        **params: Forwarded to holoviews ``opts``. Recognised keys:
            ``cmap``, ``edge_line_width``, ``node_padding``,
            ``node_alpha``, ``node_width``, ``node_sort``, ``height``,
            ``width`` / ``width_each``, ``bgcolor``.

    Returns:
        A HoloViews ``Sankey`` object, or a two-panel layout when
        ``two_panel=True``.

    Raises:
        TypeError: If ``M`` is not a ``pandas.DataFrame``.
        ImportError: If ``holoviews`` is not installed.
        RuntimeError: If saving to ``save_to`` fails via both routes.
    """
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning, module="holoviews")
        warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    if not isinstance(M, pd.DataFrame):
        raise TypeError("M must be a pandas.DataFrame")

    def _species(arr):
        return np.array([s.split('_', 1)[0] for s in arr], dtype=str)

    def _prep_edges(mat_df, left_sp, right_sp):
        rows = mat_df.index.to_numpy(str)
        cols = mat_df.columns.to_numpy(str)
        A = np.asarray(mat_df.to_numpy(copy=True), dtype=float)
        A[~np.isfinite(A)] = 0.0
        A[A < align_thr] = 0.0
        r_idx, c_idx = A.nonzero()
        if r_idx.size == 0:
            return pd.DataFrame(columns=["source", "target", "value"])
        edges = pd.DataFrame({
            "source": rows[r_idx].astype(str),
            "target": cols[c_idx].astype(str),
            "value":  A[r_idx, c_idx].astype(float),
        })
        sL = _species(edges["source"].values) == left_sp
        tR = _species(edges["target"].values) == right_sp
        sR = _species(edges["source"].values) == right_sp
        tL = _species(edges["target"].values) == left_sp
        keep_lr = sL & tR
        keep_rl = sR & tL
        if keep_rl.any():
            tmp = edges.loc[keep_rl, "source"].values.copy()
            edges.loc[keep_rl, "source"] = edges.loc[keep_rl, "target"].values
            edges.loc[keep_rl, "target"] = tmp
        edges = edges.loc[keep_lr | keep_rl].copy()
        if edges.empty:
            return edges
        return edges.groupby(["source", "target"], as_index=False, sort=False)["value"].sum()

    def _nodes_from_edges(edges, left_sp, right_sp, center_order=None):
        used = pd.Index(edges["source"]).union(pd.Index(edges["target"])).astype(str)
        df = pd.DataFrame({"index": used.values, "species": _species(used.values)})
        df["depth"] = df["species"].map({left_sp: 0, right_sp: 1}).fillna(-1).astype(int)
        if center_order is not None and right_sp in center_order:
            right_nodes = df.loc[df["species"] == right_sp, "index"].tolist()
            order_map = {k: i for i, k in enumerate(x for x in center_order if x in right_nodes)}
            df["__ord"] = df.apply(
                lambda r: order_map.get(r["index"], 10**9) if r["species"] == right_sp else r["index"],
                axis=1,
            )
            df.sort_values(["depth", "__ord"], kind="mergesort", inplace=True, ignore_index=True)
            df.drop(columns="__ord", inplace=True)
        elif center_order is not None and left_sp in center_order:
            left_nodes = df.loc[df["species"] == left_sp, "index"].tolist()
            order_map = {k: i for i, k in enumerate(x for x in center_order if x in left_nodes)}
            df["__ord"] = df.apply(
                lambda r: order_map.get(r["index"], 10**9) if r["species"] == left_sp else r["index"],
                axis=1,
            )
            df.sort_values(["depth", "__ord"], kind="mergesort", inplace=True, ignore_index=True)
            df.drop(columns="__ord", inplace=True)
        else:
            df.sort_values(["depth", "index"], kind="mergesort", inplace=True, ignore_index=True)
        return df

    # Resolve species order
    all_rows_sp = _species(M.index.to_numpy(str))
    all_cols_sp = _species(M.columns.to_numpy(str))
    ids = (
        np.array(species_order, dtype=str)
        if species_order is not None
        else np.unique(np.concatenate([all_rows_sp, all_cols_sp]))
    )

    # Lazy holoviews / bokeh imports
    try:
        import holoviews as hv
        from holoviews import dim
        from bokeh.io import output_notebook
        output_notebook(verbose=False)
        hv.extension("bokeh", logo=False)
    except ImportError as e:
        raise ImportError(
            "holoviews and bokeh are required for Sankey plots. "
            "Install them with: pip install holoviews bokeh"
        ) from e

    opts_common = dict(
        cmap=params.get("cmap", "Colorblind"),
        edge_line_width=params.get("edge_line_width", 0.5),
        node_padding=params.get("node_padding", 8),
        node_alpha=params.get("node_alpha", 1.0),
        node_width=params.get("node_width", 40),
        node_sort=params.get("node_sort", False),
        height=params.get("height", 900),
        bgcolor=params.get("bgcolor", "white"),
        edge_color=dim("source"),
        node_color=dim("depth"),
        show_values=False,
        label_position="outer",
    )

    # Two-panel path (exactly 3 species)
    if two_panel and len(ids) == 3:
        left, center, right = ids[0], ids[1], ids[2]
        e_lc = _prep_edges(M, left, center)
        e_cr = _prep_edges(M, center, right)
        center_order = sorted(
            set(e_lc["target"].tolist() + e_cr["source"].tolist())
        )
        n_lc = _nodes_from_edges(e_lc, left, center, center_order=center_order)
        n_cr = _nodes_from_edges(e_cr, center, right, center_order=center_order)
        s_lc = hv.Sankey(
            (e_lc, hv.Dataset(n_lc, kdims=["index"], vdims=["species", "depth"])),
            kdims=["source", "target"], vdims=["value"],
        ).opts(**{**opts_common, "width": params.get("width_each", 650)})
        s_cr = hv.Sankey(
            (e_cr, hv.Dataset(n_cr, kdims=["index"], vdims=["species", "depth"])),
            kdims=["source", "target"], vdims=["value"],
        ).opts(**{**opts_common, "width": params.get("width_each", 650)})
        obj = (s_lc + s_cr).cols(2)

    # Single Sankey (2 species, or 3+ without two_panel)
    else:
        rows = M.index.to_numpy(str)
        cols = M.columns.to_numpy(str)
        A = np.asarray(M.to_numpy(copy=True), dtype=float)
        A[~np.isfinite(A)] = 0.0
        A[A < align_thr] = 0.0
        r_idx, c_idx = A.nonzero()
        edges = pd.DataFrame({
            "source": rows[r_idx],
            "target": cols[c_idx],
            "value":  A[r_idx, c_idx],
        })
        if len(ids) > 2:
            sp_map = {sp: i for i, sp in enumerate(ids)}
            def _d(lbl): return sp_map.get(lbl.split('_', 1)[0], -1)
            d_src = edges["source"].map(_d).to_numpy()
            d_tgt = edges["target"].map(_d).to_numpy()
            keep = np.abs(d_src - d_tgt) == 1
            edges = edges.loc[keep].copy()
            swap = d_src[keep] > d_tgt[keep]
            if swap.any():
                s_ = edges.loc[swap, "source"].values
                t_ = edges.loc[swap, "target"].values
                edges.loc[swap, "source"] = t_
                edges.loc[swap, "target"] = s_
        edges = edges.groupby(["source", "target"], as_index=False, sort=False)["value"].sum()
        used = pd.Index(edges["source"]).union(pd.Index(edges["target"])).astype(str)
        node_df = pd.DataFrame({"index": used.values, "species": _species(used.values)})
        node_df["depth"] = (
            node_df["species"].map({ids[i]: i for i in range(len(ids))}).fillna(-1).astype(int)
        )
        node_df.sort_values(["depth", "index"], kind="mergesort", inplace=True, ignore_index=True)
        obj = hv.Sankey(
            (edges, hv.Dataset(node_df, kdims=["index"], vdims=["species", "depth"])),
            kdims=["source", "target"], vdims=["value"],
        ).opts(**{**opts_common, "width": params.get("width", 1200)})

    # Export
    if save_to:
        try:
            from holoviews import save as hv_save
            hv_save(obj, save_to, backend="bokeh")
        except Exception:
            try:
                from bokeh.io import save as bk_save
                from holoviews import render
                bk_save(render(obj, backend="bokeh"), filename=save_to)
            except Exception as e:
                raise RuntimeError(f"Failed to save HTML to '{save_to}': {e}") from e

    if show:
        from bokeh.io import show as bk_show
        from holoviews import render
        bk_show(render(obj, backend="bokeh"))

    return obj


# ---------------------------------------------------------------------------
# SAMap — UMAP scatter
# ---------------------------------------------------------------------------

def square_scatter(
    sm,
    figsize=(8, 8),
    dpi=600,
    COLORS=None,
    species_labels=None,
    legend=True,
    legend_title=None,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    legend_markersize=10,
    s=None,
    ss=None,
    axes=None,
    **kwargs,
):
    """Render a SAMap UMAP scatter on a square, publication-ready Axes.

    Wraps ``sm.scatter()``, enforces equal aspect ratio and a centred
    view window, strips any extra axes that SAMap creates internally,
    and optionally adds a clean legend.

    Args:
        sm: A ``samap.SAMAP`` instance.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(8, 8)``.
        dpi: Figure resolution. Defaults to 600.
        COLORS: ``{species_id: color}`` mapping for point and legend
            colours. When ``None`` SAMap's defaults apply.
        species_labels: ``{species_id: display_label}`` mapping used
            only in the legend. Falls back to ``species_id`` when
            ``None``.
        legend: Draw a legend when ``True``. Requires ``COLORS``.
            Defaults to ``True``.
        legend_title: Optional legend title string.
        legend_loc: Matplotlib legend location string.
            Defaults to ``"center left"``.
        legend_bbox: ``bbox_to_anchor`` tuple for the legend.
            Defaults to ``(1.02, 0.5)``.
        legend_markersize: Marker size in the legend. Defaults to 10.
        s: Uniform marker size for all species. Mutually exclusive with
            ``ss``; ignored when ``ss`` is provided.
        ss: ``{species_id: size}`` per-species marker sizes. Takes
            precedence over ``s``.
        axes: Existing ``matplotlib.Axes`` to draw onto. A new figure
            and axes are created when ``None``.
        **kwargs: Forwarded to ``sm.scatter()``. The keys
            ``legend_loc``, ``legend_loc_bounds``, and
            ``legend_fontsize`` are stripped before forwarding to avoid
            conflicts.

    Returns:
        A ``(fig, ax)`` tuple.
    """
    # Strip kwargs matplotlib / sm.scatter won't accept
    for bad in ("legend_loc", "legend_loc_bounds", "legend_fontsize"):
        kwargs.pop(bad, None)

    # Translate uniform size -> per-species dict
    if ss is None and s is not None:
        ss = {sid: float(s) for sid in sm.ids}

    # Axes setup
    if axes is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        ax = axes
        fig = ax.figure
        fig.set_size_inches(*figsize)
        fig.set_dpi(dpi)

    sm.scatter(axes=ax, COLORS=(COLORS or {}), ss=(ss or {}), **kwargs)

    # Remove any extra axes SAMap may have injected
    for extra in list(fig.axes):
        if extra is not ax:
            fig.delaxes(extra)

    # Centre and equalise the view window for a true square plot
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_position([0, 0, 1, 1])

    # Ensure fonts remain editable in Illustrator / PowerPoint exports
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Legend
    if legend and COLORS is not None:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                label=(species_labels or {}).get(sid, sid),
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=legend_markersize,
            )
            for sid, color in COLORS.items()
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            frameon=False,
        )

    return fig, ax


# ---------------------------------------------------------------------------
# SAMap — expression overlap plot
# ---------------------------------------------------------------------------

def save_expression_overlap_plot(
    sm,
    genes,
    colors,
    species_labels=None,
    overlap_color="#FFEE8C",
    figsize=(10, 10),
    dpi=600,
    output_prefix="expression_overlap",
    legend_title=None,
    markersize=10,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    margin=0.02,
    point_size=5,
    save_pdf=False,
    show=True,
):
    """Plot and save a SAMap expression-overlap UMAP.

    Wraps ``sm.plot_expression_overlap()``, enforces a square axis,
    adds a clean legend (including an overlap entry), and saves the
    result as PNG and optionally PDF.

    Args:
        sm: A ``samap.SAMAP`` instance.
        genes: ``{species_id: gene_name}`` dict passed directly to
            ``sm.plot_expression_overlap()``.
        colors: ``{species_id: color}`` dict for per-species point
            colours.
        species_labels: ``{species_id: display_label}`` mapping for the
            legend. Falls back to ``species_id`` when ``None``.
        overlap_color: Colour for cells co-expressing genes from both
            species. Defaults to ``"#FFEE8C"``.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(10, 10)``.
        dpi: Figure resolution. Defaults to 600.
        output_prefix: File path prefix (without extension) for saved
            outputs, e.g. ``"results/expression_overlap"``.
            Defaults to ``"expression_overlap"``.
        legend_title: Optional legend title string.
        markersize: Marker size in the legend. Defaults to 10.
        legend_loc: Matplotlib legend location string.
            Defaults to ``"center left"``.
        legend_bbox: ``bbox_to_anchor`` tuple for the legend.
            Defaults to ``(1.02, 0.5)``.
        margin: Fractional axis margin added after enforcing square
            limits. Defaults to 0.02.
        point_size: Uniform scatter point size forwarded to
            ``sm.plot_expression_overlap()`` for all species.
            Defaults to 5.
        save_pdf: Also save a PDF alongside the PNG when ``True``.
            Defaults to ``False``.
        show: Call ``plt.show()`` after saving when ``True``, otherwise
            close the figure. Defaults to ``True``.

    Returns:
        A ``(fig, ax)`` tuple.
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # SAMap creates its own figure internally; clear slate first
    plt.close("all")

    ss = {sid: point_size for sid in genes}
    sm.plot_expression_overlap(genes, COLORS=colors, COLORC=overlap_color, ss=ss)

    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(*figsize)

    # Enforce square plotting region
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    half = max(xmax - xmin, ymax - ymin) / 2.0
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(margin)

    # Legend — one entry per species plus the overlap entry
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label=(species_labels or {}).get(sid, sid),
            markerfacecolor=colors[sid],
            markeredgecolor="none",
            markersize=markersize,
        )
        for sid in genes
    ]
    handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label="Overlap",
            markerfacecolor=overlap_color,
            markeredgecolor="none",
            markersize=markersize,
        )
    )
    ax.legend(
        handles=handles,
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        frameon=False,
    )

    fig.savefig(f"{output_prefix}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# SAMap — joint UMAP with per-species label overlay
# ---------------------------------------------------------------------------

def plot_joint_umap(
    sm,
    species,
    label_col,
    embedding_key="X_umap_samap",
    figsize=(8, 8),
    dpi=300,
    s_other=6,
    s_target=60,
    alpha_other=0.6,
    other_color="lightgray",
    palette=None,
    save_to=None,
):
    """Plot a SAMap joint UMAP highlighting cell-type labels for one species.

    Renders two layers: background points from all other species in gray,
    and foreground points from ``species`` coloured by ``label_col``.

    Args:
        sm: A ``samap.SAMAP`` instance whose ``sams`` dict holds per-species
            SAM objects.
        species: Species key whose labels to highlight (e.g. ``"ml"``).
        label_col: Column in ``adata.obs`` containing the cell-type labels
            for ``species``.
        embedding_key: Key in ``.obsm`` for the joint UMAP coordinates.
            Defaults to ``"X_umap_samap"``.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(8, 8)``.
        dpi: Figure resolution. Defaults to 300.
        s_other: Marker size for background (non-target) cells.
            Defaults to 6.
        s_target: Marker size for the highlighted species' cells.
            Defaults to 60.
        alpha_other: Opacity for background cells. Defaults to 0.6.
        other_color: Colour for background cells. Defaults to
            ``"lightgray"``.
        palette: Optional list of hex/named colours for the target
            species' categories. Falls back to ``scanpy``'s
            ``default_20`` palette when ``None``.
        save_to: File path to save the figure (e.g. ``"umap.png"``).
            Skipped when ``None``.

    Returns:
        A ``(fig, ax)`` tuple.

    Raises:
        KeyError: If ``embedding_key`` is absent from any per-species
            ``.obsm``.
        ValueError: If ``label_col`` is not found in the target species'
            ``.obs``.
    """
    import scanpy as sc
    import anndata as ad

    adatas = {k: sm.sams[k].adata for k in sm.sams}

    # Validate inputs early
    target_adata = adatas[species]
    if label_col not in target_adata.obs.columns:
        raise ValueError(
            f"'{label_col}' not found in obs for species '{species}'. "
            f"Available columns: {list(target_adata.obs.columns)}"
        )
    for k, a in adatas.items():
        if embedding_key not in a.obsm:
            raise KeyError(
                f"Embedding key '{embedding_key}' not found in obsm for "
                f"species '{k}'. Available keys: {list(a.obsm.keys())}"
            )

    # Concatenate and carry the joint embedding onto the combined object
    combined = ad.concat(adatas, label="species", join="outer", index_unique=None)
    combined.obsm[embedding_key] = np.vstack(
        [adatas[k].obsm[embedding_key] for k in adatas]
    )

    # Build plot labels: target species gets real labels, everything else → "other"
    mask = combined.obs["species"].astype(str).eq(species)
    combined.obs["_plot_labels"] = np.where(
        mask,
        combined.obs[label_col].astype(str),
        "other",
    )
    combined.obs["_plot_labels"] = combined.obs["_plot_labels"].astype("category")

    cats = list(combined.obs["_plot_labels"].cat.categories)
    target_cats = [c for c in cats if c != "other"]
    _palette = palette or sc.pl.palettes.default_20
    color_map = {"other": other_color}
    color_map.update({c: _palette[i % len(_palette)] for i, c in enumerate(target_cats)})
    combined.uns["_plot_labels_colors"] = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Layer 1 — background (other species)
    sc.pl.embedding(
        combined,
        basis=embedding_key,
        color="_plot_labels",
        groups=["other"],
        s=s_other,
        alpha=alpha_other,
        frameon=False,
        legend_loc=None,
        show=False,
        ax=ax,
        na_color=other_color,
        na_in_legend=False,
    )

    # Layer 2 — foreground (target species, labelled)
    if target_cats:
        sc.pl.embedding(
            combined,
            basis=embedding_key,
            color="_plot_labels",
            groups=target_cats,
            s=s_target,
            frameon=False,
            legend_loc="right margin",
            show=False,
            ax=ax,
            na_color=other_color,
            na_in_legend=False,
        )

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if save_to:
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight")

    return fig, ax
