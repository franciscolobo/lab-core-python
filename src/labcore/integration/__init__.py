"""
labcore.integration
-------------------
Tools for cross-species and multi-dataset integration analysis.

Currently provides SAMap plotting utilities. Additional integrators
(LIGER, scVI, etc.) can be added as new submodules.
"""

from .plotting import sankey_plot_3, square_scatter, plot_joint_umap, save_expression_overlap_plot
