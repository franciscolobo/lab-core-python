"""
Curated gene sets used across scRNA-seq analyses.

Currently provides human cell-cycle gene lists (S and G2/M phase),
sourced from the standard Seurat/Regev cell-cycle gene set. Additional
species can be added by running `get_orthologs()` (see
`labcore.scrnaseq.orthology`) against these human lists and caching
the result here, rather than recomputing per notebook.
"""

CELL_CYCLE_GENES_HUMAN = {
    "s_genes": [
        "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
        "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
        "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
        "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
        "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
        "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
    ],
    "g2m_genes": [
        "HMGB2", "CDK1", "HN1", "CDC20", "TOP2A", "NDC80", "CKS2", "NCL",
        "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4",
        "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
        "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "JPT1",
        "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2",
        "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN",
        "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
        "CBX5", "CENPA",
    ],
}
"""dict: Human S-phase and G2/M-phase marker genes.

Keys: ``"s_genes"``, ``"g2m_genes"``.

Example:
    >>> from labcore.scrnaseq.genesets import CELL_CYCLE_GENES_HUMAN
    >>> adata = score_cell_cycle(
    ...     adata,
    ...     s_genes=CELL_CYCLE_GENES_HUMAN["s_genes"],
    ...     g2m_genes=CELL_CYCLE_GENES_HUMAN["g2m_genes"],
    ... )
"""
