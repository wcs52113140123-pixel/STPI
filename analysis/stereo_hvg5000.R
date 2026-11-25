# Reproduces the 5,000-gene Stereo-seq matrices used at GW8-GW16.
#
# For each stage, the 5,000 most variable genes are selected over all bin50 units of the
# section (all annotated cell types) with Seurat's variance-stabilising transformation
# (FindVariableFeatures, selection.method = "vst"). The selected list is written to
# <out>/<stage>_Stereo_vst5000.txt and compared with the gene list of the stored matrix.
#
# Usage:  Rscript stereo_hvg5000.R <STPI_DATA> <out_dir>
# Inputs: <STPI_DATA>/gw6_annotated/expr/<stage>_Stereo_{counts.mtx,genes.txt,cells.txt}
#         stored lists: <STPI_DATA>/gaston_multi/<stage>/genes.txt (GW8-GW12),
#                       <STPI_DATA>/gaston_gw16/genes.txt (GW16)
# Checked with Seurat 5.3.0: identical gene sets at GW10, GW12 and GW16; 4,998 of 5,000
# at GW8 (two genes at the cut-off differ).
suppressMessages({library(Seurat); library(Matrix)})
args <- commandArgs(TRUE)
D <- if (length(args) >= 1) args[1] else Sys.getenv("STPI_DATA")
O <- if (length(args) >= 2) args[2] else "."
dir.create(O, showWarnings = FALSE, recursive = TRUE)
E <- file.path(D, "gw6_annotated", "expr")
res <- list()
for (st in c("GW8", "GW10", "GW12", "GW16")) {
  m <- readMM(file.path(E, sprintf("%s_Stereo_counts.mtx", st)))
  g <- readLines(file.path(E, sprintf("%s_Stereo_genes.txt", st)))
  cl <- readLines(file.path(E, sprintf("%s_Stereo_cells.txt", st)))
  if (nrow(m) != length(g)) m <- t(m)
  m <- as(m, "CsparseMatrix"); rownames(m) <- make.unique(g); colnames(m) <- make.unique(cl)
  o <- CreateSeuratObject(m, min.cells = 0, min.features = 0)
  o <- FindVariableFeatures(o, selection.method = "vst", nfeatures = 5000, verbose = FALSE)
  v <- VariableFeatures(o)
  writeLines(v, file.path(O, sprintf("%s_Stereo_vst5000.txt", st)))
  stored <- if (st == "GW16") file.path(D, "gaston_gw16", "genes.txt") else file.path(D, "gaston_multi", st, "genes.txt")
  T <- readLines(stored)
  res[[st]] <- data.frame(stage = st, units = ncol(o), genes = nrow(o), selected = length(v),
                          shared_with_stored = length(intersect(v, T)))
}
out <- do.call(rbind, res); print(out, row.names = FALSE)
write.csv(out, file.path(O, "stereo_vst5000_check.csv"), row.names = FALSE)
cat("Seurat", as.character(packageVersion("Seurat")), "\n")
