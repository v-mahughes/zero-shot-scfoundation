#!/bin/bash/
#EXAMPLE SCGPT EVALUATION RUN
python -u scGPT_zero_shot.py --model_dir "../data/weights/scgpt/scGPT_human" --batch_size 32 --output_dir "../output/scgpt/scgpt_human/" --dataset_path "../data/datasets/pancreas_scib.h5ad" --gene_col "gene_symbols" --batch_col "tech" --label_cols "celltype"