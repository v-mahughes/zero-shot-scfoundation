
import os
import logging
import warnings
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sc_foundation_evals import cell_embeddings, scgpt_forward, data, model_output
from sc_foundation_evals.helpers.custom_logging import log


log.setLevel(logging.INFO)
 
parser = argparse.ArgumentParser(description="scGPT zero shot evals")
parser.add_argument("--model_dir", type=str, required=True, help="Path to pretrained model. 3 files are expected: best_model.pt (model weights), args.json (model args), vocab.json (model vocab)")
parser.add_argument("--batch_size", type=int, required=True, help="Model batch size")
parser.add_argument("--output_dir", type=str, required=True, help="Path to where results will be saved")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to evaluation dataset")
parser.add_argument("--gene_col", type=str, required=True, help="Name of gene annotation column")
parser.add_argument("--batch_col", type=str, required=True, help="Name of column with batch annotations")
parser.add_argument("--label_cols", type=str, required=True, help="Comma-delineated list of label annotations (i.e. col-name1,col-name2,col-name3). If there is only one column, commas may be ommited.")
parser.add_argument("--layer_key", type=str, default='counts', help="layer where raw counts are stored")
parser.add_argument("--log_norm", type=bool, default=False, help="Set to True if the counts layer has been log normalized. Set to False if the data contains raw counts.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for multithreading")
parser.add_argument("--input_bins", type=int, default=51, help="Number of bins to use on input data")
parser.add_argument("--n_cells_embed_eval", type=int, default=1000, help="Number of cells to use in cell embedding evaluation")
parser.add_argument("--model_run", type=str, default='pretrained', help="Model type")
parser.add_argument("--seed", type=int, default=7, help="Random seed")
parser.add_argument("--n_hvg", type=int, default=1200, help="Number of highly variable genes")
parser.add_argument("--max_seq_len", type=int, default=1200, help="Maximum sequence of input is controlled by max_seq_len")
args = parser.parse_args()

model_out = os.path.join(args.output_dir, "model_outputs")
label_cols = [args.label_cols]


# create the model
scgpt_model = scgpt_forward.scGPT_instance(saved_model_path = args.model_dir,
                                           model_run = args.model_run,
                                           batch_size = args.batch_size, 
                                           save_dir = args.output_dir,
                                           num_workers = args.num_workers, 
                                           explicit_save_dir = True)

# create config
scgpt_model.create_configs(seed = args.seed, 
                           max_seq_len = args.max_seq_len, 
                           n_bins = args.input_bins)

# Loading the pretrained model. The log will show that some weights cannnot be loaded, as long as it is `cls_*` it's ok, as we are evaluating it in zero-shot setting, and those layers are not used.
scgpt_model.load_pretrained_model()

input_data = data.InputData(adata_dataset_path = args.dataset_path)

# To process the data we need the vocbulary. That we get from the model.
vocab_list = scgpt_model.vocab.get_stoi().keys()

# Preprocessing according to the steps as written in the scGPT repository. We will filter for the 1200 highly variable genes here.
input_data.preprocess_data(gene_vocab = vocab_list,
                           model_type = "scGPT",
                           gene_col = args.gene_col,
                           data_is_raw = not args.log_norm,
                           counts_layer = args.layer_key, 
                           n_bins = args.input_bins,
                           n_hvg = args.n_hvg)

#Tokenize the input data
scgpt_model.tokenize_data(data = input_data,
                          input_layer_key = "X_binned",
                          include_zero_genes = False)

# ## Evaluating model outputs

# First, we will perform forward pass on the model and extract embeddings.
scgpt_model.extract_embeddings(data = input_data)

# Next, we will specify what we want to evaluate in the output evaluations. Here, we will be using output of two pre-training objectives: masked language modelling (**MLM**), aka gene expression prediction (GEP), and **MVC** (not entirely sure what this abbreviation stands for), aka gene expression prediction from cell embedding (GEPC). 
eval_pred = model_output.GeneExprPredEval(scgpt_model,
                                         data = input_data,
                                         output_dir = model_out,
                                         embedding_key = ["mlm_output", 
                                                          "mvc_output"])
eval_pred.evaluate()

eval_pred.visualize(label_key = label_cols[-1])

# # Evaluate the cell embeddings
eval_ce = cell_embeddings.CellEmbeddingsEval(scgpt_model,
                                             data = input_data,
                                             output_dir = model_out,
                                             label_key = label_cols,
                                             batch_key = args.batch_col)

# with n_cells you can specify how much to subset the obs for
eval_ce.evaluate(n_cells = args.n_cells_embed_eval)

eval_ce.visualize()
