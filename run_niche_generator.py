import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import logging
import random
import warnings
from bench.generators import DlpfcGenerator, MouseOlfactoryBulbTissueGenerator

warnings.filterwarnings("ignore")
random.seed(2024)
np.random.seed(2024)
logging.basicConfig(level=logging.INFO)


def test_generator(bench, cell_type_list_list, num_cell_list, cell_type_key='cell_type', spot_size=5, figsize=(30, 12), min_single_ratio=None, min_sum_ratio=None, save_path=None):
    nrows = len(cell_type_list_list)
    ncols = len(num_cell_list) + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)  # figsize is (col, row)
    axs = axs.flatten()
    ax_ind = 0
    for cell_type_list in cell_type_list_list:
        sc.pl.spatial(bench.adata_q, color=cell_type_key, ax=axs[ax_ind], spot_size=spot_size, show=False) 
        axs[ax_ind].invert_yaxis()
        ax_ind += 1
        for num_cell in num_cell_list:
            print(f"***** testing {cell_type_list} {num_cell} *****")
            if min_single_ratio is None:
                bench.define_niche(num_cell=num_cell, cell_type_list=cell_type_list, sample_id=bench.sample_names[0])
                niche_name = f"{'_'.join(cell_type_list)}_{str(num_cell)}"
            else:
                bench.define_niche(num_cell=num_cell, cell_type_list=cell_type_list, sample_id=bench.sample_names[0], min_single_ratio=min_single_ratio, min_sum_ratio=min_sum_ratio)
                niche_name = f"{'_'.join(cell_type_list)}_{str(num_cell)}\n min-sum={min_sum_ratio}_min-single={min_single_ratio}"
            sc.pl.spatial(bench.adata_q, color=f'{niche_name}_cell_type', ax=axs[ax_ind], spot_size=spot_size, show=False)
            axs[ax_ind].invert_yaxis()
            ax_ind += 1
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def run_dlpfc_generator():
    adata_folder = "./data/DLPFC/adata_filtered"
    # sample_names = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]
    sample_names = np.array(["151507"])
    adata_list = [anndata.read_h5ad(f"{adata_folder}/{i}_filtered.h5ad") for i in sample_names]
    dlpfc = DlpfcGenerator(adata_list=adata_list, sample_names=sample_names, cell_type_key='cell_type')
    cell_type_list_list = [
        ['Layer4'],
        ['Layer5', 'Layer6'],
        ['Layer3', 'Layer4', 'Layer5']
    ]
    num_cell_list = [50, 100, 200]
    test_generator(dlpfc, cell_type_list_list, num_cell_list, figsize=(25, 12), save_path="./adata_query/dlpfc/niche.PNG")
    print(adata_list[0])
    adata_list[0].write_h5ad(filename=f"./adata_query/dlpfc/{sample_names[0]}.h5ad", compression="gzip")


def run_mobt_generator():
    adata_folder = './data/MouseOlfactoryBulbTissue/adata_relabeled'
    # sample_names = np.array(['slidev2_filtered', '10x',  'stereoseq'])
    # sample_names = np.array(['slidev2_filtered'])
    sample_names = np.array(['stereoseq'])
    print("reading data")
    adata_list = [anndata.read_h5ad(f"{adata_folder}/{sample_name}.h5ad") for sample_name in sample_names]
    mobt = MouseOlfactoryBulbTissueGenerator(adata_list=adata_list, sample_names=sample_names, cell_type_key='cell_type', spot_size=1)
    cell_type_list_list = [
        ['GCL'],
        ['GL', 'ONL'],
        ['GCL', 'MCL', 'EPL']
    ]
    num_cell_list = [50, 100, 150]
    # mobt.define_niche(cell_type_list=cell_type_list_list[0], num_cell=num_cell_list[0], sample_id=sample_names[0], min_single_ratio=0.4, min_sum_ratio=0.95)
    test_generator(mobt, cell_type_list_list, num_cell_list, cell_type_key=mobt.cell_type_key, spot_size=1, min_single_ratio=0.25, min_sum_ratio=0.95, save_path="./adata_query/MouseOlfactoryBulbTissue/niche.png")
    print(adata_list[0])
    adata_list[0].write_h5ad(filename=f"./adata_query/MouseOlfactoryBulbTissue/{sample_names[0]}.h5ad", compression="gzip")
