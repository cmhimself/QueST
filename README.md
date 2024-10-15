# QueST: Querying Functional and Structural Niches on Spatial Transcriptomics Data via Contrastive Subgraph Emebedding

## Overview

The functional or structural spatial regions within tissues, referred to as spatial niches, are elements for illustrating the spatial contexts of multicellular organisms. A key challenge is querying shared niches across diverse tissues, which is crucial for achieving a comprehensive understanding of the organization and phenotypes of cell populations. Here we introduce QueST, a novel niche representation learning model designed for querying spatial niches across multiple samples. QueST utilizes a novel subgraph contrastive learning approach to explicitly capture niche-level characteristics and incorporates adversarial training to mitigate batch effects.

<div align="center">
    <figure>
        <img src="QueST_archetecture.png" width="900">
        <!-- <figcaption>QueST Model Architecture</figcaption> -->
    </figure>
</div>


## Getting started

### Environment

```
python                   3.9.19
numpy                    1.23.4
scanpy                   1.9.8
squidpy                  1.4.1
anndata                  0.10.6
scikit-learn             1.4.1
networkx                 3.2.1
torch                    2.2.1
torch-geometric          2.5.2
torch-scatter            2.1.2
torch-sparse             0.6.18
torch-spline-conv        1.2.2
graphcompass             0.2.2
```

### Niche Query Benchmark

- The adata objects with selected niche query problem are stored in `./bench/adata_query/` and are used as benchmarks for this study.
- We also offer the detailed generation process of these query niches in `run_niche_generator.py`. 

### Try the QueST Model

- Using QueST model to perform spatial niche query includes the following steps:

    1. Construct the spatial proximity graphs and preprocess the gene expression data;
    2. Train QueST model on the entire dataset;
    3. Computing cosine similarity between the query niche and the reference niches for query with the QueST's latent subgraph representation.

- For more details, we offer a demonstration of applying QueST to the DLPFC dataset in `demo_DLPFC.ipynb`.

## Data Availability

- For complete reference datasets and pretrained model weights, please refer to https://drive.google.com/drive/folders/1kQqReo7Groy4WMDjT2JNHzN6tXPSHrtf?usp=sharing
