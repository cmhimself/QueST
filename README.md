# Querying functional and structural niches on spatial transcriptomics data

## Overview

Cells in multicellular organisms coordinate to form structural and functional niches. With spatial transcriptomics (ST) enabling gene expression profiling in spatial contexts, it has been revealed that spatial niches serve as cohesive and recurrent units in physiological and pathological processes. These observations suggest universal tissue organization principles encoded by conserved niche patterns, and call for a query-based niche analytical paradigm beyond current computational tools. In this work, we defined the niche-query task, which is to identify similar niches across ST samples given a niche of interest (NOI). We further developed QueST, a specialized method for solving this task. QueST models each niche as a subgraph, uses contrastive learning to learn discriminative niche embeddings, and incorporates adversarial training to mitigate batch effects. In simulations and benchmark datasets, QueST outperformed existing methods repurposed for niche querying, accurately capturing niche structures in heterogeneous environments and demonstrating strong generalizability across diverse sequencing platforms. Applied to tertiary lymphoid structures in renal and lung cancers, QueST revealed functionally distinct niches associated with patient prognosis and uncovered conserved and divergent spatial architectures across cancer types. Applied to a combinatorial spatial perturbation dataset, QueST demonstrated a complete de novo discovery-oriented workflow, characterizing previously unresolved tumor nodules through querying. These results demonstrate that QueST enables systematic, quantitative profiling of spatial niches across samples, providing a powerful tool to dissect spatial tissue architecture in health and disease.

<div align="center">
    <figure>
        <img src="https://github.com/cmhimself/QueST/raw/main/docs/source/QueST_architecture.png" width="900">
        <!-- <figcaption>QueST Model Architecture</figcaption> -->
    </figure>
</div>


## Getting started

### Installation

We recommend using a conda environment

```bash
conda create -n quest python==3.9.19
```

Install necessary dependencies first before the installation of QueST

```bash
conda activate quest
pip install -r requirements.txt
```

Finally, QueST is available on PyPI and can be installed via 

```bash
pip install quest-niche
```

### Usage

See detailed usage on [Read the Docs](https://quest-niche.readthedocs.io/en/latest/index.html) website.

## Data Availability

- For reference datasets, please refer to https://cloud.tsinghua.edu.cn/d/dbf5c914da064eedbb58/
