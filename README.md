# Querying niches on spatial transcriptomics data

## Overview

Cells in multicellular organisms coordinate to form functional niches. With spatial transcriptomics (ST) enabling gene expression profiling in spatial contexts, it has become increasingly evident that spatial niches serve as cohesive and recurrent units in physiological and pathological processes. Such recurrence suggests that tissue organization may be built from conserved multicellular patterns and motivates a query-based paradigm for systematically comparing and searching them across datasets. In this work, we defined the niche-query task as identifying spatial structures similar to a given niche of interest (NOI) across ST samples, and developed QueST, a specialized method for solving this task. QueST represents tissues as collections of local multicellular assemblages and learns assemblage representations through contrastive learning, while incorporating adversarial training to reduce sample-specific variation. In simulations and real benchmark datasets, QueST outperformed existing methods repurposed for niche querying, accurately retrieving corresponding spatial structures across heterogeneous tissue contexts and sequencing platforms. Applied to tertiary lymphoid structures (TLSs) in renal and lung cancers, QueST revealed distinct TLS states associated with patient prognosis and uncovered conserved and divergent spatial architectures across cancer types. These results demonstrate that QueST enables systematic, quantitative profiling of spatial niches across samples, providing a powerful tool to dissect spatial tissue architecture in health and disease.

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
