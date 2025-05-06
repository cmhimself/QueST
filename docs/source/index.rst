Querying functional and structural niches on spatial transcriptomics data
===================================

.. note::

   This project is under active development.

Overview
--------

.. image:: QueST_archetecture.png
   :width: 700px
   :align: center

Cells in multicellular organisms coordinate within tissues to form spatial niches. While spatial transcriptomics enables gene expression profiling in spatial contexts, most existing methods remain cell-centric and fail to model niches as integrated analytical units. In this work, we defined the task of spatial niche querying, which aims to identify niches with similar structural or functional characteristics across samples, and developed QueST, a novel framework tailored for the niche query task. QueST models each niche as a subgraph, utilizes a novel subgraph contrastive learning to learn discriminative niche embeddings, and incorporates adversarial training to mitigate batch effects. Experiments showed that QueST accurately captured niche structure under heterogeneous environments on a simulation dataset, greatly outperformed state-of-the-art methods on systematic benchmarks, and generalized across sequencing platforms. Application to tertiary lymphoid structures in renal and lung cancer revealed both shared and cancer-specific niche patterns, demonstrating QueST’s ability to uncover biologically meaningful spatial organization. In summary, QueST provides a powerful framework for cross-sample niche comparison, facilitating deeper insights into the structural logic of tissue organization in health and disease.

Contents
--------

.. toctree::

   installation.md
   Tutorial 1 Querying niches on simulation data.ipynb
   Tutorial 2 Niche query benchmark on DLPFC dataset.ipynb
   Tutorial 3 Querying niches across different technologies.ipynb
   Tutorial 4 Analyze Tertiary Lymphoid Structures on ccRCC.ipynb
   API reference.md
