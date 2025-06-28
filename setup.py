from setuptools import setup

setup(
    name="quest-niche",  # PyPI distribution name
    version="0.1.0",
    description="Querying functional and structural niches on spatial transcriptomics data",
    author="Mo Chen",
    python_requires=">=3.8,<4.0",

    # use models/ folder as the "quest" package when using "import quest"
    package_dir={"quest": "models"},
    packages=["quest"],

    # only a demonstration as these should've been installed after "pip install -r requirements.txt"
    install_requires=[
        "numpy==1.23.4",
        "pandas==2.2.1",
        "scipy==1.9.1",
        "scanpy==1.9.8",
        "squidpy==1.4.1",
        "networkx==3.2.1",
        "scikit-learn==1.4.1.post1",
        "umap-learn==0.5.5",
        "anndata==0.10.6",
        "leidenalg==0.10.2",
        "igraph==0.11.4",
        "graphcompass==0.2.2",
        "torch-geometric==2.5.2"
    ],
)
