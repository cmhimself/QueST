# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile sync-tutorials

# Sync tutorial notebooks from top-level tutorial/ into source/ before build.
# Tutorials are stored once at ../tutorial/ to keep a single source of truth;
# this rule mirrors them into source/ so toctree references resolve.
sync-tutorials:
	@cp ../tutorial/*.ipynb "$(SOURCEDIR)/"

# Build targets that need tutorial notebooks present in source/ depend on
# sync-tutorials. Other targets (clean, help, ...) skip the sync.
html dirhtml singlehtml epub latex latexpdf: sync-tutorials

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
