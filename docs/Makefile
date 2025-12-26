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

.PHONY: help prepare-docs html

prepare-docs:
	python $(SOURCEDIR)/generate_class_list.py --api-dir $(SOURCEDIR)/api
	python $(SOURCEDIR)/extract_section.py --readme $(SOURCEDIR)/../../README.md --build-dir $(BUILDDIR)
	python $(SOURCEDIR)/stencil_doc.py --input $(SOURCEDIR)/../../build/stencils.c --asm-pattern "$(SOURCEDIR)/../../build/tmp/runner-linux-*/stencils.asm" --output $(BUILDDIR)/stencils.md
	python $(SOURCEDIR)/example_asm.py --input $(SOURCEDIR)/../../tools/make_example.py --asm-pattern "$(SOURCEDIR)/../../build/tmp/runner-linux-*/example.asm" --output $(BUILDDIR)/compiled_example.md

# Build documentation (generate API and extract sections first)
html: prepare-docs
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
