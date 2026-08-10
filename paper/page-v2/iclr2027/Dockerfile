# Toolchain-only image for building the PAGE ICLR 2027 paper.
# The paper sources are NOT copied in; they are bind-mounted at run time
# (see build.sh), so edits on the host take effect without a rebuild.
FROM debian:bookworm-slim

# texlive-latex-extra carries multirow; -recommended carries times/psnfss,
# booktabs and xspace; -science carries the amsthm/amssymb extras.
RUN apt-get update && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-science \
        texlive-plain-generic \
        texlive-bibtex-extra \
        make \
    && rm -rf /var/lib/apt/lists/*

# The container runs as the invoking host UID (build.sh passes -u), which has no
# home directory inside the image. Point TeX's writable caches somewhere world-
# writable or pdflatex fails before it reads a single .tex file.
ENV HOME=/tmp \
    TEXMFVAR=/tmp/texmf-var \
    TEXMFCONFIG=/tmp/texmf-config \
    TEXMFHOME=/tmp/texmf

WORKDIR /paper/iclr-2027

# Default: build both the paper and the upstream template shell.
CMD ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
