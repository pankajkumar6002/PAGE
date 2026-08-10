#!/usr/bin/env bash
# Build the ICLR 2027 paper inside a reproducible TeX Live container.
#
#   ./build.sh                 # build main.tex and iclr2027_conference.tex
#   ./build.sh main            # build just main.tex
#   ./build.sh clean           # remove latexmk intermediates (keeps PDFs)
#   ./build.sh shell           # interactive shell in the container
#
# The parent "paper/" directory is mounted, not just this one, because
# main.tex sets \graphicspath{{../}{./}} and pulls figures from ../figs/.
set -euo pipefail

IMAGE="${IMAGE:-page-iclr2027-latex}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$HERE")"
SUBDIR="$(basename "$HERE")"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is not installed or not on PATH." >&2
    echo "       Install it with:  sudo apt install docker.io" >&2
    echo "       then:             sudo usermod -aG docker \"\$USER\"  (re-login)" >&2
    echo "       Or build without Docker:  latexmk -pdf main.tex" >&2
    exit 127
fi

# `usermod -aG docker` only affects new login sessions, so the shell that ran it
# still can't reach the socket. If the user is a docker-group member on paper but
# not in this session's credentials, re-exec through `sg` instead of failing.
if ! docker info >/dev/null 2>&1; then
    if getent group docker | grep -qw "$(id -un)" && sg docker -c "docker info" >/dev/null 2>&1; then
        echo ">> note: this shell predates your 'docker' group membership; using 'sg docker'"
        echo "   log out and back in (or run 'newgrp docker') to make it permanent"
        exec sg docker -c "$(printf '%q ' "$0" "$@")"
    fi
    echo "error: cannot reach the Docker daemon at /var/run/docker.sock." >&2
    echo "       Daemon running?   systemctl is-active docker" >&2
    echo "       In the group?     sudo usermod -aG docker \"\$USER\"  (then re-login)" >&2
    exit 1
fi

# Rebuild the image only when the Dockerfile is newer than the existing image.
build_image() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo ">> building image $IMAGE (first run pulls TeX Live; ~330 MB image, a few minutes)"
        docker build -t "$IMAGE" "$HERE"
    fi
}

# -u keeps generated PDFs owned by the host user instead of root.
run() {
    docker run --rm \
        -u "$(id -u):$(id -g)" \
        -v "$PAPER_ROOT:/paper" \
        -w "/paper/$SUBDIR" \
        "$IMAGE" "$@"
}

latex() {
    echo ">> latexmk $1.tex"
    run latexmk -pdf -interaction=nonstopmode -halt-on-error "$1.tex"
}

build_image

case "${1:-all}" in
    all)
        latex main
        latex iclr2027_conference
        ;;
    clean)
        run latexmk -c main.tex
        run latexmk -c iclr2027_conference.tex
        ;;
    shell)
        docker run --rm -it \
            -u "$(id -u):$(id -g)" \
            -v "$PAPER_ROOT:/paper" \
            -w "/paper/$SUBDIR" \
            "$IMAGE" bash
        ;;
    *)
        latex "${1%.tex}"
        ;;
esac

echo ">> done"
ls -la "$HERE"/*.pdf 2>/dev/null || true
