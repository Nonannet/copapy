#!/usr/bin/env bash
set -euo pipefail

# Check if this run was triggered by a tag
if [[ "${GITHUB_REF:-}" == refs/tags/* ]]; then
    TAG_NAME="${GITHUB_REF#refs/tags/}"
    echo "Detected tag push: $TAG_NAME"
    # For GitHub Actions output
    echo "version=$TAG_NAME" >> "${GITHUB_OUTPUT:-/dev/stdout}"
    exit 0
fi

# Otherwise, extract version from pyproject.toml
if [[ -f pyproject.toml ]]; then
    VERSION=$(grep -E '^version\s*=' pyproject.toml \
              | sed -E 's/version\s*=\s*"([^"]+)"/\1/' \
              | tr -d '\r\n')
    if [[ -z "$VERSION" ]]; then
        echo "! Could not find version in pyproject.toml" >&2
        exit 1
    fi
    echo "Detected version from pyproject.toml: v$VERSION-beta"
    echo "version=v$VERSION-beta" >> "${GITHUB_OUTPUT:-/dev/stdout}"
else
    echo "! pyproject.toml not found" >&2
    exit 1
fi
