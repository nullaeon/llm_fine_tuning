#!/usr/bin/env bash

# CONFIG
IMAGE_NAME="llm_fine_tuning"
QUAY_USERNAME="nullaeon"
QUAY_REPO="llm_fine_tuning"
TAG="latest"

# Full image name for Quay
QUAY_IMAGE="quay.io/${QUAY_USERNAME}/${QUAY_REPO}:${TAG}"

# Build the image (optional — comment this out if already built)
echo "🔧 Building Docker image..."
docker build -t "${IMAGE_NAME}" .

# Tag it for Quay
echo "🔖 Tagging image as ${QUAY_IMAGE}"
docker tag "${IMAGE_NAME}" "${QUAY_IMAGE}"

# Push to Quay
echo "📤 Pushing to Quay.io..."
docker push "${QUAY_IMAGE}"

echo "✅ Done! Pushed ${QUAY_IMAGE}"
