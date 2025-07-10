#!/usr/bin/env bash

docker run --rm -it \
    --gpus all \
    --name llm_fine_tuning \
    -v "$(pwd)":/workspace \
    --user dev \
    -w /workspace \
    llm_fine_tuning \
    bash
