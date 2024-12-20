#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ========== Setup the benchmark tools ========== #

# Clone MultiPL-E repository
if [ ! -d "MultiPL-E" ]; then
    echo "Cloning MultiPL-E repository..."
    git clone https://github.com/nuprl/MultiPL-E.git
    cd MultiPL-E
    git checkout 19a25675e6df678945a6e3da0dca9473265b0055
    cd ..
fi

# ========== Setup the benchmark parameters ========== #

MODEL_DIR=$1
if [ -z "$MODEL_DIR" ]; then
    echo "Error: No model directory specified."
    exit 1
fi
echo $MODEL_DIR
# Identify the latest checkpoint directory
CHECKPOINT=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo $CHECKPOINT
# Validate that a checkpoint directory was found
if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint directories found in $MODEL_DIR"
    exit 1
fi

MODEL_LABEL=$(basename "$MODEL_DIR")

# Identify the tokenizer directory
TOKENIZER_DIR="$MODEL_DIR"

LANGUAGE="jl"
BENCHMARK_DATASET="humaneval-jl-reworded.jsonl"

BATCH_SIZE=8
MAX_TOKENS=1024
TEMPERATURE=0.2
COMPLETION_LIMIT=1

# Create output directory
RESULTS_DIR="./results/$MODEL_LABEL"
mkdir -p $RESULTS_DIR

OUTPUT_DIR="${RESULTS_DIR}/${LANGUAGE}_benchmark_temperature_${TEMPERATURE}"
mkdir -p $OUTPUT_DIR


# ========== Running model generation ========== #
echo "Running benchmark with the following parameters:"
echo "Model directory: $MODEL_DIR, Checkpoint: $CHECKPOINT, Tokenizer directory: $TOKENIZER_DIR, Benchmark dataset: $BENCHMARK_DATASET, Language: $LANGUAGE, Temperature: $TEMPERATURE, Batch size: $BATCH_SIZE, Completion limit: $COMPLETION_LIMIT, Max tokens: $MAX_TOKENS"

# Run the model generation script
echo "Running model generation script..."
python3 -u ./MultiPL-E/automodel.py \
        --name "$CHECKPOINT" \
        --tokenizer_name "$TOKENIZER_DIR" \
        --use-local \
        --dataset $BENCHMARK_DATASET \
        --temperature $TEMPERATURE \
        --batch-size $BATCH_SIZE \
        --completion-limit $COMPLETION_LIMIT \
        --output-dir-prefix $OUTPUT_DIR \
        --max-tokens $MAX_TOKENS | grep -v '^#'

echo "Model generation completed. Results are saved in $OUTPUT_DIR"


