#!/bin/bash

TASKS=("plot") # "("character" "plot")
MODELS=("gpt-5.1" "gpt-5.2") # ("gpt-5.1" "gpt-5.2" "exaone-7b" "exaone-32b" "llama-7b" "llama-70b" "qwen-8b" "qwen-32b" "gpt-oss-20b")
DATA_SEEDS=(42) #
MODEL_SEEDS=(42) #(42 43)
ROLES=("creative" "humorous") # ("helpful" "creative" "humorous")

INPUT_DIR="./data"
OUTPUT_DIR="./results"
SAMPLING_COUNT=10 # 10
TEMPERATURE=0.0

for task in "${TASKS[@]}"; do
    
    if [ "$task" == "character" ]; then
        INPUT_FILE="character_specification_s3_augmented_250225_updated.csv"
    elif [ "$task" == "plot" ]; then
        INPUT_FILE="plot_development_s3_augmented_250225.csv"
    else
        echo "Unknown task: $task"
        continue
    fi

    echo "Task: $task | Input File: $INPUT_FILE"

    for model in "${MODELS[@]}"; do
        for d_seed in "${DATA_SEEDS[@]}"; do
            for m_seed in "${MODEL_SEEDS[@]}"; do
                for role in "${ROLES[@]}"; do
                    echo "----------------------------------------------------------------"
                    echo ":: Running: Model=$model | DataSeed=$d_seed | ModelSeed=$m_seed | Role=$role (exp_type=role) ::"
                    echo "----------------------------------------------------------------"

                    python code/main.py \
                        --input_path "$INPUT_DIR" \
                        --input_file "$INPUT_FILE" \
                        --output_path "$OUTPUT_DIR" \
                        --task_type "$task" \
                        --model_name "$model" \
                        --dataset_seed "$d_seed" \
                        --model_seed "$m_seed" \
                        --sampling_count "$SAMPLING_COUNT" \
                        --temperature "$TEMPERATURE" \
                        --max_token 64 \
                        --run_type full \
                        --inference_type batch \
                        --exp_type role \
                        --system_role "$role"
                    fi
                done
            done
        done
    done
done

echo "All experiments completed."