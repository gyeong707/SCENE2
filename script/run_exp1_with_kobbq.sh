#!/bin/bash

TASKS=("kobbq")
MODELS=("exaone-7b")
DATA_SEEDS=(42)
MODEL_SEEDS=(42)

INPUT_DIR="./data"
OUTPUT_DIR="./results"
SAMPLING_COUNT=1
TEMPERATURE=0.0

for task in "${TASKS[@]}"; do
    
    if [ "$task" == "character" ]; then
        INPUT_FILE="character_specification_s3_augmented_250225_updated.csv"
    elif [ "$task" == "plot" ]; then
        INPUT_FILE="plot_development_s3_augmented_250225.csv"
    elif [ "$task" == "kobbq" ]; then
        INPUT_FILE="KoBBQ_all_samples_with_add_cols_only_amb_version_ab.csv"
    else
        echo "Unknown task: $task"
        continue
    fi

    echo "Task: $task | Input File: $INPUT_FILE"

    for model in "${MODELS[@]}"; do
        for d_seed in "${DATA_SEEDS[@]}"; do
            for m_seed in "${MODEL_SEEDS[@]}"; do
                
                echo "----------------------------------------------------------------"
                echo ":: Running: Model=$model | DataSeed=$d_seed | ModelSeed=$m_seed::"
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
                    --run_type full

                if [ $? -ne 0 ]; then
                    echo "Error occurred during execution."
                    exit 1 
                fi
                
                echo ""
            done
        done
    done
done

echo "All experiments completed."