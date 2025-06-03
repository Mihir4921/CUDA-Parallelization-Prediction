#!/bin/bash

# Compiling the CUDA program
nvcc -o matrix_mul matmulArg.cu -lm

# Output CSV file
# output_file="gpu_performance.csv"

# Create the CSV header
# echo "N,TILE_WIDTH,Threads_per_block,Blocks,GPU_Time" > "$output_file"
echo "N,TILE_WIDTH,Threads_per_block,Blocks,GPU_Time"
# Define the range of N and TILE_WIDTH values
N_values=(100000)         # Array of different N values to test
TILE_WIDTH_values=(8 16 24 32 50 64 90 128 150 192)  # Array of different TILE_WIDTH values to test

num_runs=20

# Run matrix multiplication for each combination of N and TILE_WIDTH
for N in "${N_values[@]}"; do
    for TILE_WIDTH in "${TILE_WIDTH_values[@]}"; do
        for run in $(seq 1 $num_runs); do
            # echo "Running matrix multiplication with N=$N and TILE_WIDTH=$TILE_WIDTH..."
            
            # Execute the program and capture its output
            output=$(./matrix_mul "$N" "$TILE_WIDTH")
            
            # Extract values from the output using grep and awk
            gpu_time=$(echo "$output" | grep -i "Time taken =" | awk '{print $4}')
            threads_per_block=$(echo "$output" | grep -i "Block size (threads per block):" | awk '{print $6}')
            blocks=$(echo "$output" | grep -i "Grid size (number of blocks):" | awk '{print $6}')

            # Append results to the CSV file
            # echo "$N,$TILE_WIDTH,$threads_per_block,$blocks,$gpu_time" >> "$output_file"
            echo "$N,$TILE_WIDTH,$threads_per_block,$blocks,$gpu_time"
        done
    done
done

# echo "Results saved to $output_file"
