#!/bin/bash

OUTPUT_FILE="convBenchmark5.csv"
echo "image_size, kernel_size, blocks, threadsPerBlock, mode, time(ms)" > $OUTPUT_FILE

# Parameters
image_sizes=(4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)    # Experiment with medium to large images
kernel_sizes=(3 5 7)               # Common kernel sizes
blocks=(1 2 4 8 16 32 64 128 256 512 1024)              # Use a mix of small to medium blocks
threads=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)         # Utilize up to the max threads per block

run_conv() {
    local image_size=$1
    local kernel_size=$2
    local blocks=$3
    local threads=$4
    local use_gpu=$5

    # Execute the compiled program
    output=$(./conv1 "$image_size" "$kernel_size" "$threads" "$blocks" "$use_gpu")

    if [ "$use_gpu" -eq 1 ]; then
        # GPU Mode: Extract "Parallel GPU Convolution Time"
        time=$(echo "$output" | grep "GPU " | sed -E 's/.*Time: ([0-9.]+) ms/\1/')
        mode="GPU"
    else
        # CPU Mode: Extract "Sequential CPU Convolution Time"
        time=$(echo "$output" | grep "CPU " | sed -E 's/.*Time: ([0-9.]+) ms/\1/')
        mode="CPU"
    fi

    # Check if time was extracted successfully
    if [ -z "$time" ]; then
        time="N/A"
        echo "Warning: Failed to extract time from output for mode=$mode" >&2
        echo "Output was: $output" >&2
    fi

    # Append result to the CSV
    echo "$image_size, $kernel_size, $blocks, $threads, $mode, $time" >> $OUTPUT_FILE
    echo "Run: image_size=$image_size kernel_size=$kernel_size blocks=$blocks threads=$threads mode=$mode Time=$time ms"
}

echo "BEGIN BENCHMARKING"

for image_size in "${image_sizes[@]}"; do
    for kernel_size in "${kernel_sizes[@]}"; do
        # CPU Benchma
            run_conv $image_size $kernel_size 0 0 0

            # GPU Benchmarks
            for b in "${blocks[@]}"; do
                for t in "${threads[@]}"; do

                    run_conv $image_size $kernel_size $b $t 1
		    
		done	
	    done
        
    done
done

echo "BENCHMARKING COMPLETE"