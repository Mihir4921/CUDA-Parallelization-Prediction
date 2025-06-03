#!/bin/bash

# Compiling the CUDA program
nvcc -o nbody_sim nbody.cu -lm

# Arrays for different values of numBodies and threadsPerBlock
numBodiesArray=(10240 20480 40960 81920 102400 204800 409600)
threadsPerBlockArray=(64 128 256 400 512 800 1024)
numBlocksArray=(1 2 4 8 16 32 64 128 256 450 512 800 1024 1500)

# Output header
echo "numBodies,threadsPerBlock,numBlocks,GPU_Time(s)"

# Loop over all combinations of numBodies and threadsPerBlock
for numBodies in "${numBodiesArray[@]}"
do
    for threadsPerBlock in "${threadsPerBlockArray[@]}"
    do
        for numBlocks in "${numBlocksArray[@]}"
        do
            # Calculate number of blocks
            # numBlocks=$(( (numBodies + threadsPerBlock - 1) / threadsPerBlock ))

            # Run the program and capture the output
            output=$(./nbody_sim $numBodies $threadsPerBlock $numBlocks)

            # Extract CPU and GPU times from the output
            # cpu_time=$(echo "$output" | grep "CPU sequential version:" -A 1 | tail -n 1 | awk '{print $4}')
            gpu_time=$(echo "$output" | grep "GPU version:" -A 1 | tail -n 1 | awk '{print $4}')

            # Print the results in CSV format
            echo "$numBodies,$threadsPerBlock,$numBlocks,$gpu_time"
        done
    done
done

