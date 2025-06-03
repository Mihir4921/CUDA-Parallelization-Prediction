#!/bin/bash

# Output file
OUTPUT_FILE="kmGT100k.csv"

# Create/overwrite the output file with headers
echo "elements, clusters, Iterations, blocks,threadsPerBlock,GPU" > $OUTPUT_FILE

# Arrays of values to test
#points_array=(10000 50000 100000 500000 1000000 5000000 10000000 50000000)
k_array=(10)
blocks_array=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)
threads_array=(32 50 64 100 128 200 256 400 512 750 1024)

points_array=(100000)
#k_array=(5 10 50 100 500)
#blocks_array=(1024 2048 4096 8192 16384 32768)
#threads_array=(128 256 512 750 1024)

# Function to run kmeans and extract execution time
run_kmeans() {
    local n=$1
    local k=$2
    local d=1
    local blocks=$3
    local threads=$4
    
    # Run kmeans and capture output
    output=$(./pkm $n $k $d $blocks $threads)
    
    # Extract execution time and iterations using grep and awk
    iters=$(echo "$output" | grep "iters taken:" | awk '{print $3}')
    time=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
    
    # Append results to CSV
    echo "$n,$k,$iters,$blocks,$threads,$time" >> $OUTPUT_FILE
    echo "Iterations: $iters, Time: $time"
}

# Main loop
total_runs=$((${#points_array[@]} * ${#k_array[@]} * ${#blocks_array[@]} * ${#threads_array[@]}))
current_run=0

echo "Starting performance testing..."
echo "Total configurations to test: $total_runs"

for n in "${points_array[@]}"; do
	for i in {1..10}; do
	    for k in "${k_array[@]}"; do
		for blocks in "${blocks_array[@]}"; do
		    for threads in "${threads_array[@]}"; do
			current_run=$((current_run + 1))
			echo "Running configuration $current_run/$total_runs: n=$n k=$k d=1 blocks=$blocks threads=$threads"
			run_kmeans $n $k $d $blocks $threads
		    done
		done
	    done
	done
done

echo "Testing complete. Results saved to $OUTPUT_FILE"

