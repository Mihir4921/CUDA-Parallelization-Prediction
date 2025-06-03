OUTPUT_FILE="dftGT100k.csv"

echo "elements, blocks, threadsPerBlock, GPU" > $OUTPUT_FILE


elements=(100000)
#blocks=(64)
#threads=(32)
#elements=(10000 50000 100000 500000 1000000 5000000 10000000)
#elements=(5000 6500 42500 65000 175000 425000 650000)
blocks=(512 1024 2048 4096 8192 16384 32768)
threads=(32 50 64 100 128 200 256 400 512 750 1024)

run_dft() {
	local n=$1
	local d=1
	local blocks=$2
	local threads=$3

	output=$(./dft $n $d $blocks $threads)

	time=$(echo "$output" | grep "time taken:" | awk '{print $3}')

	echo "$n, $blocks, $threads, $time" >> $OUTPUT_FILE
	echo "Elements: $n, Time: $time"

}

echo "BEGIN SCRIPT"

for n in "${elements[@]}"; do
	for i in {1..20}; do
		for b in "${blocks[@]}"; do
			for t in "${threads[@]}"; do
				echo "Running config: elements=$n blocks=$b threads=$t"
				run_dft $n $d $b $t
			done
		done
	done
done

echo "DONE"
