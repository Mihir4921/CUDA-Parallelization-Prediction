OUTPUT_FILE="mtGT10m.csv"

echo "elements, blocks, threadsPerBlock, GPU" > $OUTPUT_FILE


elements=(10000000)
#blocks=(64)
#threads=(32)
#elements=(20000 42500 300000 750000 9000000 4700000 17500000 62500000 80000000 320000000)
blocks=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)
threads=(32 50 64 100 128 200 256 400 512 750 1024)

run_mt() {
	local n=$1
	local d=1
	local blocks=$2
	local threads=$3

	output=$(./monte $n $d $blocks $threads)

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
				run_mt $n $d $b $t
			done
		done
	done
done

echo "DONE"
