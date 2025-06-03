Readme.txt

Project Code - 
1. Log in to cims
2. Load cuda module by calling 'module load cuda-12.4'
3. compile all programs using nvcc
4. Run scripts using ./script_name
5. Scripts will generate csv file with data for their specific problem(takes a long long time)
6. May need to install xgboost and shap using 'pip install shap xgboost' (recommended to use not on CIMS because visuals cannot be displayed.) 
7. Data analysis provided in jupyter notebooks, will need to install SHAP feature explanations if you would like to run selected cells that perform feature analysis.


For convolution, 
nvcc conv1.cu -o conv1 -lm
Usage: ./conv1 <problem size> <kernel size> <blocks> <threadsPerBlock> <device> 

For vector addition:
nvcc vecadd.cu -o vec
./vec <problem size> <device> <blocks> <threadsPerBlock>

For Monte Carlo:
nvcc monte.cu -o monte
./monte <problem size> <device> <blocks> <threadsPerBlock>

For KMeans:
nvcc pkmeans.cu -o pkm
Usage: ./pkm <problem size> <clusters> <device> <blocks> <threadsPerBlock>

For Discrete Fourier Transform:
nvcc dft.cu -o dft
Usage: ./dft <problem size> <device> <blocks> <threadsPerBlock>

For Mat Mul:
./run_matmul.sh

For N-Body Problem: 
./run_nbodyNew.sh