# Intelligent Parallelization Prediction

Using machine learning to **predict the best-performing CUDA launch configuration** for a kernel, instead of finding it by brute force.

## Why this exists

Every CUDA kernel has launch parameters — block size, grid dimensions, threads per block — that dramatically affect performance, and the best choice changes with the algorithm, the problem size, and the GPU it runs on. In practice developers find good values by trial and error, sweeping configurations and timing each one, which is slow and doesn't transfer to new problems or new hardware. This project asks a different question: **can a model learn the relationship between launch configuration and performance, and predict a good configuration directly?** If so, you can skip most of the manual tuning and also understand *which* parameters matter most and why.

## What it does

- Implements a set of CUDA kernels spanning very different computational profiles (memory-bound to compute-bound).
- Sweeps each kernel across problem sizes, block/grid/thread configurations, and GPUs to build a large labeled performance dataset (10K+ benchmark samples).
- Trains **XGBoost** models on that data to predict performance and recommend strong launch configurations.
- Uses **SHAP** to explain the models, surfacing which configuration parameters drive performance for each kernel.

## Workloads

Each kernel is a standalone CUDA program, parameterized so it can be swept across configurations:

| Workload      | Folder         | Profile |
|---------------|----------------|---------|
| Vector add    | `VecAdd`       | Memory-bound baseline |
| Convolution   | `Convolution`  | Mixed |
| Monte Carlo   | `MonteCarlo`   | Compute-bound |
| K-Means       | `KMeans`       | Iterative, data-dependent |
| DFT           | `DFT`          | Compute-bound |
| Matrix multiply | `matmul_final` | Compute-bound, tiling-sensitive |
| N-Body        | `nbody_final`  | Compute-bound |

## How it works

1. **Benchmark generation** — each `.cu` kernel is compiled with `nvcc` and run across a grid of problem sizes and launch configurations; driver scripts emit a CSV of (configuration → runtime/throughput) samples.
2. **Modeling** — the CSVs are aggregated in Jupyter notebooks, where XGBoost regressors are trained to map kernel + problem + launch configuration to performance.
3. **Explanation** — SHAP feature attributions show how block size, thread count, and problem size influence the prediction for each kernel.

## Running it

Built and run on NYU CIMS:

```bash
module load cuda-12.4          # load the CUDA toolkit
pip install xgboost shap       # for the analysis notebooks
```

Compile and run individual kernels, e.g.:

```bash
# Convolution
nvcc conv1.cu -o conv1 -lm
./conv1 <problem size> <kernel size> <blocks> <threadsPerBlock> <device>

# Vector addition
nvcc vecadd.cu -o vec
./vec <problem size> <device> <blocks> <threadsPerBlock>

# Monte Carlo
nvcc monte.cu -o monte
./monte <problem size> <device> <blocks> <threadsPerBlock>

# K-Means
nvcc pkmeans.cu -o pkm
./pkm <problem size> <clusters> <device> <blocks> <threadsPerBlock>

# Discrete Fourier Transform
nvcc dft.cu -o dft
./dft <problem size> <device> <blocks> <threadsPerBlock>

# Matrix multiply / N-Body (provided scripts)
./run_matmul.sh
./run_nbodyNew.sh
```

The sweep scripts generate per-workload CSVs (this takes a while). The data analysis and feature-importance work live in the Jupyter notebooks; running the analysis notebooks off the cluster is recommended so plots can render.

## Results

Full experiment design, dataset construction, model performance, and SHAP analysis are in **`Intelligent_Parallelization_Prediction.pdf`**.

## Tech stack

CUDA · C++ · Python · Jupyter · XGBoost · SHAP · NVIDIA GPUs · NYU CIMS

## Authors

Mihir Prajapati — NYU
Jackson Oleson - NYU
Yu Minematsu - NYU
