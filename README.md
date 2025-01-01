# Distributed Matrix Multiplication using MPI

This project implements parallel matrix multiplication using MPI in Python, comparing performance between serial and distributed approaches.

## Features
- Distributed matrix multiplication across multiple processes
- Performance benchmarking (serial vs parallel)
- Scalability testing with different matrix sizes and process counts

## Requirements
- Python 3.7+
- mpi4py
- numpy
- matplotlib (for visualisation)

## Installation
```bash
pip install mpi4py numpy matplotlib
```

## Usage
Run benchmarks with different process counts:
```bash
mpiexec -n 2 python multiplier.py
mpiexec -n 4 python multiplier.py
mpiexec -n 8 python multiplier.py
```

Test across multiple nodes (requires hostfile)
```bash
mpiexec -n 8 --hostfile hostfile.txt python multiplier.py
```


## Project Structure
- `matrixMultiplication.py`: Core implementation class
- `multiplier.py`: Benchmark driver and visualization
- Generated files:
  - `benchmark_results_n{processes}.json`: Performance metrics
  - `performance_metrics.png`: Efficiency visualization

## Performance Analysis
The implementation measures:
- Serial execution time
- Parallel execution time
- Speedup (serial_time/parallel_time)
- Efficiency (speedup/number_of_processes)
- Communication overhead

[Benchmarking Report](./performance_report.md)

## Limitations
Performance degradation occurs with:
- Small matrix sizes (communication overhead dominates)
- Increasing process count (diminishing returns)
- Load imbalance in matrix distribution

## Optimization Guidelines
- Process Count Selection
  - Match available CPU cores
  - Use powers of 2
  - Monitor system resources

- Matrix Size Considerations
  - Large enough for parallelization benefits
  - Consider memory constraints
  - Account for CPU cache

- Network Considerations
  - Minimize communication overhead
  - Use appropriate network interface
  - Consider topology

