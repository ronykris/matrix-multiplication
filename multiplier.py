from matrixMultiplication import MatrixMultiplier
import numpy as np
import time
import json
import matplotlib.pyplot as plt

def print_matrix(name: str, matrix: np.ndarray):
    """Print matrix in traditional mathematical format"""
    n = len(matrix)
    width = 6
    total_width = width * n
    
    print(f"\nMatrix {name}:")
    print("┌" + " " * total_width + " " + "┐")
    
    for row in matrix:
        print("│", end="")
        for elem in row:
            print(f"{elem:6.2f}", end="")
        print(" " + "│")
    
    print("└" + " " * total_width + " " + "┘")

def plot_performance_metrics(results_list, process_counts):
   matrix_sizes = sorted(list(set([r['matrix_size'] for r in results_list[0]])))
   efficiencies = []
   
   for results in results_list:
       eff = [r['efficiency'] * 100 for r in results]  # Convert to percentage
       efficiencies.append(eff)

   plt.figure(figsize=(12, 6))
   
   # Plot efficiency vs matrix size for each process count
   for i, n_proc in enumerate(process_counts):
       plt.plot(matrix_sizes, efficiencies[i], marker='o', label=f'{n_proc} processes')

   plt.xlabel('Matrix Size')
   plt.ylabel('Efficiency (%)')
   plt.title('Matrix Multiplication Efficiency vs Matrix Size')
   plt.grid(True)
   plt.legend()
   plt.xscale('log')
   plt.savefig('performance_metrics.png')
   plt.close()

def run_benchmark(n: int, multiplier: MatrixMultiplier) -> dict:
    A, B = multiplier.generate_matrices(n)
    metrics = {}
    
    if multiplier.rank == 0:
        _, serial_time = multiplier.serial_multiply(A, B)
        metrics['serial_time'] = serial_time

    parallel_start = time.time()
    local_A, local_B = multiplier.distribute_data(A, B)
    local_result = multiplier.parallel_multiply(local_A, local_B)
    
    if multiplier.rank == 0:
        matrix_shape = A.shape
    else:
        matrix_shape = None
    matrix_shape = multiplier.comm.bcast(matrix_shape, root=0)
    
    final_result = multiplier.gather_results(local_result, matrix_shape)
    parallel_time = time.time() - parallel_start
    all_parallel_times = multiplier.comm.gather(parallel_time, root=0)
    
    if multiplier.rank == 0:
        max_parallel_time = max(all_parallel_times)
        speedup = metrics['serial_time']/max_parallel_time
        efficiency = speedup/multiplier.size
        
        metrics.update({
            'matrix_size': n,
            'num_processes': multiplier.size,
            'parallel_time': max_parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'compute_time': multiplier.metrics['parallel_compute_time'],
            'gather_time': multiplier.metrics['gather_time']
        })
    return metrics


def main():
    multiplier = MatrixMultiplier()
    matrix_sizes = [50, 100, 500, 1000, 2000, 5000]
    results = []
    
    for size in matrix_sizes:
        metrics = run_benchmark(size, multiplier)
        if multiplier.rank == 0:
            results.append(metrics)
            print(f"\nResults for {size}x{size} matrix:")
            print(f"Processes: {metrics['num_processes']}")
            print(f"Serial time: {metrics['serial_time']:.4f}s")
            print(f"Parallel time: {metrics['parallel_time']:.4f}s")
            print(f"Speedup: {metrics['speedup']:.2f}x")
            print(f"Efficiency: {metrics['efficiency']:.2%}")
    
    if multiplier.rank == 0:
        with open(f'benchmark_results_n{str(multiplier.size)}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    #    # Load results from all process counts and plot
    #    process_counts = [2, 4, 8]
    #    all_results = []
    #    for n in process_counts:
    #        with open(f'benchmark_results_n{n}.json', 'r') as f:
    #            all_results.append(json.load(f))
       
    #    plot_performance_metrics(all_results, process_counts)



def test():
    multiplier = MatrixMultiplier()
    A, B = multiplier.generate_matrices(5000)

    if multiplier.rank == 0:
        print(f"\nOriginal matrices (Rank {multiplier.rank}):")
        print_matrix("A", A)
        print_matrix("B", B)

        # Perform serial multiplication
        print("\nPerforming serial multiplication...")
        serial_result, serial_time = multiplier.serial_multiply(A, B)
        print(f"Serial multiplication time: {serial_time:.6f} seconds")
        print_matrix("Serial Result", serial_result)

    # Start timing parallel implementation
    parallel_start = time.time()

    # Distribute the data
    local_A, local_B = multiplier.distribute_data(A, B)
    
    # Each process prints its portion
    print(f"\nProcess {multiplier.rank} received:")
    print(f"Local A shape: {local_A.shape}")
    print_matrix(f"Local A from rank {multiplier.rank}", local_A)
    #print_matrix(f"B in rank {multiplier.rank}", local_B)

     # Perform local multiplication
    local_result = multiplier.parallel_multiply(local_A, local_B)
    print(f"\nProcess {multiplier.rank} local multiplication result:")
    print_matrix(f"Local result from rank {multiplier.rank}", local_result)
    
    # Gather all results
    if multiplier.rank == 0:
        matrix_shape = A.shape
    else:
        matrix_shape = None

    matrix_shape = multiplier.comm.bcast(matrix_shape, root=0)    
    final_result = multiplier.gather_results(local_result, matrix_shape)    
    parallel_time = time.time() - parallel_start

    # Gather final timing information
    all_parallel_times = multiplier.comm.gather(parallel_time, root=0)
    

    # Print final results from rank 0
    if multiplier.rank == 0:
        print("\nFinal Result:")
        print_matrix("C = A × B", final_result)
        
        # Verify result
        is_correct = np.allclose(final_result, serial_result)
        print(f"\nResult is correct: {is_correct}")
        
        max_parallel_time = max(all_parallel_times)
        speedup = serial_time/max_parallel_time
        efficiency = speedup/multiplier.size  # Efficiency = Speedup/Number of Processes
        

        # Print timing comparison
        print("\nPerformance Metrics:")
        print(f"Serial time: {serial_time:.6f} seconds")
        print(f"Parallel time: {parallel_time:.6f} seconds")
        print(f"Speedup: {speedup:.4f}x")
        print(f"Efficiency: {efficiency:.2%}")
        print(f"Parallel computation time: {multiplier.metrics['parallel_compute_time']:.6f} seconds")
        print(f"Gather time: {multiplier.metrics['gather_time']:.6f} seconds")
   
if __name__ == "__main__":
    main()