from matrixMultiplication import MatrixMultiplier
import numpy as np
import time

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

def main():
    multiplier = MatrixMultiplier()
    A, B = multiplier.generate_matrices(50)

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