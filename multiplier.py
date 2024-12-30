from matrixMultiplication import MatrixMultiplier
import numpy as np

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
    A, B = multiplier.generate_matrices(20)

    if multiplier.rank == 0:
        #print(np.array2string(A, precision=2, suppress_small=True))
        print_matrix("A", A)

        #print(np.array2string(B, precision=2, suppress_small=True))
        print_matrix("B", B)

        result, execution_time = multiplier.serial_multiply(A, B)
        print("\nResult Matrix (Serial Multiplication):")
        print_matrix("C", result)

        print("\nPerformance Metrics:")
        print(f"Serial Execution Time: {execution_time:.6f} seconds")
    
if __name__ == "__main__":
    main()