from mpi4py import MPI
import numpy as np
from typing import Tuple, Optional
import time

class MatrixMultiplier:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.metrics = {}

    def generate_matrices(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.rank == 0:
            A = np.random.rand(n,n)
            B = np.random.rand(n,n)
            return A, B
        return None, None

    def serial_multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        result = np.matmul(A, B)
        execution_time = time.time() - start_time
        self.metrics['serial_time'] = execution_time
        return result, execution_time

    def distribute_data(self, A: Optional[np.ndarray], B: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # First, broadcast the dimensions from root to all processes
        if self.rank == 0:
            matrix_shape = A.shape
        else:
            matrix_shape = None

        matrix_shape = self.comm.bcast(matrix_shape, root=0)
        
        if self.rank == 0:
            # Calculate rows per process
            rows_per_process = matrix_shape[0] // self.size
            remainder = matrix_shape[0] % self.size

            # Distribute rows of A
            send_counts = np.array([rows_per_process] * self.size)  # Convert to numpy array
            send_counts[-1] += remainder
            displacements = np.array([0] + [rows_per_process] * (self.size-1))  # Convert to numpy array
            displacements = np.cumsum(displacements) #this has the indices of the rows each process will work on

            print(f"Debug - send_counts: {send_counts}")
            print(f"Debug - displacements: {displacements}")
        
            self.metrics['distribution_pattern'] = {
                'rows_per_process': rows_per_process,
                'remainder': remainder,
                'send_counts': send_counts.tolist(),
                'displacements': displacements.tolist()
            }
        else:
            send_counts = None
            displacements = None

        # Broadcast matrix B to all processes
        B = self.comm.bcast(B, root=0)

        # Broadcast send_counts to all processes
        send_counts = self.comm.bcast(send_counts, root=0)

        # Calculate local size for each process
        my_rows = send_counts[self.rank]

        # Create empty array for local data
        local_A = np.empty((my_rows, matrix_shape[1]), dtype=np.float64)

        # Scatter the data
        self.comm.Scatterv([A, 
                        send_counts * matrix_shape[1],  # elements per process
                        displacements * matrix_shape[1] if self.rank == 0 else None,  # displacement in elements
                        MPI.DOUBLE], 
                        local_A)  # receive buffer

        return local_A, B