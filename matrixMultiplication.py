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