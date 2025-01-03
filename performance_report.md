# Performance Analysis Report

## 1. Execution Time
|   Matrix Size |   2 Processes |   Time |   Speedup |   4 Processes |   Time |   Speedup |   8 Processes |   Time |   Speedup |
|--------------:|--------------:|-------:|----------:|--------------:|-------:|----------:|--------------:|-------:|----------:|
|            50 |        0.0001 | 0.0032 |      0.02 |        0.0001 | 0.0268 |      0    |        0.0001 | 0.2785 |      0    |
|           100 |        0.0083 | 0.0186 |      0.45 |        0.0089 | 0.0101 |      0.88 |        0.024  | 0.0372 |      0.65 |
|           500 |        0.0074 | 0.032  |      0.23 |        0.0069 | 0.1347 |      0.05 |        0.0383 | 0.3239 |      0.12 |
|          1000 |        0.0542 | 0.18   |      0.3  |        0.039  | 0.2711 |      0.14 |        0.0676 | 0.5063 |      0.13 |
|          2000 |        0.183  | 0.6034 |      0.3  |        0.202  | 0.7144 |      0.28 |        0.3006 | 1.1175 |      0.27 |
|          5000 |        2.2668 | 4.8795 |      0.46 |        2.4771 | 5.566  |      0.45 |        2.8009 | 6.8062 |      0.41 |

## 2. Scaling Efficiency (2000x2000 matrix)
|   Process Count | Efficiency   |
|----------------:|:-------------|
|               2 | 15.16%       |
|               4 | 7.07%        |
|               8 | 3.36%        |

## 3. Communication Overhead
|   Matrix Size |   Distribution Time |   Gathering Time | % of Total Time   |
|--------------:|--------------------:|-----------------:|:------------------|
|            50 |                   0 |           0      | 0.1%              |
|           100 |                   0 |           0.0001 | 0.8%              |
|           500 |                   0 |           0.0635 | 47.1%             |
|          1000 |                   0 |           0.0402 | 14.8%             |
|          2000 |                   0 |           0.051  | 7.1%              |
|          5000 |                   0 |           0.1044 | 1.9%              |