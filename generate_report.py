import json
import pandas as pd
from tabulate import tabulate

def generate_consolidated_report():
    # Load results for different process counts
    process_counts = [2, 4, 8]
    all_results = {}
    
    for n in process_counts:
        with open(f'benchmark_results_n{n}.json', 'r') as f:
            all_results[n] = json.load(f)

    # 1. Execution Time Table
    exec_data = []
    matrix_sizes = [50, 100, 500, 1000, 2000, 5000]
    
    for size in matrix_sizes:
        row = [size]
        # Get times from 4 processes results (index 1)
        for n in process_counts:
            result = next(r for r in all_results[n] if r['matrix_size'] == size)
            row.extend([
                f"{result['serial_time']:.4f}",
                f"{result['parallel_time']:.4f}",
                f"{result['speedup']:.2f}"
            ])
        exec_data.append(row)

    # 2. Scaling Efficiency Table
    eff_data = []
    for n in process_counts:
        result = next(r for r in all_results[n] if r['matrix_size'] == 2000)
        eff_data.append([n, f"{result['efficiency']*100:.2f}%"])

    # 3. Communication Overhead Table
    comm_data = []
    for size in matrix_sizes:
        result = next(r for r in all_results[4] if r['matrix_size'] == size)
        total_time = result['parallel_time']
        dist_time = result.get('distribution_time', 0)
        gather_time = result.get('gather_time', 0)
        comm_percentage = ((dist_time + gather_time) / total_time) * 100 if total_time > 0 else 0
        comm_data.append([
            size,
            f"{dist_time:.4f}",
            f"{gather_time:.4f}",
            f"{comm_percentage:.1f}%"
        ])

    # Generate report
    report = """# Performance Analysis Report

## 1. Execution Time
"""
    headers = ['Matrix Size'] + sum([[f'{n} Processes', 'Time', 'Speedup'] for n in process_counts], [])
    report += tabulate(exec_data, headers=headers, tablefmt='pipe')

    report += """

## 2. Scaling Efficiency (2000x2000 matrix)
"""
    report += tabulate(eff_data, headers=['Process Count', 'Efficiency'], tablefmt='pipe')

    report += """

## 3. Communication Overhead
"""
    report += tabulate(comm_data, 
                      headers=['Matrix Size', 'Distribution Time', 'Gathering Time', '% of Total Time'],
                      tablefmt='pipe')

    with open('performance_report.md', 'w') as f:
        f.write(report)

if __name__ == '__main__':
    generate_consolidated_report()