# Sorting Algorithm Benchmark: Insertion, Merge, Heap, and Randomized Quicksort

## Description
This project implements and benchmarks four classic sorting algorithms in C++:
- **Insertion Sort**
- **Merge Sort**
- **Heapsort**
- **Randomized Quicksort**

The program generates random integer arrays of varying sizes, applies each sorting algorithm, and measures the runtime.  
The goal is to compare the practical performance of `O(n²)` vs. `O(n log n)` algorithms across different input sizes.

## Algorithms
- **Insertion Sort**  
  - Simple implementation, good for small arrays, but `O(n²)` runtime growth.
- **Merge Sort**  
  - Divide-and-conquer approach, stable, always `O(n log n)`.
- **Heapsort**  
  - In-place, guaranteed `O(n log n)`, slightly higher constant factors due to heap operations.
- **Randomized Quicksort**  
  - Average-case `O(n log n)`, but `O(n²)` in the worst case.  
  - Random pivot selection reduces the chance of worst-case performance.

## Input/Output:
Example output:
```
Enter the size of the array: 10000
insertion-sorting runtime: 151647
merge-sorting runtime: 1441
heapsorting runtime: 1573
randomized quicksort runtime: 1327
```

## Files
- `IMHR.cpp` – C++ implementation of the four sorting algorithms and benchmarking
- `report.pdf` – Analysis of runtime results, complexity discussion, and conclusions

## Requirements
- C++11 or higher

## Usage
1. Compile the program:
   ```
   g++ -std=c++11 -O2 -o sort_benchmark IMHR.cpp
   ```
2. Run the executable:
   ```
   ./sort_benchmark
   ```
3. Enter the desired array size (e.g., 10000). The program prints the runtime of each sorting algorithm for that array size.

## Analysis
- Insertion Sort: runtime grows ~100× when size grows 10× → confirms O(n²)
- Merge Sort: runtime grows ~10× when size grows 10× → confirms O(n log n)
- Heapsort: runtime grows slightly faster than Merge Sort due to more comparisons/swaps
- Randomized Quicksort: average-case close to Merge Sort, but variance due to pivot randomness

## Support
If you encounter issues or have questions:
- Open a GitHub Issue
- Contact: 40924seanshen@gmail.com

## Contributing
Contributions are welcome!

1. Fork this repository
2. Create a new branch: git checkout -b feature/my-feature
3. Commit your changes
4. Open a Pull Request

## Author
Sean Shen (沈昱翔)

## License
This project is licensed under the MIT License.

## Project Status
Completed – The four sorting algorithms were implemented, benchmarked, and analyzed with both experimental data and theoretical complexity.
