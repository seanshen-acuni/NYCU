# Rod Cutting Problem – Top-Down vs. Bottom-Up Dynamic Programming

## Description
This project implements the **Rod Cutting Problem** in C++ using two dynamic programming approaches:
- **Top-Down (Memoized Recursion)**
- **Bottom-Up (Iterative DP)**

The program computes:
- The **maximum** and **minimum** revenue obtainable by cutting a rod of length `n`
- The **cutting positions** corresponding to those revenues
- A comparison of runtime between the two approaches

This project demonstrates how different DP strategies solve the same optimization problem with trade-offs in clarity, recursion overhead, and memory usage.

## Algorithm
- **Top-Down**  
  - Uses recursion with memoization (`r1`, `r2`) to avoid recomputation of subproblems.  
  - Intuitive, only computes subproblems that are needed.  
  - Requires additional memory for memoization and incurs recursive call overhead.

- **Bottom-Up**  
  - Iteratively builds solutions from the smallest subproblems up to length `n`.  
  - More efficient in space, no recursion required.  
  - Computes all subproblems even if some are not needed.

- Both versions also track **cutting solutions** (`s1`, `s2`) to reconstruct the positions for max/min revenue.  

## Input/Output
### Example input:
```
Enter the length of the rod: 10
Please enter the prices for each length(1~10):
1 5 8 9 10 17 17 20 24 30
```
### Example output:
```
Top-Down results:
Maximum Revenue: 30
Cutting Positions(max): 10 
cut into 1 parts
Minimum Revenue: 10
Cutting Positions(min): 1 1 1 1 1 1 1 1 1 1 
cut into 10 parts
compile time(top-down): 12
Bottom-up results:
Maximum Revenue: 30
Cutting Positions(max): 10 
cut into 1 parts
Minimum Revenue: 10
Cutting Positions(min): 1 1 1 1 1 1 1 1 1 1 
cut into 10 parts
compile time(bottom-up): 6
```

## Files
- `rod_cutting.cpp` – C++ implementation of both Top-Down and Bottom-Up algorithms
- `report.pdf` – Report discussing the algorithms, differences, and experimental results

## Requirements
- C++11 or higher

## Usage
1. Compile the program:
   ```
   g++ -std=c++11 -O2 -o rod_cutting rod_cutting.cpp
   ```
2. Run the executable:
   ```
   ./rod_cutting
   ```

## Analysis
- Top-Down is easier to implement and only solves needed subproblems, but recursion adds overhead.
- Bottom-Up avoids recursion and is faster, but solves all subproblems even if unnecessary.
- Both yield the same optimal results, differing only in efficiency.

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
Completed – Implemented and compared Top-Down and Bottom-Up dynamic programming for the Rod Cutting Problem.
