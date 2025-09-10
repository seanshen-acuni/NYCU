# Optimal Binary Search Tree (OBST)

## Description
This project implements the **Optimal Binary Search Tree (OBST)** algorithm using dynamic programming in C++.  
Given:
- `p[i]`: probabilities of successful searches for keys
- `q[i]`: probabilities of unsuccessful searches between keys  

The algorithm computes:
- The minimum expected search cost  
- The optimal arrangement of keys as a binary search tree  
- A structured printout of the resulting OBST  

This program demonstrates how dynamic programming can optimize binary search tree construction to minimize average search time.

## Algorithm
- Initialize dynamic programming tables `e[i][j]`, `w[i][j]`, and `root[i][j]`.  
- Recursively calculate the cost of subtrees for all possible roots.  
- Select the root that minimizes cost, storing it in `root[i][j]`.  
- Reconstruct and print the tree structure using the `Print_Optimal_BST` function.
- **Time complexity**: O(n³) for n keys (classic DP implementation)
- **Space complexity**: O(n²) for DP tables

## Input/Output
Example Output:
```
p = 0.05 0.04 0.02 0.07 0.08 0.09 0.04 0.08 0.03
q = 0.08 0.06 0.04 0.04 0.03 0.06 0.07 0.06 0.04 0.02
Smallest search cost: 2.75
Root: 2
k5
  k2  
    k1    
      d0      
      d1      
    k4    
      k3     
        d2        
        d3        
      d4   
  k7
    k6
      d5 
      d6 
    k8
      d7
      k9
        d8   
        d9
```
## Files
- `OBST.cpp` – C++ implementation of the OBST algorithm
- `report.pdf` – Report with results, analysis, and discussion of the OBST example



## Requirements
- C++11 or higher

## Usage
1. Compile the program:
   ```
   g++ -std=c++11 -O2 -o obst OBST.cpp
   ```
2. Run the executable:
   ```
   ./obst
   ```
3. The program prints the probability arrays, minimum search cost, root index, and the structure of the optimal BST.

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
Completed – This project was developed as part of my **academic research practice** in algorithm design.
