# Maze Solver with BFS

## Description
This project implements a **Breadth-First Search (BFS)** based maze solver in C++.  
It reads multiple 17×17 mazes from an input file, computes the shortest path from the start `(1,1)` to the goal `(17,17)`, and outputs the step count along with the path coordinates.

The solver supports:
- Different cell types (walkable cells, walls, weighted cells).
- Path reconstruction with predecessor tracking.
- Handling multiple maze patterns in sequence.

## Algorithm
- Each maze is represented as a **17×17 grid**.
- Internally the program starts from `[0][0]` and ends at `[16][16]` (0-based indices).  
- Output uses 1-based coordinates, so `[0][0]` prints as `{1,1}`, `[16][16]` prints as `{17,17}`.
- BFS is applied starting from array position `[0][0]` (displayed as `{1,1}`).
- Movement directions: right, down, left, up.  
- Step count:
  - Entering cell `1` adds +1  
  - Entering cell `2` adds +2 
- Parent pointers (`pi`) are maintained to reconstruct the path.  
- The result includes:
  - Total number of steps
  - Sequence of coordinates from start to goal

## Input/Output
### Input
- `input.txt`: contains 20 mazes, each of size 17×17 with integers.  

### Output
- `output.txt`: step count and path for each maze pattern.  

Example:
pattern 1

step=33

{1,1}

{2,1}

...

{17,17}

## Files
- `BFS.cpp` – BFS implementation and file handling
- `input.txt` – example input mazes
- `output.txt` – example output
- `report.pdf` – report with algorithm explanation

## Requirements
- C++11 or higher
- Standard library: `<iostream>`, `<fstream>`, `<queue>`, `<vector>`

## Usage
1. Compile the program:
   ```
   g++ -std=c++11 -O2 -o maze_solver BFS.cpp
   ```
2. Run the solver:
  ```
  ./maze_solver
  ```
3. Results will be written to:
- output.txt

## Support
If you encounter issues or have questions:
- Open a GitHub Issue
- Contact: 40924seanshen@gmail.com

## Roadmap
- Generalize to arbitrary maze sizes (N×N)
- Support diagonal movements
- Add visualization of the maze and path

## Contributing
Contributions are welcome!

1. Fork this repository
2. Create a new branch: git checkout -b feature/my-feature
3. Commit your changes
4. Open a Pull Request

## Authors and Acknowledgment
Author: Sean Shen (沈昱翔)

## License
This project is licensed under the MIT License.

## Project Status
Completed – This project was developed as part of my **academic research practice** in algorithm design.
