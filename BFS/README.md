# Maze Solver with BFS

## Description
This project implements a **Breadth-First Search (BFS)** based maze solver in C++.  
It reads multiple 17×17 mazes from an input file, computes the shortest path from the start `(1,1)` to the goal `(17,17)`, and outputs the step count along with the path coordinates.

The solver supports:
- Different cell types (walkable cells, walls, weighted cells).
- Path reconstruction with predecessor tracking.
- Handling multiple maze patterns in sequence.

This project was developed as part of **Algorithm Homework 8**.

## Algorithm
- Each maze is represented as a **17×17 grid**.  
- BFS is applied starting from the top-left corner `(1,1)` (index 0).  
- Movement directions: right, down, left, up.  
- Costs:
  - Normal cell (`1`) → step cost = 1
  - Special weighted cell (`2`) → step cost = 2
- Parent pointers (`pi`) are maintained to reconstruct the path.  
- The result includes:
  - Total number of steps
  - Sequence of coordinates from start to goal

## Input/Output
### Input
- `input.txt`: contains 20 mazes, each of size 17×17 with integers.  

### Output
- `110612008_output.txt`: step count and path for each maze pattern.  

Example:
pattern 1

step=33

{1,1}

{2,1}

...

{17,17}

## Files
- `main.cpp` – BFS implementation and file handling
- `input.txt` – example input mazes
- `110612008_output.txt` – example output
- `110612008_沈昱翔_HW8_report.pdf` – report with algorithm explanation

## Requirements
- C++11 or higher
- Standard library: `<iostream>`, `<fstream>`, `<queue>`, `<vector>`

## Usage
1. Compile the program:
   ```
   g++ -std=c++11 -O2 -o maze_solver main.cpp
   ```
2. Run the solver:
  ```
  ./maze_solver
  ```
3. Results will be written to:
- 110612008_output.txt

## Roadmap
- Generalize to arbitrary maze sizes (N×N)
- Support diagonal movements
- Add visualization of the maze and path

## Authors and Acknowledgment
- Author: Sean Shen (沈昱翔)

## License
This project is licensed under the MIT License.

## Project Status
For Academic Portfolio – This project was developed as part of coursework and is shared for demonstration purposes. No active maintenance is planned.
