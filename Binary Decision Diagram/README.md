# Reduced Ordered BDD (ROBDD) Node Minimization

## Description
This project builds a **Reduced Ordered Binary Decision Diagram (ROBDD)** for a given Boolean function under multiple **variable orders**, and outputs the **minimum number of unique nodes** among those orders.

The solver supports:
- Boolean functions in sum-of-products form.
- Uppercase letters represent negation of the corresponding lowercase variable (e.g., `A` = NOT `a`).
- Multiple variable orderings tested sequentially.
- Node reduction with:
  - Rule 1: If both children are the same, return that child.
  - Unique table: Nodes with identical `(var, low, high)` are shared.

## Algorithm
**Example input (case1.txt):**
  ```
  ab+cd.
  acbd.
  abcd.
  ```
- `<function>`: OR is `+`, AND is concatenation, **uppercase = NOT lowercase**.  
  e.g. `ab+CD` means `(a AND b) OR (NOT c AND NOT d)`.
- `<orderX>`: variable ordering string without spaces, e.g. `abcd`, `acbd`.
- For each ordering:
  - Recursively build the BDD by assigning variables in order.
  - Use reduction rules and unique table to minimize nodes.
  - Count unique nodes.
- Final output: the **minimum unique node count** among all orderings.

## Input/Output
- Input: `caseX.txt` (format as above)
- Output: `outX.txt` (a single integer)

## Files
- `BDD.cpp` – BDD implementation and file handling
- `Makefile` – build script (produces executable `BDD`)
- `case1.txt`, `case2.txt`, `case3.txt`, `case4.txt` – sample inputs
- `out1.txt`, `out2.txt`, `out3.txt`, `out4.txt` – sample outputs (generated)
- `ans1.txt`, `ans2.txt`, `ans3.txt`, `ans4.txt` - correct answer of each case

## Requirements
- C++11 or higher  
- Standard library: `<iostream>`, `<fstream>`, `<unordered_map>`, `<vector>`, `<string>`, `<set>`, `<climits>`, `<algorithm>`, `<sstream>`

## Usage
1. Compile the program with Makefile:
   ```
   make
   ```
2. Run the solver on test cases:
  ```
  ./BDD case1.txt out1.txt
  ./BDD case2.txt out2.txt
  ./BDD case3.txt out3.txt
  ./BDD case4.txt out4.txt
  ```

## Roadmap
- Add support for parentheses in Boolean expressions
- Add parser for more general Boolean syntax
- Provide visualization of the BDD structure

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
Completed – This project was developed as part of my academic research practice in EDA tools.







