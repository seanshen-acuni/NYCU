# CMOS Transistor and Inverter Design (SPICE Simulation)

## Description
This project implements transistor-level CMOS circuit design and simulation using **HSPICE** with the `cic018.l` 0.18μm CMOS technology library.  
It covers both **NMOS/PMOS transistor characterization** and **CMOS inverter design** under different operating conditions.

## Contents
1. **NMOS / PMOS Design and Characterization**
   - Designed (W/L) for NMOS and PMOS transistors under VD=0.9 V, VG=0.9 V, IDS=10 μA.
   - Simulated IDS–VDS characteristics for multiple VG values (0.7 V–1.1 V).
   - Netlist: `NPMOS.sp`

2. **CMOS Inverter Design**
   - Supply voltage: VDD = 1.8 V, GND = 0 V
   - Two simulation sweeps were performed:
     - **IO variation**: VO = VI = 0.9 V, IO = 10 μA / 20 μA / 30 μA
     - **VI variation**: VO = 0.9 V, IO = 20 μA, VI = 0.7–1.1 V
   - Simulated VO–VI transfer characteristics to verify inverter sizing.
   - Netlists:
     - `CMOS_inv_io_caseA.sp`
     - `CMOS_inv_io_caseB.sp`
     - `CMOS_inv_vi_caseA.sp`
     - `CMOS_inv_vi_caseB.sp`

## Files
- `.sp` files – SPICE netlists for simulation
- `cic018.l` – 0.18 μm CMOS technology library (required for simulation)
- `report.pdf` – Report with schematics, sizing details, and simulation results

## Requirements
- **HSPICE** (or compatible SPICE simulator)
- `cic018.l` technology file in the same directory

## Usage
1. Compile and run a netlist using HSPICE:
   ```
   hspice cmos_inverter_io_variation_caseA.sp > caseA.lis
   ```
2. Inspect the .lis file for measurements and view waveforms in HSPICE Waveform Viewer (HSPUI).

## Author
Sean Shen (沈昱翔)

## License
This project is part of an academic portfolio. All rights reserved by the author.
Not intended for commercial use.

## Project Status
Completed – This project was developed as part of my academic research practice in CMOS transistor and inverter design.
