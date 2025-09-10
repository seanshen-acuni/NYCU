# CMOS Digital Circuit Design and SPICE Simulation

## Description
This repository contains transistor-level CMOS digital circuits implemented and simulated in SPICE.  
The projects demonstrate **device sizing, load-driven buffer optimization, sequential logic, and dynamic logic design** under a 0.18 μm CMOS technology model (`cic018.l`).

Key learning outcomes:
- Extracting unit inverter capacitance and using it as a sizing baseline  
- Designing buffer chains to drive large capacitive loads efficiently  
- Measuring propagation delay, rise/fall times, and voltage swing with `.meas` automation  
- Implementing sequential logic with CMOS D latches  
- Analyzing dynamic logic (footed vs. unfooted structures)  

## Files
- `unit_inv_C.sp` – Unit inverter sized at Vdd=1.8 V, Vi=Vo=0.9 V, used for input capacitance measurement  
- `buffer_chain_16C.sp` – Multi-stage buffer chain driving ~16× unit capacitance; transient analysis with pulse input  
- `buffer_chain_fo4_16C.sp` – Fanout-of-4 buffer chain optimized for delay; rise/fall time and delay measured at each stage  
- `CMOS_posedge_D_latch.sp` – Positive-edge triggered CMOS D latch; verified with a clocked input sequence  
- `2_input_decoder_footed_unfooted.sp` – Dynamic logic 2-input decoder; comparison between footed and unfooted designs  
- `cic018.l` – 0.18 μm CMOS technology library models  
- `report.pdf` – Supporting report with schematics, measurement setup, and results

## Tools & Methods
- **SPICE simulation** (HSPICE / NGSPICE)  
- Technology model: `cic018.l` (0.18 μm CMOS)  
- Analyses performed:
  - `.dc` sweeps for inverter characterization and capacitance extraction  
  - `.tran` transient simulation with square-wave and PWL inputs  
  - `.meas` statements for delay, rise/fall time, and voltage swing  
  - Functional validation of sequential and dynamic circuits  

## Usage
1. Place `cic018.l` in the working directory.  
2. Run any netlist with a SPICE simulator, e.g.:  
   ```
   hspice buffer_chain_fo4_16x_load.sp > buffer_chain_fo4_16x_load.lis
   ```
3. Inspect .lis and waveform outputs to confirm capacitance, delay, and functionality.

## Author
Sean Shen (沈昱翔)

## License
This project is part of an academic portfolio. All rights reserved by the author.
Not intended for commercial use.

## Project Status
Completed – Circuits designed, simulated, and documented as part of my academic training in CMOS digital design.
