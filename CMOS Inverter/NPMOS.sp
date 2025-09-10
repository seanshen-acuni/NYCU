* NMOS PMOS
*.temp 27
.option list node post
.lib "cic018.l" tt
.unprotect
*.lib "cic018.l" TT   
*.OPTIONS ACCT $accounting and runtime statistics
*.OPTIONS POST $storing simuation results for AvanWaves in binary
*.OPTIONS NOMOD $suppresses the printout of model parameters
*.OPTIONS NOPAGE $suppresses page ejects for title headings
*.OPTIONS BRIEF $enable printback
*.OPTIONS INGOLD=0 $engineering format
*.OPTIONS method=gear
*.options list node post
* Voltage Sources
vdd vdd gnd 1.8
vgate vg gnd 0.9
vds vd gnd 0

* NMOS and PMOS Transistors
MN		vd		vg 			gnd 		gnd 		n_18 		W=0.29783u 		L=1u 		m=1
MP 		vd 		vg 			vdd 		vdd 		p_18 		W=1.0962u 		L=1u 		m=2

* Sweep VG for IDS-VDS Characteristics
.dc vds 0 1.8 0.01
.probe dc i(MN) i(MP)

* Measurement for Specific VG
.meas dc ix1 find i(MN) when v(vd)=0.9
.meas dc ix2 find i(MP) when v(vd)=0.9

* VG Alterations for IDS-VDS Comparison
.alter
vgate vg gnd 0.7
.alter
vgate vg gnd 0.8
.alter
vgate vg gnd 0.9
.alter
vgate vg gnd 1.0
.alter
vgate vg gnd 1.1

.end
