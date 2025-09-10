******Inverter Design***********
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
vdd vdd gnd 1.8
vddx vss gnd 0
vddp vip gnd 0.9
vdi vin1 gnd 0.9
M1     vip      vin1    vss    vss    n_18    W=0.49u    L=430.00n    m=1
M2     vip      vin1    vdd    vdd    p_18    W=0.45u    L=196.00n    m=2

.dc vddp 0 1.8v 0.0001
.probe dc i(M2) i(M1) 
.meas dc ix1 find i(M1) at = 0.9
.meas dc ix2 find i(M2) at = 0.9 
*.meas dc rop param='0.6/(ix2-ix1)'
*.meas dc ix5 find i(m5) at =1.2
*************
*.alter
*vdi vin1 gnd 0.7
*vdi vin1 gnd 1.2
*****************
*.alter
*vdi vin1 gnd 0.8
*vdi vin1 gnd 1.15
********************
*.alter
*vdi vin1 gnd 1.0
*vdi vin1 gnd 1.05
*********************
*.alter
*vdi vin1 gnd 1.1
*vdi vin1 gnd 0.975
************************
*.alter
*vdi vin1 gnd 0.95
**********
*.alter
*vdi vin1 gnd 0.9
.end
