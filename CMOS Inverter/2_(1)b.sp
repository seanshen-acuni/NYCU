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
vddi vin1 gnd 0
M1     vip      vin1    vss    vss    n_18    W=0.29u    L=980.00n    m=1
M2     vip      vin1    vdd    vdd    p_18    W=0.29u    L=302.00n    m=2

.dc vddi 0 1.8 0.0001
.probe dc v(vip) v(vin1)

.end