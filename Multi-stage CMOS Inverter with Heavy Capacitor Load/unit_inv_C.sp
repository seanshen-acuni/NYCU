******Inverter Design***********
.temp 27
.option list node post
.lib "cic018.l" tt
.unprotect
.option captab=1
.option dccap=1
vdd vdd gnd 1.8
vddx vss gnd 0
vddp vip gnd 0.9
vdi vin1 gnd 0.9
M1     vip      vin1    vss    vss    n_18    W=0.29u      L=977.00n    m=1
M2     vip      vin1    vdd    vdd    p_18    W=0.29u    L=303.00n    m=2
************************************************************
*************************************************************
.dc vddp 0 1.8v 0.0001
.probe dc i(M2) i(M1) 
.meas dc ix1 find i(M1) at = 0.9
.meas dc ix2 find i(M2) at = 0.9 
.meas dc cp find cap(vin1) at=0.9
.end
