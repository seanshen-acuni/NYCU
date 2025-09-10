.temp 27
.option list node post
.lib "cic018.l" tt
.unprotect
VDD  vdd  gnd   1.8
VSS  vss  gnd   0

.SUBCKT inv_unit in out vdd vss
M1 out in  vss  vss  n_18 W=0.29u L=0.977u m=1
M2 out in  vdd  vdd  p_18 W=0.29u L=0.303u m=2
.ENDS

.SUBCKT dand_footed out Ain Bin phi vdd vss
MPRE  out phi  vdd  vdd  p_18 W=0.29u L=0.303u m=2
MFOOT net_foot phi  vss  vss  n_18 W=0.29u L=0.977u m=1
MNA   out Ain  net_mid net_mid n_18 W=0.29u L=0.977u m=1
MNB   net_mid Bin net_foot net_foot n_18 W=0.29u L=0.977u m=1
.ENDS

.SUBCKT dand_unfooted out Ain Bin phi vdd vss
MPRE  out phi  vdd  vdd  p_18 W=0.29u L=0.303u m=2
MNA   out Ain  net_mid net_mid n_18 W=0.29u L=0.977u m=1
MNB   net_mid Bin     vss     vss   n_18 W=0.29u L=0.977u m=1
.ENDS

Xa_inv  A  Abar  vdd vss inv_unit
Xe_inv  E  Ebar  vdd vss inv_unit

Xdand1   AE_pre  A     E     phi  vdd vss dand_footed
Xinv_ae  AE_pre  AE    vdd vss inv_unit
Xdand2   AEn_pre A     Ebar  phi  vdd vss dand_footed
Xinv_aen AEn_pre AEn   vdd vss inv_unit
Xdand3   AbE_pre Abar  E     phi  vdd vss dand_footed
Xinv_abe AbE_pre AbE   vdd vss inv_unit
Xdand4   AbEn_pre Abar Ebar  phi  vdd vss dand_footed
Xinv_aben AbEn_pre AbEn vdd vss inv_unit

Cload1 AE   gnd  2.105E-13
Cload2 AEn  gnd  2.105E-13
Cload3 AbE  gnd  2.105E-13
Cload4 AbEn gnd  2.105E-13

Vphi phi 0 PULSE(0 1.8 0n 0.5ns 0.5ns 25ns 50ns)
VA   A   0 PULSE(0 1.8 34ns 0.5ns 0.5ns 30ns 60ns)
VE   E   0 PULSE(0 1.8 4ns 0.5ns 0.5ns 20ns 40ns)

.tran 0.01ns 500ns
.probe tran v(A) v(Abar) v(E) v(Ebar) v(AE) v(AEn) v(AbE) v(AbEn) v(phi)

.end
