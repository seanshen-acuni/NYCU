.option post=2 probe
.lib "cic018.l" tt

Vdd vdd gnd 1.8
Vss vss gnd 0

Vin vin 0 PULSE(0 1.8 0 1n 1n 20n 40n)  * Input pulse for testing

.subckt inverter in out vdd vss Wp=0.29u Lp=303n Wn=0.29u Ln=977n
M1 out in vdd vdd p_18 W=Wp L=Lp
M2 out in vss vss n_18 W=Wn L=Ln
.ends inverter

Xinv1 vin vout1 vdd vss inverter Wp=0.29u Lp=303n Wn=0.29u Ln=977n
Xinv2 vout1 vout2 vdd vss inverter Wp=1.16u Lp=303n Wn=1.16u Ln=977n
Xinv3 vout2 vout3 vdd vss inverter Wp=4.64u Lp=303n Wn=4.64u Ln=977n
Xinv4 vout3 vout4 vdd vss inverter Wp=18.56u Lp=303n Wn=18.56u Ln=977n


Cl vout4 0 5.2635E-14

.tran 0.001u 0.5u uic
.probe v(vout1) v(vout2) v(vout3) v(vout4)

.meas TRAN Vmax1 MAX V(vout1)
.meas TRAN Vmin1 MIN V(vout1)
.meas TRAN Trise1 TRIG V(vout1) VAL='Vmin1+0.1*(Vmax1-Vmin1)' RISE=1
+                TARG V(vout1) VAL='Vmin1+0.9*(Vmax1-Vmin1)' RISE=1
.meas TRAN Tfall1 TRIG V(vout1) VAL='Vmin1+0.9*(Vmax1-Vmin1)' FALL=1
+                TARG V(vout1) VAL='Vmin1+0.1*(Vmax1-Vmin1)' FALL=1

.meas TRAN Vmax2 MAX V(vout2)
.meas TRAN Vmin2 MIN V(vout2)
.meas TRAN Trise2 TRIG V(vout2) VAL='Vmin2+0.1*(Vmax2-Vmin2)' RISE=1
+                TARG V(vout2) VAL='Vmin2+0.9*(Vmax2-Vmin2)' RISE=1
.meas TRAN Tfall2 TRIG V(vout2) VAL='Vmin2+0.9*(Vmax2-Vmin2)' FALL=1
+                TARG V(vout2) VAL='Vmin2+0.1*(Vmax2-Vmin2)' FALL=1

.meas TRAN Vmax3 MAX V(vout3)
.meas TRAN Vmin3 MIN V(vout3)
.meas TRAN Trise3 TRIG V(vout3) VAL='Vmin3+0.1*(Vmax3-Vmin3)' RISE=1
+                TARG V(vout3) VAL='Vmin3+0.9*(Vmax3-Vmin3)' RISE=1
.meas TRAN Tfall3 TRIG V(vout3) VAL='Vmin3+0.9*(Vmax3-Vmin3)' FALL=1
+                TARG V(vout3) VAL='Vmin3+0.1*(Vmax3-Vmin3)' FALL=1

.meas TRAN Vmax4 MAX V(vout4)
.meas TRAN Vmin4 MIN V(vout4)
.meas TRAN Trise4 TRIG V(vout4) VAL='Vmin4+0.1*(Vmax4-Vmin4)' RISE=1
+                TARG V(vout4) VAL='Vmin4+0.9*(Vmax4-Vmin4)' RISE=1
.meas TRAN Tfall4 TRIG V(vout4) VAL='Vmin4+0.9*(Vmax4-Vmin4)' FALL=1
+                TARG V(vout4) VAL='Vmin4+0.1*(Vmax4-Vmin4)' FALL=1
.end
