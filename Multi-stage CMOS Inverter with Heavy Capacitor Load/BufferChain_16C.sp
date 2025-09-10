.option post=2 probe
.lib "cic018.l" tt

Vdd vdd 0 1.8
Vss vss 0 0
Vin vin 0 PULSE(0 1.8 0n 0.5ns 0.5ns 50ns 100ns)
.param Cp=3.2897E-15    
.param g=4  

.subckt inverter in out vdd vss Wp=0.29u Lp=303n Wn=0.29u Ln=977n
M1 out in vdd vdd p_18 W=Wp L=Lp
M2 out in vss vss n_18 W=Wn L=Ln
.ends inverter

Xinv1 vin vout1 vdd vss inverter Wp=0.29u Lp=303n Wn=0.29u Ln=977n
Xinv2 vout1 vout2 vdd vss inverter Wp=1.16u Lp=303n Wn=1.16u Ln=977n
Xinv3 vout2 vout3 vdd vss inverter Wp=4.64u Lp=303n Wn=4.64u Ln=977n

Cl vout3 0 5.2635E-14

.tran 0.01ns 500ns
.probe v(*)

.end




