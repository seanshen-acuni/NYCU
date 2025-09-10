******Inverter Design***********
.temp 27
.option list node post
.lib "cic018.l" tt
.unprotect
vdd vdd gnd 1.8
vddx vss gnd 0
*vddp vip gnd 0.9
*vdi vin1 gnd 0.9

.SUBCKT inv11 in out AVDD AVSS
MM1 out in AVDD AVDD p_18 W=0.29u L=303n m=2
MM0 out in AVSS AVSS n_18 W=0.29u L=977n m=1
.ENDS
.SUBCKT patransistor in out clk clkb AVDD AVSS
MM1 out clkb in AVDD p_18 W=0.29u L=303n m=2
MM0 out clk in  AVSS n_18 W=0.29u L=977n m=1
.ENDS
X3 o2 ai1 clk clkb vdd vss patransistor
X4 VINX ai1 clkb clk vdd vss patransistor
X1 ai1 o1 vdd vss inv11
X2  o1 o2 vdd vss inv11
X5  clk clkb vdd vss inv11
*X6  VIN VIX vdd vss inv11
*X7  VIX VINX vdd vss inv11

.tran    0.001u    1.28us    uic
.probe dc v(o1) v(o2) v(ai1) v(VINX) v(clk) v(clkb) 
*v(VIN) 

Vclk clk VSS PULSE(0 1.8 0 0.002u 0.002u 0.018u 0.04us)
V1   VINX   GND   PWL(0n 0V 0.04u 0V 0.041u 1.8V 0.12u 1.8V 0.121u 0V 0.16u 0V 0.161u 1.8V 0.2u 1.8V 0.201u 0 0.28u 0 0.281u 1.8V 0.32u 1.8V 0.321u 0V 0.36u 0V 0.361u 1.8V 0.48u 1.8V 0.481u 0V 0.56u 0V 0.561u 1.8V 0.6u 1.8V 0.601u 0V 0.64u 0V )
.end
