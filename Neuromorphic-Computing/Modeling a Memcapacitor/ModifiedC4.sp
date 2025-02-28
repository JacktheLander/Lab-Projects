Biolek C4 Model for a Bipolar Threshold Memcapacitive Device

********************************************************************************
* This Spice Model was modified from my original code in BiolekC4.sp by ChatGPT
********************************************************************************

.SUBCKT BiolekC4 p n
.PARAM Clow=1pF Chigh=100pF Cinit=Clow Beta=70u Vth=3V b1=1u b2=10m

* Define f(Vc) using sign function
.FUNC f(Vc, b) { Beta * (Vth - Vc) * (Vc / sqrt(Vc*Vc + 1u)) }

* Define step function
.FUNC STP (x, b)={1/( 1 + exp(-x/b)) }
.FUNC ABS (x ,b)={ x * ( STP (x ,b)- STP (-x ,b ))}

* Fix W(Vc, x) to balance step sizes
.FUNC W(Vc, x) = { STP(Vc - Vth, b1) * STP(Chigh - x, b2) - STP(-Vc - Vth, b1) * STP(x - Clow, b2) }

* Internal state update
Gx 0 x value ={ f( V(p, n), b1) * W( V(p, n), V(x) ) }
Rx x 0 100meg
Cx x 0 1 IC = { Cinit }

* Memcapacitor voltage source
Ec p n value = { V(Q) / (V(x)+1p) }

* Charge tracking
GQ 0 Q value = { I(Ec) }
CQ Q 0 10u
RQ Q 0 100meg
.ENDS BiolekC4

.options reltol = 0.1u
Vs 1 0 SIN(0 4 50k)
Ri 1 2 1m
Xmem 2 0 BiolekC4

Rp 2 0 1k

.tran 0.1u 100u 20u 0.1u

.CONTROL
RUN
PLOT V(Xmem.Q)
PLOT V(Xmem.x)
PLOT V(Xmem.Q) vs V(2) retraceplot
PLOT V(2)
PLOT V(1)
.endc

.end
