********************************************************************************
* Fixed Bipolar Threshold Memcapacitive Device
********************************************************************************
.SUBCKT BiolekC4 p n
.PARAM Clow=1pF Chigh=100pF Cinit={ (Clow + Chigh) / 2 } Beta=70u Vth=3V b1=1m b2=1m

* Define f(Vc) with Stronger Switching
.FUNC f(Vc, b) { Beta * (Vc - Vth) * STP(Vc - Vth, b1) - Beta * (Vc + Vth) * STP(-Vc - Vth, b1) }

* Define sharper step function
.FUNC STP (x, b)={1/( 1 + exp(-x/b)) }

* Define Harder Weight Function
.FUNC W(Vc, x) = { STP(Vc - Vth, b1) - STP(-Vc - Vth, b1) }

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
*PLOT V(2)
*PLOT V(1)
.endc

.end
