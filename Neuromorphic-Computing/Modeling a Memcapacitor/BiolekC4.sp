Biolek C4 Model for a Bipolar Threshold Memcapacitive Device

********************************************************************************
.SUBCKT BiolekC4 p n
********************************************************************************
* Device Constants
*********************************************************************************
.PARAMS Clow=1pF Chigh=100pF Cinit={ (Clow + Chigh) / 2 } Beta=70u Vth=0V b1=10m b2=1u

********************************************************************************
* f Function
********************************************************************************
.FUNC f(Vc, b) { Beta*(Vc - 0.5*( ABS(Vc + Vth, b) - ABS(Vc - Vth, b) ))}

********************************************************************************
* W Function
* The Theta Function is STP - a step function
********************************************************************************
.FUNC STP (x, b)={1/( 1 + exp(-x/b) )}
.FUNC ABS (x ,b)={ x *( STP (x ,b)- STP (-x ,b ))}
.FUNC W(Vc, x) = { STP(Vc, b1) * STP(1-x/Chigh, b2) + STP(-Vc, b1) * STP(x/Clow-1, b2) }

********************************************************************************
* Internal State Variable x(t)
********************************************************************************
Gx 0 x value ={ f( V(p, n), b1) * W( V(p, n), V(x) + 1p) }
Rx x 0 100meg
Cx x 0 1 IC ={ Cinit }

********************************************************************************
* Dependent Current Source of Memcapacitive Port
********************************************************************************
Ec p n value = { V(Q) / V(x)}

********************************************************************************
* Charge Q
********************************************************************************
GQ 0 Q value = {I(Ec)}
CQ Q 0 1
RQ Q 0 100meg

********************************************************************************
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