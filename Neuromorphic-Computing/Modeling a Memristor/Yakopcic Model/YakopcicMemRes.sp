Yakopcic memristor model for Oblea memristive device

********************************************************************************
*************** the sub circuit for the memristor model ************************
********************************************************************************
* C. Yakopcic, T. M. Taha, G. Subramanyam, and R. E. Pino, "Generalized memristive
*     device SPICE model and its application in circuit design," Comput. Des. Integr.
*     Circuits Syst. IEEE Trans., vol. 32, no. 8, pp. 1201–1214, 2013.
********************************************************************************
* SPICE model for memristive devices
* Created by Chris Yakopcic
* Last Update: 12/21/2011
********************************************************************************
* Connections:
* p - top electrode
* n - bottom electrode
* x - External connection to plot state variable that is not used otherwise
********************************************************************************
.SUBCKT YakopcicMemRes p n PARAMS: xinit=0.11
********************************************************************************
* Fitting parameters to model different devices
* a1, a2, b:      Parameters for IV relationship
* Vp, Vn:         Pos. and neg. voltage thresholds
* Ap, An:         Multiplier for SV motion intensity
* xp1, xn:         Points where SV motion is reduced
* alphap, alphan: Rate at which SV motion decays
* xo:             Initial value of SV
* eta:            SV direction relative to voltage
********************************************************************************
* Oblea, Antonio S., Achyut Timilsina, David Moore, and Kristy a. Campbell. 2010.
*	    "Silver Chalcogenide Based Memristor Devices.” The 2010 International
*	    Joint Conference on Neural Networks (IJCNN) (July): 1–3.
*	    doi:10.1109/IJCNN.2010.5596775.
*	    http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=5596775.
********************************************************************************
* device constants
* Vp=0.16V, Vn=0.15V, Ap=4000, An=4000, xp=0.3, xn=0.5,
* alphap=1 alphan=5, a1=0.17, a2=0.17, b=0.05, x0=0.11, eta=1.
********************************************************************************
.PARAMS a1=0.17 a2=0.17 b=0.05 Vp=0.16 Vn=0.15 Ap=4000 An=4000 xp=0.3 xn=0.5
+alphap=1 alphan=5 x0={xinit} eta=1

********************************************************************************
.PARAM Cval=1
********************************************************************************
* Windows functions ensure zero state variable motion at memristor boundaries
********************************************************************************
.FUNC wp(x)    {(xp-x)/(1-xp) + 1}
.FUNC wn(x)    {x/(1-xn)}

********************************************************************************
* Threshold functions:
* Use TERNARY_FCN(x, y, x) for ngspice and IF(x, y, z) for other Pspice versions
********************************************************************************
.FUNC GAp(V)   { Ap*(exp(V)  - exp(Vp))}
.FUNC GAn(V)   {-An*(exp(-V) - exp(Vn))}
.FUNC G(V)     {TERNARY_FCN(V <= Vp, TERNARY_FCN(V >= -Vn, 0, GAn(V)), GAp(V))}

********************************************************************************
* The motion functions
* Use TERNARY_FCN(x, y, x) for ngspice and IF(x, y, z) for other Pspice versions
********************************************************************************
.FUNC Fc(x)     {TERNARY_FCN(x >= xp, exp(-alphap*(x-xp))*wp(x), 1)}
.FUNC Fd(x)     {TERNARY_FCN(x <= (1-xn), exp(alphan*(x+xn-1))*wn(x) , 1)}
.FUNC Fm(v, x)  {TERNARY_FCN(eta*v >= 0, Fc(x), Fd(x))}

********************************************************************************
* IV Response - Hyperbolic sine due to MIM structure
* Use TERNARY_FCN(x, y, x) for ngspice and IF(x, y, z) for other Pspice versions
********************************************************************************
.FUNC FuncIV(v, x) {TERNARY_FCN(v >= 0, a1*x*sinh(b*v), a2*x*sinh(b*v))}

********************************************************************************
* the state variable x(t)
* dx/dt = F(V(t),x(t))*G(V(t))
********************************************************************************
.IC V(x)=x0
Cx x 0 {1}
Gx 0 x value = {eta * G(V(p,n)) * Fm(V(p,n),V(x))}

********************************************************************************
* dependent current source for memristor IV response
********************************************************************************
Gm p n value = {FuncIV(V(p,n), V(x))}

********************************************************************************
.ENDS YakopcicMemRes

********************************************************************************
********************* Parameters for Signal Source(s) **************************
********************************************************************************
* SINE    : V1 1 0 DC 0 SIN(OFF AMPL FREQ TD THETA)
* PULSE   : V1 1 0 DC 0 PULSE(VLO VHI TD TR TF PW PER)
* DC      : V1 1 0 1.2 DC
* TRIANGLE: V1 1 0 DC 0 PULSE(VLO VHI TD TR TF PW PER)
* SQUARE  : V1 1 0 DC 0 PULSE(VLO VHI TD TR TF PW PER)
********************************************************************************
* V1: Sine, ampl = 0.5, freq = 100, off = 0
********************************************************************************
.PARAM  Vampl = 0.5
.PARAM  Freq  = 100
********************************************************************************
Vs 1  0 DC 0 SIN(0 {Vampl} {Freq})

********************************************************************************
********************* General Options for Xyce *********************************
********************************************************************************
* AztecOO is the default linear solver for parallel execution. However, AztecOO
* doesn't work at this point. KLU is selected for linear solver instead. It is
* serial execution. Method of Gear is to fix the problem with exceeding limits.
********************************************************************************
*.OPTIONS LINSOL TYPE=KLU
*.OPTIONS TIMEINT METHOD=GEAR

********************************************************************************
******************* Parameters for transient analysis **************************
********************************************************************************
* Analysis type         : Transient Analysis
* Time Step             : 1us
* Number of Simulations : 10000
* Start Transient Time  : 0us
* Stop Transient Time   : 10000us (Time Step * Number of Simulation)
* Transient Option      : .TRAN TSTEP TSTOP  <TSTART <TMAX(step)> > <UIC>
********************************************************************************
.TRAN 1us 10ms 0us 1us

********************************************************************************
************************ Network Connection ************************************
********************************************************************************
* Network Type          : Crossbar
* Rows (inputs)         : 1
* Columns (outputs)     : 1
* Number of Modules     : 1 (X is the memristor subcircuit)
********************************************************************************
X0  1  0    YakopcicMemRes

********************************************************************************
********************** Measurements for analysis *******************************
********************************************************************************
* Measurement Type      : SAVE
* Measurement Interval  : from 0us to 10000us
* Measured Elements     : V(1)
* Measured Elements     : I(V1)
********************************************************************************
*.PRINT TRAN FORMAT=csv FILE=MemResOblea.csv
*+	 V(1) I(V1) V(X0:X)
********************************************************************************
* the following controls are specifically for ngspice. Other Pspice version can
* have different sets of controls.
********************************************************************************
.CONTROL
*SET wr_singlescale
*SET wr_vecnames
*OPTION numdgt =7
RUN
*WRDATA YakopcicMem.csv V(1) I(V1) V(X0.x)
PLOT -I(Vs) vs V(1) retraceplot
.ENDC
********************************************************************************
.END
