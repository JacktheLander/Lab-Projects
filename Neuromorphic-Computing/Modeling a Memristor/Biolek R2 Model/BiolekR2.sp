Biolek R2 Model for a Bipolar Threshold Memristive Device

********************************************************************************
* Connections:
* p - top electrode
* n - bottom electrode
* x - External connection to plot state variable that is not used otherwise
********************************************************************************
.SUBCKT BiolekR2 p n

********************************************************************************
* Device Constants
*********************************************************************************
.PARAMS Ron=1k Roff=10k Rinit=(Ron+Roff)/2 Beta=1e13 Vth=4.6V b1=10u b2=10u x0={Rinit}
.PARAM Cval=1

********************************************************************************
* Theta Function
********************************************************************************
.FUNC theta(x, b)  {1 / (1 + EXP(-x / b))}

********************************************************************************
* Gamma Function
********************************************************************************
.FUNC gamma(x, b)  {x * ({theta(x, b)} - {theta(-x, b)})}

********************************************************************************
* f Function
********************************************************************************
.FUNC f(Vin) { Beta * (Vin - (0.5 * ( {gamma((Vin + Vth), b1)} - {gamma((Vin - Vth), b1)} ))) }

********************************************************************************
* W Function
********************************************************************************
.FUNC W(Vin, x) { ({theta(Vin, b1)} * {theta(1 - (x / Roff), b2)}) + ({theta(-Vin, b1)} * {theta((x / Ron) - 1, b2)}) }

********************************************************************************
* IV Response
********************************************************************************
.FUNC FuncIV(v, x) {v / x}

********************************************************************************
* The Internal State Variable x(t)
********************************************************************************
.IC V(x)=x0
Cx x 0 {1}
Gx 0 x value = { {f(V(p, n))} * {W(V(p, n), V(x))} }

********************************************************************************
* dependent current source for memristor IV response
********************************************************************************
Gm p n value = {FuncIV(V(p,n), V(x))}

********************************************************************************
.ENDS BiolekR2



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
.TRAN 0.1n 0.1u

********************************************************************************
************************ Network Connection ************************************
********************************************************************************
* Network Type          : Crossbar
* Rows (inputs)         : 1
* Columns (outputs)     : 1
* Number of Modules     : 1 (X is the memristor subcircuit)
********************************************************************************
X0  1  0    BiolekR2

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
*WRDATA BiolekR2.csv V(1) I(V1) V(X0.x)
PLOT -I(Vs) vs V(1) retraceplot
PLOT v(x0.x)
.ENDC
********************************************************************************
.END

