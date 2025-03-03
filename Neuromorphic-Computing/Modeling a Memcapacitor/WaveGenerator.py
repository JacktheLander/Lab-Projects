from scipy.signal import square, sawtooth
import numpy as np
# import matplotlib.pyplot as plt

# ******************************************************************************
"""
Class for wave generator:
    Inputs: Ampl, Offset, Freq, SamplePerCycle, NumCycles
"""
class WaveGenerator:
    # initializing method
    def __init__(self, SignalType=None, Ampl=1.0, Offset=0.0, Freq=1.0, Sample=1000, NumCycles=1.0,
        Verbose=False):
        # class name
        self.ClassName = "WaveGenerator"

        # set the constants
        self.Ampl       = Ampl
        self.Offset     = Offset
        self.Freq       = Freq
        self.Sample     = Sample
        self.NumCycles  = NumCycles
        self.SignalType = SignalType
        self.Verbose    = Verbose
        self.ts         = None
        self.WaveTypes  = ["Sine", "Square", "Triangle", "Sawtooth", "Piecewise",\
                "SpikeTrain"]

        # display the message
        if self.Verbose:
            # display the information
            Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
            print(Msg)

        # check the signal type
        self._CheckError(self.SignalType)

        # check the signal type
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::__init__()"

            # display the information
            Msg = "...%-25s: Signal Type = %s" % (FunctionName, self.SignalType)
            print(Msg)

            # display the information
            Msg = "...%-25s: Ampl = %.3g, Off = %.3g, Freq = %.3g, Sample = %.3g, Cycle = %.3g" % \
                    (FunctionName, self.Ampl, self.Offset, self.Freq, self.Sample, \
                    self.NumCycles)
            print(Msg)

    # **************************************************************************
    def _CheckError(self, SignalType):
        # set the function name
        FunctionName = "WaveGenerator::_CheckError()"

        # check the signal type
        if SignalType is None:
            # set the error message
            ErrMsg = "%s:<%s> => invalid signal type" % (FunctionName, str(self.SignalType))
            raise ValueError(ErrMsg)

        # check the signal type
        if SignalType not in self.WaveTypes:
            # set the error message
            ErrMsg = "%s:<%s> => not implemented yet" % (FunctionName, str(self.SignalType))
            raise ValueError(ErrMsg)

    # **************************************************************************
    # set functions
    # **************************************************************************
    def SetSignalType(self, SignalType):
        self.SignalType = SignalType

    def SetSample(self, Sample):
        self.Sample = Sample

    def SetNumCycles(self, NumCycles):
        self.NumCycles = NumCycles

    # **************************************************************************
    # set time space
    # **************************************************************************
    def _SetTimeTs(self, Freq=None, Sample=None, NumCycles=None):
        # check the frequency
        if Freq is None:
            Freq = self.Freq
        else:
            self.Freq = Freq

        # check the sample
        if Sample is None:
            Sample = self.Sample
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # calculate the period
        Period = 1.0 / Freq

        # determine the range for x in terms of seconds
        Start = 0.0
        Stop  = Period * NumCycles
        NumSpaces = int(Sample * NumCycles)

        # create a linear space
        return np.linspace(Start, Stop, NumSpaces)

    # **************************************************************************
    def GetTimeTs(self, Freq=None, Sample=None, NumCycles=None):
        return self._SetTimeTs(Freq=Freq, Sample=Sample, NumCycles=NumCycles)

    # **************************************************************************
    # sine wave
    # **************************************************************************
    def Sine(self, Ampl=None, Offset=None, Freq=None, Sample=None, NumCycles=None, ts=None):
        # check the verbose flag
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::Sine()"

            # display the information
            Msg = "...%-25s: generating Sine function..." % (FunctionName)
            print(Msg)

        # check the amplitude
        if Ampl is None:
            Ampl = self.Ampl
        else:
            self.Ampl = Ampl

        # check the offset
        if Offset is None:
            Offset = self.Offset
        else:
            self.Offset = Offset

        # check the frequency
        if Freq is None:
            Freq = self.Freq
        else:
            self.Freq = Freq

        # check the sample
        if Sample is None:
            Sample = self.Sample
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # check the time variable
        if ts is None:
            # create a linear space
            ts = self._SetTimeTs(Freq, Sample, NumCycles)

        # generate the sine wave
        # y = np.asarray([Ampl * np.sin(i*Freq*2*np.pi) + Offset for i in ts])
        y = Ampl * np.sin(2 * np.pi * Freq * ts) + Offset

        return y, ts

    # ***************************************************************************
    # square wave
    # **************************************************************************
    def Square(self, Ampl=None, Offset=None, Freq=None, Sample=None, NumCycles=None, ts=None):
        # check the verbose flag
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::Square()"

            # display the information
            Msg = "...%-25s: generating Square wave..." % (FunctionName)
            print(Msg)

        # check the amplitude
        if Ampl is None:
            Ampl = self.Ampl
        else:
            self.Ampl = Ampl

        # check the offset
        if Offset is None:
            Offset = self.Offset
        else:
            self.Offset = Offset

        # check the frequency
        if Freq is None:
            Freq = self.Freq
        else:
            self.Freq = Freq

        # check the sample
        if Sample is None:
            Sample = self.Sample
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # check the time variable
        if ts is None:
            # create a linear space
            ts = self._SetTimeTs(Freq, Sample, NumCycles)

        # square wave generator
        #  y = Ampl * square(2.0 * np.pi * Freq * ts + np.pi) + Offset
        y = Ampl * square(2.0 * np.pi * Freq * ts) + Offset

        return y, ts

    # ***************************************************************************
    # pulse wave
    # **************************************************************************
    def Pulse(self, PulseWidth, PulsePeriod, Ampl=None, Pos=True, Sample=None, \
            NumCycles=None, DelayFirst=False, ts=None):
        # set the function name
        FunctionName = "WaveGenerator::Pulse()"

        # check the verbose flag
        if self.Verbose:
            # display the information
            Msg = "...%-25s: generating Pulse wave..." % (FunctionName)
            print(Msg)

        # check the amplitude
        if Ampl is None:
            Ampl = self.Ampl
        else:
            self.Ampl = Ampl

        # check positve pulse option
        if Pos:
            # adjust the amplitude
            Ampl /= 2.0
            Offset = Ampl
        else:
            Offset = 0.0

        # set the frequency
        Freq = self.Freq = 1.0 / PulsePeriod

        # check the sample
        if Sample is None:
            Sample = self.Sample
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # check the time variable
        if ts is None:
            # create a linear space
            ts = self._SetTimeTs(Freq, Sample, NumCycles)

        # set duty cycle
        if PulseWidth < PulsePeriod:
            DutyCycle =  PulseWidth / PulsePeriod
        else:
            # set the error message
            ErrMsg = "%s: PulseWidth = <%.5g> <= PulsePeriod = <%.5g>" % (FunctionName, \
                    PulseWidth, PulsePeriod)
            raise ValueError(ErrMsg)

        # square wave generator
        #  y = Ampl * square(2.0 * np.pi * Freq * ts + np.pi) + Offset
        y = Ampl * square(2.0 * np.pi * Freq * ts, duty=DutyCycle) + Offset

        # check the flag for setting delay before pulse
        if DelayFirst:
            Vs = np.flip(y, axis=0)
        else:
            Vs = y

        return Vs, ts

    # ***************************************************************************
    # triangle wave
    # **************************************************************************
    def Triangle(self, Ampl=None, Offset=None, Freq=None, Sample=None, \
            NumCycles=None, ts=None):
        # check the verbose flag
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::Triangle()"

            # display the information
            Msg = "...%-25s: generating Triangle wave..." % (FunctionName)
            print(Msg)

        # check the amplitude
        if Ampl is None:
            Ampl = self.Ampl
        else:
            self.Ampl = Ampl

        # check the offset
        if Offset is None:
            Offset = self.Offset
        else:
            self.Offset = Offset

        # check the frequency
        if Freq is None:
            Freq = self.Freq
        else:
            self.Freq = Freq

        # check the sample
        if Sample is None:
            Sample = self.Sample
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # check the time variable
        if ts is None:
            # create a linear space
            ts = self._SetTimeTs(Freq, Sample, NumCycles)

        # wave type:
        #   1 (default) : rising ramp
        #   0           : falling ramp
        #   0.5         : triangle
        Type = 0.5

        # get the wave form
        y = Ampl * sawtooth(2.0 * np.pi * Freq * ts - 3 *np.pi/2.0, width=Type) + Offset

        return y, ts

    # ***************************************************************************
    # Sawtooth wave
    # **************************************************************************
    def Sawtooth(self, Ampl=None, Offset=None, Freq=None, Sample=None, NumCycles=None, \
            ts=None, Type=1.0):
        # check the verbose flag
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::Sawtooth()"

            # display the information
            Msg = "...%-25s: generating Sawtooth wave..." % (FunctionName)
            print(Msg)

        # check the amplitude
        if Ampl is None:
            Ampl = self.Ampl
        else:
            self.Ampl = Ampl

        # check the offset
        if Offset is None:
            Offset = self.Offset
        else:
            self.Offset = Offset

        # check the frequency
        if Freq is None:
            Freq = self.Freq
        else:
            self.Freq = Freq

        # check the sample
        if Sample is None:
            Sample = self.Sample
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # check the time variable
        if ts is None:
            # create a linear space
            ts = self._SetTimeTs(Freq, Sample, NumCycles)

        # wave type: 1 (default) : rising ramp, 0: falling ramp
        # Type = 1.0

        # get the wave form
        y = Ampl * sawtooth(2.0 * np.pi * Freq * ts - np.pi, width=Type) + Offset

        return y, ts

    # **************************************************************************
    # create time space
    # **************************************************************************
    def CreateTimeTs(self, Freq, Sample, NumCycles):
        # create a linear space
        return self._SetTimeTs(Freq, Sample, NumCycles)

    # **************************************************************************
    def _LinearEquation(self, P1, P2):
        # calculate slope
        m = (P2[1] - P1[1]) / (P2[0] - P1[0])
        b = P2[1] - m * P2[0]
        return m, b

    # **************************************************************************
    # piece wise linear function
    # **************************************************************************
    def PWL(self, PointSet, Sample=None, NumCycles=None):
        # check the verbose flag
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::PWL()"

            # display the information
            Msg = "...%-25s: generating piecewise wave..." % (FunctionName)
            print(Msg)

        # check the sample
        if Sample is None:
            Sample = int(self.Sample)
        else:
            self.Sample = Sample

        # check the number of cycles
        if NumCycles is None:
            NumCycles = self.NumCycles
        else:
            self.NumCycles = NumCycles

        # get the x value of the first point
        x1, y1 = PointSet[0]

        # get the x value of the last point
        xL, yL = PointSet[-1]

        # one cycle
        OneCycleTs = np.linspace(x1, xL, int(Sample))

        # set the value
        Start = x1
        Stop  = x1 + (xL - x1) * NumCycles
        NumSpaces = Sample * NumCycles

        # set the time variable
        ts = np.linspace(Start, Stop, int(NumSpaces))

        # reset the y value
        NumTsOneCycle = len(OneCycleTs)
        y = np.zeros(NumTsOneCycle * NumCycles)

        # get the number of points
        NumPoints = len(PointSet) - 1

        # reset the variable
        j = 0

        # construct the function
        for i in range(NumPoints):
            # get two points
            P1 = PointSet[i]
            x1, y1 = P1

            P2 = PointSet[i+1]
            x2, y2 = P2

            # check the x coordinate
            if x1 == x2:
                # save the y value
                y[j] = y2

                # update the index
                j += 1

                # skip the rest
                continue

            # get the slope and the intercept
            m, b = self._LinearEquation(P1, P2)

            # check the time
            while OneCycleTs[j] < P2[0]:
                # calculate the y values
                y[j] = m * OneCycleTs[j] + b

                # update the
                j += 1

        # repeat the cycle for the rest of cycles
        if NumCycles > 1:
            # set the one cycle
            OneCycle = y[0:j]

            # copy the values
            for i in range(NumCycles - 1):
                StartIndex = j
                EndIndex = j + NumTsOneCycle - 1
                y[StartIndex:EndIndex] = OneCycle

                # update index
                j = EndIndex

        return y, ts

    # **************************************************************************
    # spike train
    # **************************************************************************
    """
        Spike times = indices of spikes
        Spike Ampl  = 1.0
        Spike width = 1ms
        Points/Spikes (Pps) = 10
        Interval    = TimeInterval * 1ms = 1500ms = 1.5s
    """
    def SpikeTrainTs(self, Pps=10, TimeInterval=500):
        # check the verbose flag
        if self.Verbose:
            # set the function name
            FunctionName = "WaveGenerator::SpikeTrainTs()"

            # display the information
            Msg = "...%-25s: generating spike train timestep..." % (FunctionName)
            print(Msg)

        # calculate the time interval
        Interval = TimeInterval * 1e-3

        # set the time and spike variables
        ts = np.linspace(0.0, Interval, TimeInterval * Pps)
        return ts

    # **************************************************************************
    def SpikeTrain(self, SpikeTimes, ts=None, Ampl=1.0, Pps=10, TimeInterval=500):
        # set the function name
        FunctionName = "WaveGenerator::SpikeTrain()"

        # check the verbose flag
        if self.Verbose:
            # display the information
            Msg = "...%-25s: generating spike train..." % (FunctionName)
            print(Msg)

        # set the spike time length
        SizeSpikeTimes = len(SpikeTimes)

        # point per half spike
        Pphs = Pps / 2

        # set the time and spike variables
        if ts is None:
            ts = self.SpikeTrainTs(Pps=Pps, TimeInterval=TimeInterval)

        # reset the output
        y  = np.zeros(len(ts))

        # positive polarity
        Polarity = 1.0

        # set the spike
        for i in range(SizeSpikeTimes):
            # set the index
            TimeVal = SpikeTimes[i]

            # check the index
            if TimeVal == 0:
                # skip the rest
                continue
            elif TimeVal < 0:
                # a negative spike
                Polarity = -1.0
            else:
                # a positive spike
                Polarity = 1.0

            # set the spike index
            j = np.absolute(TimeVal)
            j *= Pps
            Start = int(j - Pphs + 1)
            Stop  = int(j + Pphs)

            # set the amplitude
            y[Start:Stop] = Ampl * Polarity

        return y, ts

    # **************************************************************************
    def ResetVerboseFlag(self):
        self.Verbose = False

# ******************************************************************************
if __name__ == "__main__":
    # import module
    import matplotlib.pyplot as plt
    import sys

    # signal type
    # SignalType = None
    # SignalType = "Sine"
    # SignalType = "Square"
    # SignalType = "Triangle"
    # SignalType = "Harmonic"
    SignalType = "SpikeTrain"

    # set the parameters
    Ampl    = 2
    Freq    = 1e3
    Offset  = 0
    N       = 2
    NumStep = 1000
    Verbose = True

    # get the object
    Input = WaveGenerator(SignalType=SignalType, Ampl=Ampl, Offset=Offset, Freq=Freq, \
            Sample=NumStep, NumCycles=N, Verbose=Verbose)

    # create time variable
    ts = Input.CreateTimeTs(Freq, NumStep, N)
    XLabel = "Time (s)"

    # create the sine wave
    # Vsin, ts = Input.Sine(Ampl, Offset, Freq, NumStep, N, ts=ts)
    Vsin, ts = Input.Sine(ts=ts)
    print("Vsin  = ", Vsin.shape)

    # create the square wave
    # Vsquare, ts = Input.Square(Ampl, Offset, Freq, NumStep, N, ts=ts)
    Vsquare, ts = Input.Square(ts=ts)

    # create the pulse
    Scale = 4.0
    PulsePeriod = 1.0 / Freq
    PulseWidth  = PulsePeriod / Scale
    DelayFirst  = True
    Vpulse, ts = Input.Pulse(PulseWidth, PulsePeriod, ts=ts, DelayFirst=DelayFirst)

    # # create the triangle wave
    # # Vtriangle, ts = Input.Triangle(Ampl, Offset, Freq, NumStep, N, ts=ts)
    # Vtriangle, ts = Input.Triangle(ts=ts)
    #
    # # create the sawtooth wave
    # # Vssawtooth, ts = Input.Sawtooth(Ampl, Offset, Freq, NumStep, N, ts=ts)
    # Vssawtooth, ts = Input.Sawtooth(ts=ts)
    #
    # # set the points
    # Points = [[-1.0,-1.0], [1.0, 2.0], [3.0, 0], [4.0, 2.0], [6.0, 0]]
    N = 1
    Points = [[0.0, 0.0], [1.0, 0.0], [1.0, 8.0], [5.0, 8.0], [5.0, 0], [9.0, 0.0]]
    PointSet = np.asarray(Points)

    # create piece wise linear function
    VsPWL, tsPWL = Input.PWL(PointSet, NumStep, N)
    VsPWL, tsPWL = Input.PWL(PointSet)

    # spike train
    # SpikeTimes = [0,1,3,7,11,0.0,0.0]
    # SpikeTrain, ts = Input.SpikeTrain(SpikeTimes)
    # ts    *= 1e3
    XLabel = "Time (ms)"

    # **************************************************************************
    # set the font family
    # **************************************************************************
    # FontSize = 12
    FontSize = 14
    # FontSize = 16
    # FontSize = 18
    # FontSize = 20
    # FontSize = 22
    # FontSize = 24
    # FontSize = 28
    font = {"family": "Times New Roman", "size": FontSize}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    # **************************************************************************
    # scale
    Scale = 1e3
    # Scale = 1
    ts *= Scale

    # get the figure handle
    Fig = plt.figure()
    plt.grid(linestyle="dotted")

    # plot the figure
    # plt.plot(ts, Vsin, "-", label="Vsin")
    # plt.plot(ts, Vsquare, "--", label="Vsquare")
    # plt.plot(ts, Vpulse, "--", label="Vpulse")
    # plt.plot(ts, Vtriangle, "-.", label="Vstriangle")
    # plt.plot(ts, Vssawtooth, ":", label="Vssawtooth")
    plt.plot(tsPWL, VsPWL, "-", label="Vs-PWL")
    # plt.plot(ts, SpikeTrain, "-", label="Vs-Spike")
    plt.xlabel(XLabel)
    plt.ylabel("Vs (V)")
    plt.axis("tight")
    # plt.xlim(0, 20)
    # plt.ylim(Vsin.min() * 1.1, Vsin.max() * 1.1)
    plt.legend(loc="best")
    plt.show()
