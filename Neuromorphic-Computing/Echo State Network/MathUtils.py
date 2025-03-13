# import modules
import numpy as np
import torch
from scipy.interpolate import interp1d

# ******************************************************************************
# calculate coefficients of a linear equation y = mx + b
#       P1 = [x1, y1], P2 = [x2, y2]
# ******************************************************************************
def LinearEquationCoefs(P1, P2):
    # **************************************************************************
    # get coordinate
    # **************************************************************************
    x1, y1  = P1
    x2, y2  = P2

    # **************************************************************************
    # calculate slope and the y-intercept
    # **************************************************************************
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x2
    return m, b

# ******************************************************************************
# find a number of linear points between X1 and X2
#   X1 = [t1, y1], X2 = [t2, y2]
# ******************************************************************************
def LinearPoints(X1, X2, Num=1):
    # get the coefficients
    m, b = LinearEquationCoefs(X1, X2)

    # set the points
    start = X1[0]
    stop  = X2[0]

    # interpolating x
    x, dx = np.linspace(start, stop, num=Num, endpoint=True, retstep=True)

    # recalculating y values
    y = m * x + b
    return x, y, dx

# ******************************************************************************
# InterpolateX
# ******************************************************************************
def InterpolateX(x, Num=2):
    # save the start and stop values
    Start  = x[0]
    Stop   = x[-1]
    Length = len(x)

    # extend the x values
    NumPoints = Length + Num
    NewX, dx  = np.linspace(Start, Stop, num=NumPoints, endpoint=True, retstep=True)
    return NewX, dx

# ******************************************************************************
# interpolate linearly the y values: x, y, Num=2, Type="cubic"
# ******************************************************************************
def InterpolateY(x, y, Num=2, Type="linear", NewX=None):
    # interpolate into a function f
    f = interp1d(x, y, kind=Type)

    # extend the x values
    if NewX is None:
        NewX, dx = InterpolateX(x, Num=Num)

    # calculate y values
    NewY = f(NewX)
    return NewX, NewY

# ******************************************************************************
# Find the two largest factors of an integer number
# ******************************************************************************
def _FindFactors2D(x, Dim=2):
    # **************************************************************************
    # find the factors of the number
    # **************************************************************************
    Factors = [i for i in range(1, x+1) if x % i == 0]
    #
    # print("Factors     = ", Factors)

    # **************************************************************************
    # copy the list
    # **************************************************************************
    TempList = list(Factors)

    # **************************************************************************
    # loop until the list only has two elements
    # **************************************************************************
    while True:
        if len(TempList) <= Dim:
            # break the loop
            break;
        else:
            # remove the first and the last item in the list
            TempList = TempList[1:-1]

    # **************************************************************************
    # check the list
    # **************************************************************************
    if len(TempList) < Dim:
        Vals = [TempList[0], TempList[0]]
    else:
        Vals = TempList

    # print(x, ": Vals = ", Vals)
    return Vals

# ******************************************************************************
def _CheckFactorList(FactorList):
    # **************************************************************************
    # copy list
    # **************************************************************************
    RemIndices  = []
    RetList     = []

    # **************************************************************************
    # get indices of unwanted element
    # **************************************************************************
    for i in range(len(FactorList)):
        if 1 in FactorList[i]:
            RemIndices.append(i)

    # **************************************************************************
    # get the list
    # **************************************************************************
    for i in range(len(FactorList)):
        if i not in RemIndices:
            RetList.append(FactorList[i])
    # print("RemIndices   = ", RemIndices)
    # print("RetList      = ", RetList)
    return RetList

# ******************************************************************************
def FindFactors(x, Dim=2):
    # **************************************************************************
    # factors for 2D values
    # **************************************************************************
    if Dim == 2:
        return _FindFactors2D(x, Dim=Dim)

    # **************************************************************************
    # factors for 3D values
    # **************************************************************************
    # find the factors of the number
    Factors = [i for i in range(1, x+1) if x % i == 0]

    # **************************************************************************
    # copy the list
    # **************************************************************************
    TempList = list(Factors)

    # **************************************************************************
    # remove the first and the last item in the list
    # **************************************************************************
    TempList    = TempList[1:-1]
    NumFactors  = len(TempList)
    # print("TempList = ", TempList)

    # **************************************************************************
    # generate a list of factors
    # **************************************************************************
    FactorList  = []
    for i in range(NumFactors):
        # check with the symetric element in the list
        j = i + 1
        if TempList[i] == TempList[-j]:
            break
        else:
            X       = TempList[i]
            Y, Z    = _FindFactors2D(TempList[-j], Dim=2)
            FactorList.append([X, Y, Z])

    # **************************************************************************
    # remove invalid elements
    # **************************************************************************
    # print("FactorList   = ", FactorList)
    TempList    = _CheckFactorList(FactorList)
    # print("TempList     = ", TempList, ", length = ", len(TempList))
    # exit()

    # **************************************************************************
    # check the list for valid elements
    # **************************************************************************
    if len(TempList) == 0:
        # set the error message
        ErrMsg = "FindFactors() => invalid factors"
        raise ValueError(ErrMsg)

    # **************************************************************************
    # get the middle list element
    # **************************************************************************
    while True:
        # get the length of the list
        NumVals = len(TempList)
        if NumVals >= 3:
            # remove the first and the last item in the list
            TempList = TempList[1:-1]
        # elif NumVals == 2:
        #     ListVals = TempList[1]
        #     break
        else:
            ListVals = TempList[0]
            break
    Vals = ListVals
    return Vals

# ******************************************************************************
def FindFactorFlag(x, Dim=2):
    # **************************************************************************
    # factors for 2D values
    # **************************************************************************
    if Dim == 2:
        return _FindFactors2D(x, Dim=Dim)

    # **************************************************************************
    # factors for 3D values
    # **************************************************************************
    # find the factors of the number
    Factors = [i for i in range(1, x+1) if x % i == 0]

    # **************************************************************************
    # copy the list
    # **************************************************************************
    TempList = list(Factors)

    # **************************************************************************
    # remove the first and the last item in the list
    # **************************************************************************
    TempList    = TempList[1:-1]
    NumFactors  = len(TempList)
    # print("TempList = ", TempList)

    # **************************************************************************
    # generate a list of factors
    # **************************************************************************
    FactorList  = []
    for i in range(NumFactors):
        # check with the symetric element in the list
        j = i + 1
        if TempList[i] == TempList[-j]:
            break
        else:
            X       = TempList[i]
            Y, Z    = _FindFactors2D(TempList[-j], Dim=2)
            FactorList.append([X, Y, Z])

    # **************************************************************************
    # remove invalid elements
    # **************************************************************************
    # print("FactorList   = ", FactorList)
    TempList    = _CheckFactorList(FactorList)
    # print("TempList     = ", TempList, ", length = ", len(TempList))
    # exit()

    # **************************************************************************
    # check the list for valid elements
    # **************************************************************************
    if len(TempList) == 0:
        # set the error message
        Msg     = "FindFactorsFlag() => invalid factors for <%d>" % x
        print("\033[1;32m%s\033[1;m" %Msg)
        return None, False

    # **************************************************************************
    # get the middle list element
    # **************************************************************************
    while True:
        # get the length of the list
        NumVals = len(TempList)
        if NumVals >= 3:
            # remove the first and the last item in the list
            TempList = TempList[1:-1]
        # elif NumVals == 2:
        #     ListVals = TempList[1]
        #     break
        else:
            ListVals = TempList[0]
            break
    Vals = ListVals
    return Vals, True

# ******************************************************************************
def _LinearTransformingNumpy(Data, YRange, Alpha=None, Beta=None, BiasFlag=True, Verbose=False):
    """
    this function transforms Datasets from x-range to y-range using the function:
        y = alpha * x + beta

        If BiasFlag is false, the offset beta is removed
    """
    # **************************************************************************
    # set the function name
    # **************************************************************************
    FunctionName = "LinearTransforming()"

    # **************************************************************************
    # check the flag
    # **************************************************************************
    if Verbose:
        # display the message
        Msg = "...%-25s: linearly transforming Datasets inputs with bias = <%s>..." % \
                (FunctionName, str(BiasFlag))
        print(Msg)

    # **************************************************************************
    # check apha and  beta
    # **************************************************************************
    if (Alpha is None) or (Beta is None):
        # x and y ranges
        xmin    = np.amin(Data)
        xmax    = np.amax(Data)
        ymin, ymax = YRange

        # **********************************************************************
        # calculate slope (alpha) and y-intercept (beta)
        # **********************************************************************
        Alpha   = (ymax - ymin) / (xmax - xmin)
        Beta    = ymax - Alpha * xmax

    # **************************************************************************
    # check the flag
    # **************************************************************************
    if BiasFlag:
        # transforming Datasets using y = alpha * x + beta
        return np.add(np.multiply(Data, Alpha), Beta), Alpha, Beta
    else:
        # reset the beta
        Beta    = 0.0

        # transforming Datasets using y = alpha * x
        return np.multiply(Data, Alpha), Alpha, Beta

# ******************************************************************************
def _LinearTransformingTorch(Data, YRange, Alpha=None, Beta=None, BiasFlag=True, Verbose=False):
    """
    this function transforms Datasets from x-range to y-range using the function:
        y = alpha * x + beta

        If BiasFlag is false, the offset beta is removed
    """
    # **************************************************************************
    # set the function name
    # **************************************************************************
    FunctionName = "LinearTransformingTorch()"

    # **************************************************************************
    # check the flag
    # **************************************************************************
    if Verbose:
        # display the message
        Msg = "...%-25s: linearly transforming Datasets inputs with bias = <%s>..." % \
                (FunctionName, str(BiasFlag))
        print(Msg)

    # **************************************************************************
    # check apha and  beta
    # **************************************************************************
    if (Alpha is None) or (Beta is None):
        # x and y ranges
        xmin    = torch.min(Data)
        xmax    = torch.max(Data)
        ymin, ymax = YRange

        # **********************************************************************
        # calculate slope (alpha) and y-intercept (beta)
        # **********************************************************************
        Alpha   = (ymax - ymin) / (xmax - xmin)
        Beta    = ymax - Alpha * xmax

    # **************************************************************************
    # check the flag
    # **************************************************************************
    if BiasFlag:
        # transforming Datasets using y = alpha * x + beta
        return torch.add(torch.multiply(Data, Alpha), Beta), Alpha, Beta
    else:
        # reset the beta
        Beta    = 0.0

        # transforming Datasets using y = alpha * x
        return torch.multiply(Data, Alpha), Alpha, Beta

# ******************************************************************************
def LinearTransforming(Data, YRange, Alpha=None, Beta=None, BiasFlag=True, Verbose=False):
    return _LinearTransformingTorch(Data, YRange, Alpha=Alpha, Beta=Beta, \
            BiasFlag=BiasFlag, Verbose=Verbose)

    # if TorchFlag:
    #     return _LinearTransformingTorch(Data, YRange, Alpha=Alpha, Beta=Beta, \
    #             BiasFlag=BiasFlag, Verbose=Verbose)
    # else:
    #     return _LinearTransformingNumpy(Data, YRange, Alpha=Alpha, Beta=Beta, \
    #             BiasFlag=BiasFlag, Verbose=Verbose)

# ******************************************************************************
def MakeEvenNumber(ANumber):
    if (ANumber % 2) == 0:
        # it it even number
        RetVal  = ANumber
    else:
        RetVal  = ANumber + 1
    return RetVal

# ******************************************************************************
def InterQuartileRange(Data):
    """
    Inter Quartile Range approach to finding the outliers is the most commonly
        used and most trusted approach used in the research field.
        IQR = Quartile3 – Quartile1
        upper limit = Q3 +1.5*IQR
        lower limit = Q1 – 1.5*IQR
        Outliers < lower limit, Outliers > upper limit
        https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/
    """
    # **************************************************************************
    # calculate quantiles 1 and 3
    # **************************************************************************
    Q1  = np.percentile(Data, 25, method="midpoint")
    Q3  = np.percentile(Data, 75, method="midpoint")
    IQR = Q3 - Q1

    # **************************************************************************
    # calculate the upper and lower limits
    # **************************************************************************
    Upper   = Q3 + 1.5*IQR
    Lower   = Q1 - 1.5*IQR

    # **************************************************************************
    # find and remove outliers from Datasets
    # **************************************************************************
    Outliers    = [x for x in Data if (x < Lower ) or (x > Upper)]
    if len(Outliers) > 0:
        Outliers    = np.asarray(Outliers)
        ValidData   = np.setdiff1d(Data, Outliers)
    else:
        ValidData   = Data

    return np.average(ValidData), np.std(ValidData)

# ******************************************************************************
def RemoveOutlier(Data, m=2.0):
    # calculate the absolute difference from median
    AbsData     = np.abs(Data - np.median(Data))

    # calculate the median value
    MedianVal   = np.median(AbsData)

    # calculate the ratio
    s = AbsData/(MedianVal if MedianVal else 1.0)
    return Data[s<m]

# ******************************************************************************
def rmse(Data, Target):
    # calculate square error
    SquareError = np.square(np.subtract(Target.flatten(), Data.flatten()))
    return np.sqrt(np.mean(SquareError))

# ******************************************************************************
def nrmse(Data, Target, RMSE=None):
    """
    using the formula:
        NRMSE   = RMSE / (ymax - ymin)
    """
    if RMSE is None:
        # calculate rmse
        RMSE    = rmse(Data, Target)

    # get the max, min, and mean values of Datasets
    Ymax    = np.amax(Data)
    Ymin    = np.amin(Data)
    # Ymean   = np.mean(Data)
    return RMSE / (Ymax - Ymin)

# ******************************************************************************
def mse(Data, Target):
    # calculate square error
    SquareError = np.square(np.subtract(Target.flatten(), Data.flatten()))
    return np.mean(SquareError)

# ******************************************************************************
def nmse(Data, Target):
    """
    using the formula:
        NMSE    =  mean ((Target - Data)^2 / (mean(Target) * mean(Data)))
    """
    SquareError     = np.square(np.subtract(Target.flatten(), Data.flatten()))
    MeanTarget      = np.mean(Target)
    MeanData        = np.mean(Data)
    return np.mean(SquareError / (MeanTarget * MeanData))

# ******************************************************************************
if __name__ == '__main__':
    # import module
    import matplotlib.pyplot as plt

    # set test points
    t1 = 0.2
    y1 = -1.1
    t2 = 0.8
    y2 = 0.6
    X1 = [t1, y1]
    X2 = [t2, y2]
    Num = 6

    x1  = 1
    x2  = 4
    y1  = 3
    y2  = 6
    Dim = 3
    # m, b    = LinearEquationCoefs([x1, y1], [x2, y2])
    #
    # print("m = ", m, ", b = ", b)
    # exit()

    # # get the linear points
    # x, y, dx = LinearPoints(X1, X2, Num=Num)

    # Vals = FindFactors(10, Dim=Dim)
    # print("10 = ", Vals)
    # Vals = FindFactors(11, Dim=Dim)
    # print("11 = ", Vals)
    # Vals = FindFactors(25, Dim=Dim)
    # print("25 = ", Vals)
    # Vals = FindFactors(50, Dim=Dim)
    # print("50 = ", Vals)
    # Vals = FindFactors(65, Dim=Dim)
    # print("65 = ", Vals)
    # Vals = FindFactors(80, Dim=Dim)
    # print("80 = ", Vals)
    # Vals = FindFactors(100, Dim=Dim)
    # print("100 = ", Vals)
    # Vals = FindFactors(250, Dim=Dim)
    # print("250 = ", Vals)
    #
    # Vals = FindFactors(275, Dim=Dim)
    # print("275 = ", Vals)

    Vals = FindFactors(300, Dim=Dim)
    print("300 = ", Vals)
    exit()


    # print("X1 = ", X1)
    # print("X2 = ", X2)
    # print("x  = ", x)
    # print("y  = ", y)

    # expanding the values of y
    Num  = 3
    # Type = "linear"
    Type = "cubic"
    x1, y1, dx1 = InterpolateY(x, y, Num=Num, Type=Type)

    # print("x1  = ", x1)
    # print("y1  = ", y1)

    # set the font family
    font = {"family": "Times New Roman", "size": 14}
    plt.rc('font', **font)  # pass in the font dict as kwargs

    Fig = plt.figure("Line, Num = %d" % (Num))
    plt.grid(linestyle="dotted")
    plt.plot(x, y, "o", label="line")
    plt.plot(x1, y1, "s-", label="linear")
    # plt.xlabel("Time (us)")
    # plt.ylabel("State Variable")
    plt.axis("tight")
    plt.legend(loc="best")
    plt.show()
