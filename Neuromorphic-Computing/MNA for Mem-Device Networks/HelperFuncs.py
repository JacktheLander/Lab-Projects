# import modules
import os, sys
from os.path import join, isdir, isfile

# ******************************************************************************
# functions in this module
# ******************************************************************************
def GetWinHomePath():
    # get the username from the environmental variables
    # HomePath    = os.getenv("HOMEPATH")
    # HomeDrive   = os.getenv("HOMEDRIVE")
    HomePath   = os.getenv("USERPROFILE")
    # FolderPath  = join(HomeDrive, HomePath, "Documents")
    FolderPath  = join(HomePath, "Documents")
    return FolderPath

# ******************************************************************************
def GetLinuxHomePath():
    return os.getenv("HOME")

# ******************************************************************************
def GetMacOSHomePath():
    return os.getenv("HOME")

# ******************************************************************************
def GetWinDataPath(SpecialPath=None):
    if SpecialPath is not None:
        DataPath    = join(SpecialPath, "Datasets")
    else:
        DataPath    = join(GetWinHomePath(), "Datasets")
    return DataPath

# ******************************************************************************
def GetLinuxDataPath(SpecialPath=None):
    # **************************************************************************
    # check for special Datasets path
    # **************************************************************************
    if SpecialPath is not None:
        DataPath    = join(SpecialPath, "Datasets")
    else:
        DataPath    = join(os.getenv("HOME"), "Datasets")
    return DataPath

# ******************************************************************************
def GetMacOSDataPath(SpecialPath=None):
    # **************************************************************************
    # check for special Datasets path
    # **************************************************************************
    if SpecialPath is not None:
        DataPath    = join(SpecialPath, "Datasets")
    else:
        DataPath    = join(os.getenv("HOME"), "Datasets")
    return DataPath

# ******************************************************************************
def GetHomePath():
    # **************************************************************************
    # check the platform to import mem-models
    # **************************************************************************
    Platform = sys.platform
    if (Platform == "linux2") or (Platform == "linux"):
        # get the home path
        HomePath        = GetLinuxHomePath()

    elif Platform == "win32":
        HomePath        = GetWinHomePath()

    elif Platform == "darwin":
        HomePath        = GetMacOSHomePath()

    else:
        # format error message
        Msg = "unknown platform => <%s>" % (Platform)
        raise ValueError(Msg)

    return HomePath

# ******************************************************************************
def _GetDataPath(SpecialPath=None):
    # **************************************************************************
    # check for Datasets path
    # **************************************************************************
    Platform = sys.platform
    if (Platform == "linux2") or (Platform == "linux"):
        # get the home path
        DataPath    = GetLinuxDataPath(SpecialPath=SpecialPath)

    elif Platform == "win32":
        DataPath    = GetWinDataPath(SpecialPath=SpecialPath)

    elif Platform == "darwin":
        DataPath    = GetMacOSDataPath(SpecialPath=SpecialPath)

    else:
        # format error message
        Msg = "unknown platform => <%s>" % (Platform)
        raise ValueError(Msg)

    return DataPath

# ******************************************************************************
def _CheckCurrentPath(DatasetName):
    # **************************************************************************
    # check the current path
    # **************************************************************************
    DataPath = join("Datasets", DatasetName)
    return isdir(DataPath), DataPath

# ******************************************************************************
def GetDataPath(DatasetName=None):
    # **************************************************************************
    # set the function name
    # **************************************************************************
    FunctionName    = "Helper:GetDataPath()"

    # **************************************************************************
    # check for dataset in the current folder
    # **************************************************************************
    Flag, DataPath  = _CheckCurrentPath(DatasetName)
    if Flag:
        return DataPath
    else:
        # display the information
        Msg = "...%-25s: dataset is not in <%s>, check standard path" % (FunctionName, \
                DataPath)
        print(Msg)

    # **************************************************************************
    # check the Datasets path
    # **************************************************************************
    DataPath    = _GetDataPath()
    if not isdir(DataPath):
        # **********************************************************************
        # format error message
        # **********************************************************************
        Msg = "<%s>: invalid Datasets path => <%s>" % (FunctionName, DataPath)
        raise ValueError(Msg)
    else:
        # display the information
        Msg = "...%-25s: found dataset in <%s>" % (FunctionName, DataPath)
        print(Msg)

    # **************************************************************************
    # set the Datasets path
    # **************************************************************************
    SpecialFlag = "opt" in DataPath
    if (DatasetName is not None) and (not SpecialFlag):
        return join(DataPath, DatasetName)
    else:
        return DataPath

# ******************************************************************************
def GetModelPath():
    return join(GetHomePath(), "PythonMemModels")

# ******************************************************************************
def GetCommonFuncPath():
    return join(GetHomePath(), "MemCommonFuncs")

# ******************************************************************************
def GetScriptPath():
    return join(GetHomePath(), "PythonScripts")

# ******************************************************************************
def _GetTempPath(FileName=None):
    # **************************************************************************
    # check the platform to import mem-models
    # **************************************************************************
    Platform = sys.platform
    if (Platform == "linux2") or (Platform == "linux"):
        # **********************************************************************
        # Temporary path
        # **********************************************************************
        TempPathOne = "/tmp"
        TempPathTwo = "/stash/tlab/tmp"

        # **********************************************************************
        # Temporary path
        # **********************************************************************
        TempPath    = "/tmp"
        if isfile(join(TempPathOne, FileName)):
            TempPath = TempPathOne
        elif isfile(join(TempPathTwo, FileName)):
            TempPath = TempPathTwo
        else:
            TempPath = "/tmp"

    # **************************************************************************
    # temporary path on windows
    # **************************************************************************
    elif Platform == "win32":
        TempPath    = GetWinDataPath()

    # **************************************************************************
    # temporary path on macOS
    # **************************************************************************
    elif Platform == "darwin":
        TempPath    = GetMacOSDataPath()

    else:
        # format error message
        Msg = "unknown platform => <%s>" % (Platform)
        raise ValueError(Msg)

    return TempPath

# ******************************************************************************
def GetUserName():
    # **************************************************************************
    # check the platform to import mem-models
    # **************************************************************************
    Platform = sys.platform
    if (Platform == "linux2") or (Platform == "linux"):
        UserName    = os.getenv("USER")
    elif Platform == "win32":
        UserName    = os.getenv("USERNAME")
    elif Platform == "darwin":
        UserName    = os.getenv("USER")
    else:
        # format error message
        Msg = "unknown platform => <%s>" % (Platform)
        raise ValueError(Msg)

    return UserName

# ******************************************************************************
def CheckTempPath(FileName):
    # **************************************************************************
    # Get the temporary path file
    # **************************************************************************
    TempPath     = _GetTempPath(FileName)
    TempPathFile = join(TempPath, FileName)

    # **************************************************************************
    # check the file status
    # **************************************************************************
    if not isfile(TempPathFile):
        TempPath = None

    return TempPath

# ******************************************************************************
if __name__ == '__main__':
    # **************************************************************************
    # printout the path
    # **************************************************************************
    print("ModelPath        = ", GetModelPath())
    print("CommonFuncPath   = ", GetCommonFuncPath())
    print("GetScriptPath    = ", GetScriptPath())
    exit()
