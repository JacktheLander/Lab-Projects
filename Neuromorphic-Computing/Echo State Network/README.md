# Echo State Network
<img src="ESN.png" />

---
## Training Log Ouput

### **Dataset Initialization**  
==> Instantiating **<Mnist>**... <br>
...MNIST::_CheckPath()      : checking the Datasets path ... <br>
...Helper:GetDataPath()     : found dataset in **<\Neuromorphic Computing\Datasets>** <br>
...MNIST::_BuildDataVectors(): building MNIST training and testing Datasets... <br>
...MNIST::_TrainingAndTestingSets(): loading MNIST Datasets... <br>
...MNIST::_ReadMNISTData()  : loading MNIST Datasets... <br>

---

### **Loading MNIST Data**  
...MNIST::_LoadMNIST()      : loading **<Training>** Datasets from **<\Neuromorphic Computing\Datasets\MNIST>**... <br>
...MNIST::_LoadMNIST()      : loading **<Testing>** Datasets from **<\Neuromorphic Computing\Datasets\MNIST>**... <br>
...MNIST::_SelectSubset()   : set Datasets indices ... <br>
...MNIST::_GetClassGroupIndices(): set class indices for **<(60000,)>**... <br>
...MNIST::_SelectSubset()   : set Datasets indices ... <br>
...MNIST::_GetClassGroupIndices(): set class indices for **<(10000,)>**... <br>
...MNIST::_BuildDataVectors(): saving Datasets to file **<\Neuromorphic Computing\Datasets\MNIST_Train_10000_Test_1000.npz>**... <br>

---

### **Final Dataset Configuration**  
==> Instantiating **<Mnist>**... <br>
...MNIST::__init__()        : **Datasets location** = **\Neuromorphic Computing\Datasets** <br>
...MNIST::__init__()        : **Classes** = 10, **Total trains** = 60000, **Total tests** = 10000 <br>
...MNIST::__init__()        : **DataFile** = **\Neuromorphic Computing\Datasets\MNIST_Train_10000_Test_1000.npz** <br>
...MNIST::__init__()        : **Epochs** = 1, **Num Trains** = 10000, **Num Tests** = 1000, **Vector Length** = 784 <br>
...MNIST::__init__()        : **ScaleVal** = 1.0, **ZeroMeanFlag** = False <br>

---

### **Preprocessing & Training**  
...MNIST::GetDataVectors()  : loading MNIST dataset, **ZeroMean = <False>**... <br>
...MNIST::_NormalInputDataTorch(): loading and scaling MNIST Datasets to **<1.0>**... <br>

---

### **Training the ESN Network**  
...ESN::TrainReservoir()    : train ESN network ... <br>
...ESN::_ReservoirStates()  : get reservoir states ... <br>
**100% (9999 of 9999) |####################| Elapsed Time: 0:00:07 Time:  0:00:07** <br>

---

### **Testing the ESN Network**  
...ESN::TestReservoir()     : test ESN network ... <br>
...ESN::_ReservoirStates()  : get reservoir states ... <br>
**100% (999 of 999) |######################| Elapsed Time: 0:00:00 Time:  0:00:00** <br>

---

### **Final Classification Accuracy**  
**Classification = 92.00%** 🎯 <br>
