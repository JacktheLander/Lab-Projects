import torch, os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ******************************************************************************
# Convolution Neural Network
# ******************************************************************************
class ConvNeuralNet(nn.Module):

    def __init__(self, MumClasses):
        super(ConvNeuralNet, self).__init__()
        self.ConvLayer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.ConvLayer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.MaxPool1   = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.ConvLayer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.ConvLayer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.MaxPool2   = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.FC1    = nn.Linear(1600, 128)
        self.ReLU1  = nn.ReLU()
        self.FC2    = nn.Linear(128, MumClasses)

    # **************************************************************************
    # Progresses data across layers
    # **************************************************************************
    def forward(self, x):
        out = self.ConvLayer1(x)
        out = self.ConvLayer2(out)
        out = self.MaxPool1(out)

        out = self.ConvLayer3(out)
        out = self.ConvLayer4(out)
        out = self.MaxPool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.FC1(out)
        out = self.ReLU1(out)
        out = self.FC2(out)
        return out

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"

    # **************************************************************************
    # Define relevant variables for the ML task
    # **************************************************************************
    BatchSize   = 64
    MumClasses  = 10
    LearningRate = 0.001
    NumEpochs  = 20
    Classes    = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
            'ship', 'truck')

    # **************************************************************************
    # Device will determine whether to run the training on GPU or CPU.
    # **************************************************************************
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **************************************************************************
    # Use transforms.compose method to reformat images for modeling,
    # and save to variable all transform for later use
    # **************************************************************************
    AllTransforms = transforms.Compose([transforms.Resize((32,32)),\
            transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],\
            std=[0.2023, 0.1994, 0.2010])])

    # **************************************************************************
    # Create Training dataset
    # **************************************************************************
    TrainDataset = torchvision.datasets.CIFAR10(root='./data', train=True,
            transform=AllTransforms, download=True)

    # **************************************************************************
    # Create Testing dataset
    # **************************************************************************
    TestDataset = torchvision.datasets.CIFAR10(root='./data', train=False,
            transform=AllTransforms, download=True)

    # **************************************************************************
    # Instantiate loader objects to facilitate processing
    # **************************************************************************
    TrainLoader = torch.utils.data.DataLoader(dataset=TrainDataset, batch_size= BatchSize,
            shuffle=True, num_workers=2)

    TestLoader = torch.utils.data.DataLoader(dataset=TestDataset, batch_size=BatchSize,
            shuffle=True, num_workers=2)

    # **************************************************************************
    # create a convolution neural network
    # **************************************************************************
    CNNModel = ConvNeuralNet(MumClasses)
    CNNModel.to(Device)

    # **************************************************************************
    # Set Loss function with criterion
    # **************************************************************************
    Criterion = nn.CrossEntropyLoss()

    # **************************************************************************
    # Set optimizer with optimizer
    # **************************************************************************
    Optimizer = torch.optim.SGD(CNNModel.parameters(), lr=LearningRate, weight_decay=0.005, \
            momentum=0.9)

    # **************************************************************************
    # total step
    # **************************************************************************
    TotalStep = len(TrainLoader)

    # **************************************************************************
    # We use the pre-defined number of Epochs to determine how many iterations
    # to train the network on
    # **************************************************************************
    for Epoch in range(NumEpochs):
        # **********************************************************************
    	# Load in the data in batches using the TrainLoader object
        # **********************************************************************
        for i, (Images, Labels) in enumerate(TrainLoader):
            # ******************************************************************
            # Move tensors to the configured device
            # ******************************************************************
            Images = Images.to(Device)
            Labels = Labels.to(Device)

            # ******************************************************************
            # Forward pass
            # ******************************************************************
            Outputs = CNNModel(Images)

            # ******************************************************************
            # Backward and optimize
            # ******************************************************************
            Loss    = Criterion(Outputs, Labels)
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()

        # **********************************************************************
        # display the training results
        # **********************************************************************
        print("...Epoch [%d/%d], Loss: %.4f" % (Epoch+1, NumEpochs, Loss.item()))

    # **************************************************************************
    # testing network
    # **************************************************************************
    with torch.no_grad():
        # **********************************************************************
        # initializing
        # **********************************************************************
        Correct = 0
        Total   = 0

        # **********************************************************************
        # testing CNN
        # **********************************************************************
        for Images, Labels in TestLoader:
            # ******************************************************************
            # Move tensors to the configured device
            # ******************************************************************
            Images = Images.to(Device)
            Labels = Labels.to(Device)

            # ******************************************************************
            # get the outputs of the trained CNN
            # ******************************************************************
            Outputs = CNNModel(Images)

            # ******************************************************************
            # Move tensors to the configured device
            # ******************************************************************
            _, Predicted = torch.max(Outputs.data, 1)
            Total   += Labels.size(0)
            Correct += (Predicted == Labels).sum().item()

    # **************************************************************************
    # display the performance of CNN
    # **************************************************************************
    Perf    = 100.0 * Correct / Total
    print('Accuracy of the network on the <10000> test images: %.4f %%' % Perf)

    # **************************************************************************
    # display the performance for each class
    # **************************************************************************
    # prepare to count Predictions for each class
    CorrectPred = {ClassName: 0 for ClassName in Classes}
    TotalPred = {ClassName: 0 for ClassName in Classes}

    # **************************************************************************
    # again no gradients needed
    # **************************************************************************
    with torch.no_grad():
        for data in TestLoader:
            Images, Labels = data

            # ******************************************************************
            # Move tensors to the configured device
            # ******************************************************************
            Images = Images.to(Device)
            Labels = Labels.to(Device)

            # ******************************************************************
            # get the outputs of the trained CNN
            # ******************************************************************
            Outputs = CNNModel(Images)
            _, Predictions = torch.max(Outputs, 1)
            # collect the correct Predictions for each class
            for Label, Prediction in zip(Labels, Predictions):
                if Label == Prediction:
                    CorrectPred[Classes[Label]] += 1
                TotalPred[Classes[Label]] += 1

    # **************************************************************************
    # print accuracy for each class
    # **************************************************************************
    for ClassName, CorrectCount in CorrectPred.items():
        Accuracy = 100 * float(CorrectCount) / TotalPred[ClassName]
        print('...Accuracy for class <%6s>:  is %.2f' % (ClassName, Accuracy))
