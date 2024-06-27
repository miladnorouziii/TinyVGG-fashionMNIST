from Modules.TinyVGG import *
from Modules.DatasetHandler import *
from colorama import init, Fore, Style
import os 
import torch
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class Main():

    device = "cpu"
    configs = {"epochs": None,
        "validationPercent": None,
        "learningRate": None,
        "testPercent": None,
        "batchSize": 20,
        "optimizerType": None,
        "hiddenUnits": None, 
        "hiddenLayers": None
    }
    trainLoader:DataLoader
    testLoader:DataLoader
    dataset = DatasetHandler()


    def colorText(self, text, color):
        init()
        colorCode = ""
        if color == "g":
            colorCode = "\033[32m"
        elif color == "r":
            colorCode = "\033[31m"
        elif color == "y":
            colorCode = "\033[33m"
        elif color == "c":
            colorCode = "\033[36m"
        elif color == "m":
            colorCode = "\033[35m"
        return f"{colorCode}{text}\033[0m"
    

    def checkHardware(self):
        print(self.colorText("Checking your hardware ...\n", "y"))
        try:
            os.system('nvidia-smi')
        except Exception as e:
            print(self.colorText(f"Error -> {e}\n", "r"))
        if torch.cuda.is_available():
            print(self.colorText("\nCUDA is available.", "g"))
            numberOfGpus = torch.cuda.device_count()
            print(self.colorText(f"Number of available GPUs: {numberOfGpus}", "g"))
            for i in range (numberOfGpus):
                gpuProperties = torch.cuda.get_device_properties(i)
                print(self.colorText(f"GPU{i}: {gpuProperties.name}, (CUDA cores: {gpuProperties.multi_processor_count})", "g"))
            self.device = torch.device("cuda")
        else:
            print(self.colorText("OOps! your GPU doesn't support required CUDA version. Running on CPU ...\n", "r"))
            self.device = torch.device("cpu")
    

    def getUserParams(self):
        self.configs["epochs"] = int(input("\n-> Enter iteration number: "))
        self.configs["learningRate"] = float(input("-> Enter learning rate: "))
        self.configs["batchSize"] = int(input("-> Enter batch size:(default 20): "))
        self.configs["optimizerType"] = input("-> Which optimizer do you want to choose?(SGD/Adam): ")
    

    def accuracyFunc(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100 
        return acc

    
    def startNN(self):
        self.checkHardware()
        self.getUserParams()
        self.trainLoader, self.testLoader = self.dataset.getData(self.configs["batchSize"])
        model = TinyVGG(inputShape=1, 
            hiddenUnits=10, 
            outputShape=10
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer : torch.optim
        if self.configs['optimizerType'] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learningRate'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['learningRate'])
        modelStartTime = timer()
        training_losses = []
        test_losses = []
        epochArray = []
        for epoch in tqdm(range(self.configs['epochs'])):
            train_loss, train_acc = 0, 0
            for batch, (X, y) in enumerate(self.trainLoader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                train_loss += loss
                train_acc += self.accuracyFunc(y_true=y, y_pred=y_pred.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(self.trainLoader)
            train_acc /= len(self.trainLoader)
            training_losses.append(train_loss)
            epochArray.append(epoch)
            print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
            model.eval() 
            with torch.inference_mode():
                test_loss, test_acc = 0, 0
                for batch, (inputs, labels) in enumerate(self.testLoader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss
                    test_acc += self.accuracyFunc(y_true=labels, y_pred=outputs.argmax(dim=1))
                test_loss /= len(self.testLoader)
                test_acc /= len(self.testLoader)
                test_losses.append(test_loss)
                print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
        modelEndTime = timer()
        totalTime = modelEndTime - modelStartTime
        print(f"Train time on {self.device}: {totalTime:.3f} seconds")
        plt.plot(epochArray, torch.Tensor(training_losses).numpy(), label = "Train Loss")
        plt.plot(epochArray, torch.Tensor(test_losses).numpy(), label = "Test Loss")
        plt.title("Loss plots")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()





if __name__ == "__main__":
    main = Main()
    main.startNN()