from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class DatasetHandler():

    def getData(self, batchSize):
        trainData = datasets.FashionMNIST(root="Dataset", train=True,
            download=True, transform=transforms.ToTensor()
        )
        testData = datasets.FashionMNIST(root="Dataset", train=False,
            download=True, transform=transforms.ToTensor()
        )
        trainDataloader = DataLoader(trainData,
            batch_size=batchSize,  
            shuffle=True
        )
        testDataloader = DataLoader(testData,
            batch_size=batchSize,
            shuffle=False
        )
        train_features_batch, train_labels_batch = next(iter(trainDataloader))
        return trainDataloader, testDataloader

