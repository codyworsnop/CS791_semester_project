from interface import implements
from IModel import IModel
import torch.nn as nn
import torch.nn.functional as F
from LayerTypes import LayerTypes 
import torch.optim as optim
import numpy as np
import torch
from PerformanceCounter import PerformanceCounter

class PytorchModel(implements(IModel)):

    def __init__(self, model): 

        self.Model = model 
        self.PerformanceCounter = PerformanceCounter()
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model.to(self.Device) #convert the model to use GPU 

    def Prepare(self):
        pass 

    def Fit(self, generator): 

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.Model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(5):

            print("processing epoch:", epoch, "of:", 5)
            self.PerformanceCounter.Start()                 
            batches = len(generator)

            running_loss = 0.0 
            for batch in range(batches): 

                print("processing batch:", batch, "of:", batches, "Elapsed time:", self.PerformanceCounter.Waypoint())

                optimizer.zero_grad()

                features, labels = generator.__getitem__(batch)
                features = np.swapaxes(features, 3, 1) #pytorch expects batch x channel size x width x height 
                features, labels = torch.from_numpy(features).float(), torch.from_numpy(labels).float() #pytorch only wants tensors
                features, labels = features.to(self.Device), labels.to(self.Device)
                
                outputs = self.Model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                print("Accuracy:", self.calculate_accuracy(labels, outputs))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch + 1, running_loss))
                running_loss = 0.0

    def calculate_accuracy(self, labels, prediction):
        rounded_predictions = torch.round(prediction)
        correct = (rounded_predictions == labels).sum().float()
        return correct / (labels.shape[0] * labels.shape[1])

    def Test(self, generator):
        pass 

