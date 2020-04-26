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

        super(PytorchModel, self).__init__()
        self.Model = model 
        self.PerformanceCounter = PerformanceCounter() 

    def Prepare(self):
        pass 

    def Fit(self, generator): 

        criterion = nn.CrossEntropyLoss()
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

                outputs = self.Model(torch.from_numpy(features))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def Test(self, generator):
        pass 

