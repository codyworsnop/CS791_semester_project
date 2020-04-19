from interface import Interface

class IModel(Interface):

    def Prepare(self):
        pass 

    def Fit(self, generator): 
        pass

    def Test(self, generator):
        pass