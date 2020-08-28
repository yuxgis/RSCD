class Config():
    def __init__(self):
        pass
    @staticmethod
    def getConfigOption():
        option = {

            'lr':0.001,
            'istrain':True,
            'device':"cpu",
            'batch_size':5

        }
        return option




