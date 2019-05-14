import numpy as np
import pandas as pd

def informationGain(parentEntropy, childrensEntropy):
    return parentEntropy-(np.sum(childrensEntropy)/len(childrensEntropy))
        

