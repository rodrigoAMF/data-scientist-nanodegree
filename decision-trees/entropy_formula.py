import numpy as np

def entropy(m, n):
    # M = number of red balls
    # N = number of blue balls
    pRed = m/(m+n)
    pBlue = n/(m+n)
    
    return (-pRed*np.log2(pRed)-pBlue*np.log2(pBlue))

def multiClassEntropy(elements):
    numberElements = len(elements)
    
    entropy = 0
    for i in range(numberElements):
        pi = elements[i]/np.sum(elements)
        entropy += pi*np.log2(pi)
    
    return -entropy
        

print("2 balls => Entropy:%.3f" % entropy(4, 10))
print("3 balls => Entropy:%.3f" % multiClassEntropy(np.array([8,3,2])))
