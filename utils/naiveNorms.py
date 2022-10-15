import numpy as np
from numpy import linalg as LA

class NaiveNorms:
    def __init__(self, weight1, weight2, bruteforce = True):
        #weight1 is the weight matrix from the input to hidden layer
        #weight2 is from the hidden layer to the output
        assert weight1.shape[0] == weight2.shape[1]
        self.weight1 = weight1
        self.weight2 = weight2
        self.processWeights()
        if bruteforce and self.hiddenSize < 20:
            self.activations = self.allAct()
        else:
            self.activations = None

    def processWeights(self):
        #weight1 size: hidden * input; weight2 size: output * hidden
        hiddenSize = self.weight2.shape[1]
        outSize = self.weight2.shape[0]
        self.hiddenSize = hiddenSize
        self.outSize = outSize
        self.propWeights = []
        for i in range(outSize):
            weight = self.weight1.copy()
            vector = self.weight2[i]
            for j in range(vector.size):
                weight[j,:] *= vector[j]
            self.propWeights.append(weight)

    def computeNorm(self, weight, activation, mode):
        gradient = np.matmul(activation, weight)
        return LA.norm(gradient, mode)

    def _generateAct(self, n, length, prefix, activations):
        if n == length:
            activations.append(prefix.copy())
            return
        self._generateAct(n+1, length, prefix+[1], activations)
        self._generateAct(n+1, length, prefix+[0], activations)

    def allAct(self):
        activations = []
        self._generateAct(0, self.hiddenSize, [], activations)
        return activations

    def maxDimNorm(self, dim, mode):
        #the weight matrix is of size input * hidden
        if not self.activations:
            print("Too many hidden units. Do not try brute force.")
            return -1
        norm = 0
        for act in self.activations:
            norm = max(norm, self.computeNorm(self.propWeights[dim], act, mode))
        return norm

    def BFNorms(self, mode):
        norms = []
        for i in range(self.outSize):
            norms.append(self.maxDimNorm(i, mode))
        return norms

    def BFJacobian(self, mode):
        norm = 0
        act = [1]*self.hiddenSize
        for act in self.activations:
            dact = np.diag(np.array(act))
            mat = np.matmul(self.weight2, dact)
            mat = np.matmul(mat, self.weight1)
            curr = LA.norm(mat, mode)
            #print(curr)
            if curr > norm:
                norm = curr
        return norm

    def maxSigVal(self, mat):
        _, s, _ = LA.svd(mat)
        return s[0]

    def maxInfNorm(self, mat):
        return LA.norm(mat, np.inf)

    def SingularNorms(self):
        norm1 = self.maxSigVal(self.weight1)
        print("SV of first w is: ", norm1)
        norms = []
        for i in range(self.outSize):
            norms.append(norm1 * self.maxSigVal(np.array([self.weight2[i]])))
        return norms

    def InfNorms(self):
        norm1 = self.maxInfNorm(self.weight1)
        print("InfNorm of first w is: ", norm1)
        norms = []
        for i in range(self.outSize):
            norms.append(norm1 * self.maxInfNorm(np.array([self.weight2[i]])))
        return norms