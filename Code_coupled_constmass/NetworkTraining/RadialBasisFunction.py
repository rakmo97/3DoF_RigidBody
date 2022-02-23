import numpy as np


class RBF:
    
    def __init__(self, xData, yData, order=[], function='thin_plate', mahalanobis=True):
        
        self.xData = xData
        self.yData = yData
        
        self.mahalanobis = mahalanobis
        self.numOutputs = yData.shape[1]
        
        if self.mahalanobis:
            self.S = np.cov(self.xData.T)
            
        if type(function) == str:
            if function == 'thin_plate':
                self.function = lambda r: r**2 * np.log(r)
            
        elif callable(function):
            self.function = function
        
        else:
            raise Exception('Illegal function input to RBF object')
        
        if not order or order==xData.shape[0]: # assume full order if order is empty
            self.order = xData.shape[0]   
            self.FullFit()
            
        # else:
        #     self.order = order
        #     LowOrderFit(order)
    
    def FullFit(self):
        
        phi = self.function
        
        self.w = np.empty([self.order,self.numOutputs])
        self.Phi = np.empty([self.xData.shape[0],self.xData.shape[0]])
        
        
        for i in range(self.xData.shape[0]):
            for j in range(self.xData.shape[0]):
                
                if self.mahalanobis:
                    r = np.sqrt((self.xData[i] - self.xData[j]).T @ np.linalg.inv(self.S) @ (self.xData[i] - self.xData[j]))
                else:
                    r = np.sqrt((self.xData[i] - self.xData[j]).T @ (self.xData[i] - self.xData[j]))
                
                self.Phi[i,j] = phi(r)
                
        for i in range(self.numOutputs):
            self.w[:,i] = np.linalg.inv(self.Phi) @ self.yData[:,i]
            
        
        
        
    # def LowOrderFit(order):
            

    def predict(self, xInput):
        
        yOutput = np.empty([1,self.numOutputs])
        phi = self.function
        xInput = xInput.reshape(1,-1)
        
        for i in range(self.numOutputs):
            s = 0
            for j in range(self.xData.shape[0]):
                if self.mahalanobis:
                    r = np.sqrt((xInput- self.xData[j]).T @ np.linalg.inv(self.S) @ (xInput - self.xData[j]))
                else:
                    r = np.sqrt((xInput - self.xData[j]).T @ (xInput - self.xData[j]))
                
                s += self.w[j]*phi(r)
            
            yOutput[i] = s
                
        
        return yOutput
            
