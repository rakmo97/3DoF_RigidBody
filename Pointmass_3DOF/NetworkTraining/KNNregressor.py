import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

class KNNReg:
    
    def __init__(self, xData, yData, k=1, mahalanobis=True, weight_by_distance=True, t_train_data=[], x_test_data=[], y_test_data=[], t_test_data=[]):
        
        self.xData = xData
        self.yData = yData
        self.t_train_data = t_train_data
        
        self.x_test_data = x_test_data
        self.y_test_data = y_test_data
        self.t_test_data = t_test_data
        
        self.mahalanobis = mahalanobis
        self.weight_by_distance = weight_by_distance
        
        self.numOutputs = yData.shape[1]
        self.k = k
    
            

    def predict(self, state_in, print_density_info=True):
        
        mahalanobis = self.mahalanobis
        k = self.k
        
        state_data = self.xData
        ctrl_data = self.yData
        weight_by_distance = self.weight_by_distance
        
        
        if mahalanobis:
            S = np.cov(state_data.T)
            Si = np.linalg.inv(S)
            distances = np.sqrt(np.einsum('ji,jk,ki->i',(state_in-state_data).T,Si,(state_in-state_data).T))
            # for i in range(state_data.shape[0]):
            #     distances[i] = np.sqrt((state_in-state_data[i]).T@Si@(state_in-state_data[i]))
        else:
            distances = np.linalg.norm(state_data - state_in,axis=1)

            
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        if print_density_info:
            np.set_printoptions(precision=3)
            plt.figure
            plt.subplot(121)
            plt.plot(distances,'.')
            plt.xlabel('Index [-]')
            plt.ylabel('Mahalanobis Distance [-]')
            plt.subplot(122)
            plt.hist(distances)
            plt.xlabel('Mahalanobis Distance [-]')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.title("Distance Distribution for state {} | mean {:.3f} | std {:.3f}".format(state_in,mean_distance,std_distance))
            plt.show()
        
            print('---')
            print("State: {}".format(state_in))
            print("Mean Distances: {}".format(mean_distance))
            print("Std Distances:  {}".format(std_distance))
        
        
        if k==1:
            low_i = np.argmin(distances)   
            ctrl_out = ctrl_data[low_i]
        else:
            kSmallest_i = sorted(range(len(distances)), key = lambda sub: distances[sub])[:k]
            kSmallest = ctrl_data[kSmallest_i]
                        
            if (kSmallest < 1e-5).any():
                ctrl_out = kSmallest[0]                
            
            else:
                if weight_by_distance:
                    weights_for_avg = 1.0/distances[kSmallest_i]
                    ctrl_out = np.average(kSmallest,axis=0, weights=weights_for_avg)
                    
                else:
                    ctrl_out = np.mean(kSmallest,axis=0)
        
        return ctrl_out, mean_distance, std_distance
            
