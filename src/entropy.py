import pandas as pd
import numpy as np 

class entropy_shanon():
    def __init__(self, decision_matrix, types):
        self.decision_matrix = decision_matrix
        self.types = types
        self.m, self.n = decision_matrix.shape
        
    def Normalization(self):
        column_names = self.decision_matrix.columns
        for col in column_names:
            self.decision_matrix[col] = self.decision_matrix[col] / self.decision_matrix[col].sum()
            
        return self.decision_matrix
            
        
    def Caculate_Weight(self):
        N = self.Normalization().values
        
        entropy_vector = -1/np.log(self.m) * np.sum(N*np.log(N),axis=0)
        entropy_weight = entropy_vector / np.sum(entropy_vector)
        entropy_weight = np.ndarray.tolist(entropy_weight)
    
        return(entropy_weight)