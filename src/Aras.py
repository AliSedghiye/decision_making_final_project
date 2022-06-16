import numpy as np
import pandas as pd

class Aras:
    def __init__(self, x, w, Types):
        self.x = x
        self.w = w
        self.types = Types
        
        self.m, self.n = x.shape

        
    def new_matrix(self):
        X0 = np.zeros(self.n)
        for i in range(self.m):
            for j in range((self.n)-1):
                if (self.types[j] == 'max'):
                    X0[j] = np.max(self.x, axis = 0)[j]
                else:
                    X0[j] = np.min(self.x, axis = 0)[j]
        X_new = np.vstack([self.x,X0]) 
        return X_new
        
    def normalize(self):
        x_new = self.new_matrix()
        
        X_norm = np.zeros((self.m + 1, self.n))
        for i in range(self.m + 1):
            for j in range(self.n):
                X_norm[i][j] = x_new[i][j] / np.sum(x_new, axis = 0)[j]
        return X_norm      
    
    def weighted_Normalized_Decision_Matrix(self):
        X_norm = self.normalize()
        
        X_norm_weight = np.zeros((self.m + 1, self.n))
        for i in range(self.m + 1):
            for j in range(self.n):
                X_norm_weight[i][j] = X_norm[i][j] * self.w[j]
        return X_norm_weight
                
    def finding_A0(self):
        X_norm_weight = self.weighted_Normalized_Decision_Matrix()
        
        A0 = np.zeros(self.n)
        for i in range(self.m):
            for j in range((self.n)-1):
                if (self.types[j] == 'max'):
                    A0[j] = np.max(X_norm_weight, axis = 0)[j]
                else:
                    A0[j] = np.min(X_norm_weight, axis = 0)[j]           
        return A0
    
    def final_score(self):
        A0 = self.finding_A0()
        X_norm_weight = self.weighted_Normalized_Decision_Matrix()
        
        final_score = []
        for i in range(self.m):
                final_score.append(np.sum(X_norm_weight[i,:]) / np.sum(A0))
                
        Name = np.array([i for i in range(1, self.m + 1)])
        Final_result = pd.DataFrame({'final score':final_score})
        Final_result.index = Name
        Final_result['Ranking'] = Final_result['final score'].rank(ascending = False)
        
        Final_result = Final_result.sort_values(by='Ranking')
        Final_result['alt'] = Final_result.index
        Final_result.reset_index(drop=True, inplace=True)
        alternative = Final_result['alt'].values
        
        return alternative