import numpy as np
import pandas as pd


class saw():
    def __init__(self, X, W, Types):
        self.X = X
        self.W = W
        self.Types = Types
        self.m, self.n = X.shape
        
    def step_1(self):
        R = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range((self.n)-1):
                if (self.Types[j] == 'max'):
                    R[i,j] = self.X.iloc[i][j] / np.max(self.X, axis = 0)[j]
                else:
                    R[i,j] = np.min(self.X, axis = 0)[j] / self.X.iloc[i][j]
        return R
    
    
    def step_2(self):
        R = self.step_1()
        
        U = np.zeros((self.m,self.n))
        for i in range(self.m):
            for j in range(self.n):
                U[i,j] = R[i,j] * self.W[j]
        score = np.sum(U, axis = 1)
        score = np.ndarray.tolist(score)
        
        return score
        
    def final_ranking(self):
        score = self.step_2()
        
        Name = np.array([i for i in range(1,self.X.shape[0] + 1)])
        Final_result = pd.DataFrame({'final score':score})
        Final_result.index = Name
        Final_result['Ranking'] = Final_result['final score'].rank(ascending = False)  
        
        return Final_result

    def final_step(self):
        df = self.final_ranking()
    
        df = df.sort_values(by='Ranking')
        df['alt'] = df.index
        df.reset_index(drop=True, inplace=True)
        alternative = df['alt'].values
        
        return alternative