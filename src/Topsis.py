import numpy as np
import pandas as pd

class Topsis():
    def __init__(self, X, W, Types):
        self.X = X
        self.W = W
        self.Types = Types
        self.m,self.n = X.shape 
        
    def step_1(self):
        X_norm = np.zeros((self.m, self.n))
        
        for i in range(self.m):
            for j in range(self.n):
                X_norm[i][j] = self.X.iloc[i][j] / np.sqrt(np.sum(self.X ** 2, axis = 0)[j])
                
        return X_norm

    def Create_weighted_Normalized_Decision_Matrix(self):
        X_norm = self.step_1()
        
        X_norm_weight = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                X_norm_weight[i][j] = X_norm[i][j] * self.W[j]
                
        return X_norm_weight

    def Finding_Ideal_of_criteria(self):
        X_norm_weight = self.Create_weighted_Normalized_Decision_Matrix()
        
        ideal_negative = np.zeros(self.n)
        ideal_positive = np.zeros(self.n)
        for i in range((self.n)-1):
            if (self.Types[i] == 1):
                ideal_negative[i] = np.max(X_norm_weight[:,i] , axis = 0)
                ideal_positive[i] = np.min(X_norm_weight[:,i] , axis = 0)
            else:
                ideal_negative[i] = np.min(X_norm_weight[:,i] , axis = 0)
                ideal_positive[i] = np.max(X_norm_weight[:,i] , axis = 0)
                
        return ideal_positive, ideal_negative
    
    def calculate_distance_from_ideal(self):
        ideal_positive, ideal_negative = self.Finding_Ideal_of_criteria()
        X_norm_weight = self.Create_weighted_Normalized_Decision_Matrix()
        
        dis_positive = np.zeros((self.m, self.n))
        dis_negative = np.zeros((self.m, self.n))
        alt_positive = np.zeros(self.m)
        alt_negative = np.zeros(self.m)
        for i in range(self.m):
            for j in range(self.n):
                dis_positive[i,j] = (X_norm_weight[i,j] - ideal_positive[j]) ** 2
                dis_negative[i,j] = (X_norm_weight[i,j] - ideal_negative[j]) ** 2 
        alt_positive = np.sqrt(np.sum(dis_positive , axis = 1))
        alt_negative = np.sqrt(np.sum(dis_negative , axis = 1))
        
        return alt_positive, alt_negative
    
    
    def final_score(self):
        alt_positive, alt_negative = self.calculate_distance_from_ideal()
        
        final_score = []
        for i in range(self.m):
            final_score.append((alt_negative[i] / (alt_positive[i] + alt_negative[i])))
            
        return final_score
    
    def final_ranking(self):
        final_score = self.final_score()
        
        Name = np.array([i for i in range(1, self.m + 1)])
        Final_result = pd.DataFrame({'final score':final_score})
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