import numpy as np
import pandas as pd

class vikor():
    def __init__(self, X, W, Types):
        self.w = W
        self.types = Types
        self.X = X
        self.m,self.n = X.shape
        
    def step_1(self):
        
        f = np.zeros((self.n,2))
        for i in range ((self.n)-1):
            if (self.types[i] == 'max'):
                f[i,0] = self.X.max(0)[i]
                f[i,1] = self.X.min(0)[i]
            else:
                f[i,0] = self.X.min(0)[i]
                f[i,1] = self.X.max(0)[i]
                
        return f
                
    def step_2(self):
        f = self.step_1()
        
        s = np.zeros(self.m)
        r = np.zeros(self.m)
        for i in range(self.m):
            k = 0
            o = 0
            for j in range(self.n):
                k = k + self.w[j] * ((f[j,0] - self.X.iloc[i][j] ) / (f[j,0] - f[j,1]))
                u = self.w[j] * ((f[j,0] - self.X.iloc[i][j] ) / (f[j,0] - f[j,1]))
                if (u > o):
                    o = u
                    r[i] = round(o, 3)
                else:
                    r[i] = round(o, 3)
            s[i] = round(k, 3)
        return s, r
            
    def step_3(self):
        s, r = self.step_2()
        
        Q = np.zeros(self.m)
        nou = (self.n + 1) / (2 * self.n)
        for i in range(self.m):
             Q[i] = round((nou * (s[i] - min(s)) / (max(s) - min(s))) + 
                ((1 - nou) * (r[i] - min(r)) / (max(r) - min(r))), 3)
                
        return Q, s, r
                
                
    def step_4(self):
        Q, s, r = self.step_3()
        
        dic = {
            'alt' : range(1, len(self.X)+1), 
            'S' : s,
            'R' : r,
            'Q' : Q,
        }
        
        df = pd.DataFrame(dic).set_index('alt')
        df = df.sort_values(by=['Q', 'S', 'R'])
        df['rank'] = range(1, len(df)+1)
        
        return df
    
    def condition(self):
        df = self.step_4()
        
        count = 1
        count_2 = 1
        q_1 = df.iloc[0]
        for i in range(1, len(df)):
            count_2 += 1
            row_2 = df.iloc[i]
            if row_2['Q'] - q_1['Q'] < 1/(len(df)-1) :
                df['rank'].iloc[i] = count


            else : 
                count += 1
                df['rank'].iloc[i] = count_2
                
                
        flag = (df.iloc[0]['S'] == df['S'].min()) or (df.iloc[0]['R'] == df['R'].min())
        if flag == False :
            df['rank'].iloc[1] = df['rank'].iloc[0]
            
        
        return df
                
    
    def final_step(self):
        df = self.condition()
        
        df['alt'] = df.index
        df.reset_index(drop=True, inplace=True)
        
        result = df['alt'].values
        
        return result
