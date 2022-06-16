import numpy as np
import pandas as pd

from entropy import entropy_shanon
from Aras import Aras
from saw import saw
from Topsis import Topsis
from vikor import vikor

data = pd.read_excel('../data/fastfood.xlsx', sheet_name='Sheet2')
new_data = data.drop('Unnamed: 0', axis=1)


Types = np.array(['min','max','max','max','max','max','max'])
W = entropy_shanon(new_data, Types).Caculate_Weight()


aras_alt = Aras(new_data, W, Types).final_score()
saw_alt = saw(new_data, W, Types).final_step()
Topsis_alt = Topsis(new_data, W, Types).final_step()
vikor_alt = vikor(new_data, W, Types).final_step()

alt_ls = [aras_alt, saw_alt, Topsis_alt, vikor_alt]

class Average:
    def __init__(self, alt):
        self.alt = alt
        self.rank = {'rank' : list(range(1, len(alt)+1))}
    
    def create_df(self):
        d = {'rank' : list(range(1, len(self.alt)+2)), }
        count = 1
        for i in self.alt:
            d[f'alt{count}'] = i
            count += 1
        return pd.DataFrame(d)
    
    def final_rank(self):
        df = self.create_df()
        ls = []
        for i in range(len(df)):
            sum_col = 0
            average = 0
            row = df.iloc[i]
            for j in df.columns[1:]:
                sum_col += row[j]
                average = sum_col/(len(df.columns)-1)
            ls.append(average)
        df['final'] = ls
        df = df.sort_values(by='final')
        return df


final = Average(alt_ls)
final_dataframe= final.final_rank()

print(final_dataframe)

