import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


class CorrelationAnalysis:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Data loaded")
    
    def clean_data(self):
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(" ", "_")
        print("Columns:", self.df.columns)
    
    def calculate_correlation(self):
        r, p_value = pearsonr(self.df['age'], self.df['net_worth'])
        print("Pearson r:", r)
        print("P-value:", p_value)
        return r, p_value
    
    def visualize(self):
        x = self.df['age']
        y = self.df['net_worth']
        
        # create scatter plot
        plt.figure()
        plt.scatter(x, y)
        
        # create regression line (best fit)
        m, b = np.polyfit(x, y, 1)   # m = slope, b = intercept
        plt.plot(x, m*x + b)
        
        plt.xlabel('Age')
        plt.ylabel('Net Worth')
        plt.title('Age vs Net Worth with Relationship Line')
        
        plt.show()


# MAIN
if __name__ == "__main__":
    
    analysis = CorrelationAnalysis('age_networth.csv')
    
    analysis.load_data()
    analysis.clean_data()
    analysis.calculate_correlation()
    analysis.visualize()