# 📊 Correlation Analysis: Age vs Net Worth

## 👤 Prepared by:
Benjelyn Reves Patiag  
MSE803 Data Analytics  

---

# 📌 1. About this Task

This task is about **correlation analysis**.
Tto check:

👉 Is there relationship between **age** and **net worth?**  
👉 If age increase, net worth also increase or not?

use:
- Python  
- Pandas  
- Scipy  
- Matplotlib  

---

# 📂 2. Dataset

File used: `age_networth.csv`

It has 2 columns:

- **Age** → age of person  
- **Net Worth** → how much money person have  

---

# ⚙️ 3. Tools Used

### 🐼 Pandas
used to load and clean data  

### 📊 Scipy
used to calculate correlation  

### 🎨 Matplotlib
used to draw graph (scatter plot)

---

# 🧮 4. Code Used

```python
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
```

---

# 📊 5. Result

- Correlation (r) = **0.88**
- P-value = **0.0007**

---

# 🧠 6. Explanation

r = 0.88 means strong positive relationship  

When age go up, net worth also go up  

---

# 📈 7. Graph Explanation

Graph shows upward trend  

young people → less money  
older people → more money  

---

# ⚠️ 8. Important Note

Not 100% perfect  

some young people rich  
some old people not rich  

---

# 📉 9. P-value Meaning

p-value = 0.0007 (< 0.05)  

means result is real, not random  

---

# 🎯 10. Final Conclusion

Older people usually have more money  

---

# 📌 11. Learning

- learn correlation  
- learn scipy  
- learn visualization  

---
