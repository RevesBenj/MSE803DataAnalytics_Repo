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

df = pd.read_csv('age_networth.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

r, p_value = pearsonr(df['age'], df['net_worth'])

print("Correlation:", r)
print("P-value:", p_value)

plt.scatter(df['age'], df['net_worth'])
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Age vs Net Worth')
plt.show()
```

---

# 📊 5. Result

- Correlation (r) = **0.88**
- P-value = **0.0007**

---

# 🧠 6. Explanation (Simple)

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
