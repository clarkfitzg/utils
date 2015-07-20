import matplotlib.pyplot as plt
import pandas as pd

normal = pd.read_csv('normal.csv', header=None)

normal.hist()
plt.savefig('normal_hist.pdf')

# This is what the first bit of data looks like:
with open('normalpts.tex', 'w') as f:
    normal[:5].to_latex(f)

q = normal.quantile([0, 0.25, 0.5, 0.75, 1])

q.to_latex('quantiles.tex')
