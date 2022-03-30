from pandas import DataFrame
from scipy.stats import uniform
from scipy.stats import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.DataLoader import get_FI_score

train_data = get_FI_score("PG")
# train_data = get_FI_score("AC")
train_df = pd.DataFrame(train_data)
new_score = ['Fi_score']
train_df.columns = new_score
train_df['type'] = 'train'
train_df =train_df.sample(frac=0.01)


test_data = get_FI_score("PG", False)
test_df = pd.DataFrame(test_data)
new_score = ['Fi_score']
test_df.columns = new_score
test_df['type'] = 'test'
test_df =test_df.sample(frac=0.01)



df = pd.concat([train_df, test_df])
print(df)
df_grouped = df.groupby(('type'))
df['ind'] = range(len(df))
"""
# sample data
df = DataFrame({'gene' : ['gene-%i' % i for i in np.arange(10000)],
               'pvalue' : uniform.rvs(size=10000),
               'chromosome' : ['ch-%i' % i for i in randint.rvs(0, 2,size=10000)]})

# -log_10(pvalue)
df['minuslog10pvalue'] = -np.log10(df.pvalue)
df.chromosome = df.chromosome.astype('category')
df.chromosome = df.chromosome.cat.set_categories(['ch-%i' % i for i in range(2)], ordered=True)
df = df.sort_values('chromosome')
# How to plot gene vs. -log10(pvalue) and colour it by chromosome?
df['ind'] = range(len(df))
df_grouped = df.groupby(('chromosome'))
"""

# manhattan plot
fig = plt.figure(figsize=(10,4),dpi=100)
ax = fig.add_subplot(111)
# colors = ["#30A9DE","#EFDC05","#E53A40","#090707"]
colors = ["#30A9DE","#EFDC05"]
x_labels = []
x_labels_pos = []
for num, (name, group) in enumerate(df_grouped):
    group.plot(kind='scatter', x='ind', y='Fi_score',color=colors[num % len(colors)], ax=ax)
    x_labels.append(name)
    x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
# add grid
ax.grid(axis="y",linestyle="--",linewidth=.5,color="gray")
ax.tick_params(direction='in',labelsize=13)
ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels)

ax.set_xlim([0, len(df)])
ax.set_ylim([0, 50])
# x axis label
ax.set_xlabel('Fi_score',size=14)
# plt.savefig('Manhattan Plot in Python.png',dpi=900,bbox_inches='tight',facecolor='white')
plt.show()
