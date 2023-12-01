# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to create prec_recall graphs for various models

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

#Read data from files
lr_pd = pd.read_csv('../Data/Trivial_LR_prec_rec.csv', index_col=0)
eng_lr_pd = pd.read_csv('../Data/Eng_LR_prec_rec.csv', index_col=0)
av_pd = pd.read_csv('../Data/Average_in_airport_prec_rec.csv', index_col=0)
rnd_pd = pd.read_csv('../Data/Random_prec_rec.csv', index_col=0)
#rf_pd = pd.read_csv('../Data/RF_prec_rec.csv', index_col=0)

dfs = [lr_pd, eng_lr_pd, av_pd, rnd_pd]

for df in dfs:
  df.drop(df[df.Precision < 1].index, inplace=True)
  df.drop(df[df.Recall < 1].index, inplace=True)
  df.drop(df[df.Precision > 90].index, inplace=True)

# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(10)
fig.set_size_inches(8, 6)

#Fill the axis with data
axes.plot(lr_pd.Recall, lr_pd.Precision, label = "LogReg", color = 'lightgreen')
axes.scatter(lr_pd.Recall, lr_pd.Precision, color = 'lightgreen')

axes.plot(av_pd.Recall, av_pd.Precision, label = "Average delay at the origin", color = 'r') 
axes.scatter (av_pd.Recall, av_pd.Precision, color = 'r') 

axes.plot(rnd_pd.Recall, rnd_pd.Precision, label = "Random", color = 'b') 
axes.scatter (rnd_pd.Recall, rnd_pd.Precision, color = 'b') 

axes.plot(eng_lr_pd.Recall, eng_lr_pd.Precision, label = "Eng LR", color = 'g') 
axes.scatter (eng_lr_pd.Recall, eng_lr_pd.Precision, color = 'g') 

#axes.plot(rf_pd.Recall, rf_pd.Precision, label = "Random Forest", color = 'brown') 
#axes.scatter (rf_pd.Recall, rf_pd.Precision, color = 'brown') 

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.45, color='gray', ls = '--')
axes.text(70, 40, '80% Recall', size=12)

# Write cutoff vaulues on the graph
for index in range(len(eng_lr_pd.Cutoff)):
  axes.text(eng_lr_pd.Recall[index]-0.02, 1 + eng_lr_pd.Precision[index], eng_lr_pd.Cutoff[index], size=9)
'''
for index in range(len(lr_pd.Cutoff)):
  axes.text(lr_pd.Recall[index]-0.02, 1 + lr_pd.Precision[index], lr_pd.Cutoff[index], size=9)
for index in range(len(av_pd.Cutoff)):
  axes.text(av_pd.Recall[index]-0.02, 1 + av_pd.Precision[index], av_pd.Cutoff[index], size=9)
for index in range(len(rnd_pd.Cutoff)):
  axes.text(rnd_pd.Recall[index]-0.02, 1 + rnd_pd.Precision[index], rnd_pd.Cutoff[index], size=9)

for index in range(len(rf_pd.Cutoff)):
  axes.text(rf_pd.Recall[index]-0.02, 1 + rf_pd.Precision[index], rf_pd.Cutoff[index], size=9)
'''
#Set legend position
axes.legend(loc = 'upper right')

#Setup the x and y 
axes.set_ylabel('Precision')
axes.set_xlabel('Recall')
axes.set_ylim(5, 80)

# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.show()
fig.savefig(f"../Images/Precision and recall.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------

def impute_precision(x,y, x_to_impute):
    int_idx = 0
    for i in range(len(x)-1):
        if ((x[i] > x_to_impute) & (x[i+1] < x_to_impute)):
            int_idx = i
    impute_value = y[int_idx+1] - (y[int_idx+1] - y[int_idx])* ((x[int_idx+1]-x_to_impute)/(x[int_idx+1]-x[int_idx]))
    return impute_value

# COMMAND ----------

dfs = [rnd_pd, av_pd, lr_pd, eng_lr_pd]
df_names = ['Random', 'Baseline', 'Trivial LR', 'Engineered LR']
prec_dic = {}
for df, name in zip(dfs, df_names):
    prec_dic[name] = [round(impute_precision(df['Recall'], df['Precision'], 80), 1)]

prec_df = pd.DataFrame.from_dict(prec_dic)
prec_df

# COMMAND ----------

#Read data from files
balanced = pd.read_csv('../Data/Eng_LR_prec_rec_balanced.csv', index_col=0)
unbalanced = pd.read_csv('../Data/Eng_LR_prec_rec.csv', index_col=0)

dfs = [balanced, unbalanced]
'''
for df in dfs:
  df.drop(df[df.Precision < 1].index, inplace=True)
  df.drop(df[df.Recall < 1].index, inplace=True)
  df.drop(df[df.Precision > 90].index, inplace=True)
'''
# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(10)
fig.set_size_inches(8, 6)

#Fill the axis with data
axes.plot(balanced.Recall, balanced.Precision, label = "Balanced", color = 'lightgreen')
axes.scatter(balanced.Recall, balanced.Precision, color = 'lightgreen')

axes.plot(unbalanced.Recall, unbalanced.Precision, label = "Unbalanced", color = 'g') 
axes.scatter(unbalanced.Recall, unbalanced.Precision, color = 'g')

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.45, color='gray', ls = '--')
axes.text(70, 40, '80% Recall', size=12)

# Write cutoff vaulues on the graph
for index in range(len(balanced.Cutoff)):
  axes.text(balanced.Recall[index]-0.02, 1 + balanced.Precision[index], balanced.Cutoff[index], size=9)

#Set legend position
axes.legend(loc = 'upper right')

#Setup the x and y 
axes.set_ylabel('Precision')
axes.set_xlabel('Recall')
axes.set_ylim(5, 80)

# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.show()
fig.savefig(f"../Images/Precision and recall balanced.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------


