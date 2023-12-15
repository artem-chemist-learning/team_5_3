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
lr_pd = pd.read_csv('../Data/Trivial_LR_test.csv', index_col=0)
eng_lr_pd = pd.read_csv('../Data/Eng_LR_validation_unbalanced.csv', index_col=0)
av_pd = pd.read_csv('../Data/Average_in_airport.csv', index_col=0)
rnd_pd = pd.read_csv('../Data/Random.csv', index_col=0)
eng_lr_pd_blns = pd.read_csv('../Data/Eng_LR_validation.csv', index_col=0)
mlp_pd = pd.read_csv('../Data/MLP_validation.csv', index_col=0)

dfs = {"LR trivial" :lr_pd
        ,"LR engineered unbalanced":eng_lr_pd
        ,"LR engineered balanced":eng_lr_pd_blns
        ,"Mean origin delay":av_pd
        ,"Random":rnd_pd
        ,"MLP":mlp_pd
        }

colors = {"LR trivial" :'lawngreen'
        ,"LR engineered unbalanced":'mediumseagreen'
        ,"LR engineered balanced":'g'
        ,"Mean origin delay":'r'
        ,"Random":'black'
        ,"MLP":'b'}

for df in dfs.values():
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
for name, df in dfs.items():
  axes.plot(df.Recall, df.Precision, label = name, color = colors[name])
  axes.scatter(df.Recall, df.Precision, color =  colors[name])
  # Write cutoff vaulues on the graph
  for index in range(len(df.Cutoff)):
    axes.text(df.Recall[index]-0.02, 1 + df.Precision[index], df.Cutoff[index], size=7)

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.45, color='gray', ls = '--')
axes.text(70, 40, '80% Recall', size=12)

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

prec_dic = {}
for name, df in dfs.items():
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

#Read data from files
lr_pd = pd.read_csv('../Data/Trivial_LR_prec_rec.csv', index_col=0)
lr_pd_test = pd.read_csv('../Data/Trivial_LR_test.csv', index_col=0)

dfs = [lr_pd, lr_pd_test]

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
axes.plot(lr_pd.Recall, lr_pd.Precision, label = "LR on Train", color = 'lightgreen')
axes.scatter(lr_pd.Recall, lr_pd.Precision, color = 'lightgreen')

axes.plot(lr_pd_test.Recall, lr_pd_test.Precision, label = "LR on Test", color = 'g') 
axes.scatter(lr_pd_test.Recall, lr_pd_test.Precision, color = 'g')

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.45, color='gray', ls = '--')
axes.text(70, 40, '80% Recall', size=12)

# Write cutoff vaulues on the graph
for index in range(len(lr_pd_test.Cutoff)):
  axes.text(lr_pd_test.Recall[index]-0.02, 1 +lr_pd_test.Precision[index], lr_pd_test.Cutoff[index], size=9)

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
fig.savefig(f"../Images/Test vs train.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------


