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
av_pd = pd.read_csv('../Data/Average_in_airport_prec_rec.csv', index_col=0)
rnd_pd = pd.read_csv('../Data/Random_prec_rec.csv', index_col=0)

# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(10)

#Fill the axis with data
axes.plot(lr_pd.Recall, lr_pd.Precision, label = "LogReg", color = 'g')
axes.scatter(lr_pd.Recall, lr_pd.Precision,  label = "Probability cutoff", color = 'g')

axes.plot(av_pd.Recall, av_pd.Precision, label = "Average delay at the origin", color = 'r') 
axes.scatter (av_pd.Recall, av_pd.Precision, label = "Predicted delay cutoff, min", color = 'r') 

axes.plot(rnd_pd.Recall, rnd_pd.Precision, label = "Random", color = 'b') 
axes.scatter (rnd_pd.Recall, rnd_pd.Precision, label = "Probability cutoff", color = 'b') 

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.55, color='gray', ls = '--')
axes.text(70, 50, '80% Recall', size=12)

# Write cutoff vaulues on the graph
for index in range(len(lr_pd.Cutoff)):
  axes.text(lr_pd.Recall[index]-0.02, 1 + lr_pd.Precision[index], lr_pd.Cutoff[index], size=9)
for index in range(len(av_pd.Cutoff)):
  axes.text(av_pd.Recall[index]-0.02, 1 + av_pd.Precision[index], av_pd.Cutoff[index], size=9)
for index in range(len(rnd_pd.Cutoff)):
  axes.text(rnd_pd.Recall[index]-0.02, 1 + rnd_pd.Precision[index], rnd_pd.Cutoff[index], size=9)

#Set legend position
axes.legend(loc = 'upper left')

#Setup the x and y 
axes.set_ylabel('Precision')
axes.set_xlabel('Recall')
axes.set_ylim(5, 80)

# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.show()
fig.savefig(f"Precision and recall.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------


