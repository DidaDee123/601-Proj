import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Data/cleaned_data.csv")


fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

vars_to_plot = ['age', 'glucose', 'hba1c', 'cholesterol', 'triglycerides', 'creatinine']
titles = ['Age Distribution', 'Glucose Distribution', 'HbA1c Distribution', 'Cholesterol Distribution', 'Triglycerides Distribution', 'Creatinine Distribution']

for i, var in enumerate(vars_to_plot):
    sns.histplot(data=df, x=var, hue='gender', kde=True, bins=30, ax=axes[i], palette='Set2', alpha=0.5)
    axes[i].set_title(titles[i], fontsize=14)
    axes[i].set_xlabel(var.capitalize(), fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)

plt.tight_layout()
plt.savefig("Data/cleaned_data/cleaned_histograms.png")
plt.close()

binary_vars = ['has_hba1c', 'has_cholesterol', 'has_glucose', 'has_triglycerides', 
               'has_creatinine', 'encounter_inpatient', 'outcome_a', 'outcome_b']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, var in enumerate(binary_vars):
    sns.countplot(data=df, x=var, ax=axes[i], palette='pastel')
    axes[i].set_title(f'{var} Count', fontsize=12)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.savefig("Data/cleaned_data/binary_counts.png")
plt.close()