from flask import Flask
from flask import render_template, request
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.io import output_notebook
from io import BytesIO
import base64
from bokeh.plotting import figure, show, output_notebook
from bokeh.embed import file_html
from bokeh.resources import CDN

#notes 

#pour run tu fais py -m flask run
#Ctrl click sur le lien et Ctrl C pour fermer 


#importation du dataset déjà nettoyer (pour éviter de refaire tout le travail à chaque fois)
df1=pd.read_csv("data_clean.csv",sep=",")






#toutes les fonctions utiles 
def encoding(data):
    for column in data.columns:
        if data[column].dtype != 'int64' and data[column].dtype != 'float' and column!='readmitted' and column!='patient_nbr' and column!='age':
            if len(data[column].unique()) != 2:
                target_mean = data.groupby(column)['readmitted'].mean()
                data[column] = data[column].map(target_mean)
    return data

def encoding_onehot(data):
    for column in data.columns:
        if data[column].dtype == 'object' and column not in ['readmitted', 'patient_nbr', 'age']:
            one_hot_encoded = pd.get_dummies(data[column], prefix=column, drop_first=True)
            data = pd.concat([data, one_hot_encoded], axis=1)
            data = data.drop(column, axis=1)
    return data





#toutes les variables annexes
colonnes_selected=['race','gender','age','admission_type','discharge_disposition','admission_source','time_in_hospital','num_procedures', 'number_diagnoses', 'max_glu_serum', 'A1Cresult','insulin','change', 'diabetesMed', 'readmitted']
df2=encoding(df1.copy())








#code pour les graphiques
def matrice_corr(df3):
   # Calcul de la matrice de corrélation
  correlation_matrix = df3[colonnes_selected].corr()
  # Paramètres de la figure
  plt.figure(figsize=(12, 10))
  # Création de la carte de corrélation avec une palette de couleurs
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
  # Ajout du titre
  plt.title('Matrice de Corrélation entre les Variables')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique">'
  return img_html

def graph2A(data):
  # Calculate the mean readmission probability for each race category
  mean_prob = data.groupby('race')['readmitted'].mean()
  # Create the bar plot
  plt.bar(mean_prob.index, mean_prob)
  plt.title("Readmitted Probability by Race")
  plt.xlabel("Race")
  plt.ylabel("Readmitted Probability")
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique">'
  return img_html


def graph2B(data):
  plt.figure(figsize=(8, 6))
  sns.countplot(x='gender', hue='readmitted', data=data)
  plt.title('Comparison of Gender and Readmission')
  plt.xlabel('Gender')
  plt.ylabel('Count')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique">'
  return img_html

def graph2C(data):
  # Calculate the sum and count of readmitted for each age group
  age_groups = data.groupby('age')['readmitted'].agg(['sum', 'count'])
  # Create the figure and axis objects
  fig, ax1 = plt.subplots(figsize=(10, 6))
  # Plot the sum of readmitted
  ax1.plot(age_groups.index, age_groups['sum'], color='blue', marker='o')
  ax1.set_xlabel("Age")
  ax1.set_ylabel("Sum of Readmitted", color='blue')
  # Create a twin axis sharing the x-axis
  ax2 = ax1.twinx()
  # Plot the count of readmitted
  ax2.plot(age_groups.index, age_groups['count'], color='red', marker='s')
  ax2.set_ylabel("Count of Readmitted", color='red')
  # Set the title
  plt.title("Sum and Count of Readmitted by Age")
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique">'
  return img_html


def graph3A(df):
  plt.figure(figsize=(8, 6))
  # Calculate percentage of 'readmitted' for each category
  data_percentage = (df.groupby(['rare_medicine_taken', 'readmitted']).size() / df.groupby('rare_medicine_taken').size()).reset_index(name='percentage')
  # Create a bar plot
  sns.barplot(x='rare_medicine_taken', y='percentage', hue='readmitted', data=data_percentage)
  plt.title('All Columns in rare_medicines Equal to No vs Readmitted (Percentage)')
  plt.xlabel('All Columns in rare_medicines Equal to No')
  plt.ylabel('Percentage')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique">'
  return img_html

def graph3B(df):
  # Create a subset of the DataFrame with the relevant columns
  subset = df[['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone','rosiglitazone', 'insulin','readmitted']]
  # Plot multiple bar plots in a grid layout
  fig, axes = plt.subplots(4, 2, figsize=(12, 16))
  axes = axes.flatten()
  # Iterate over each column and create a bar plot
  for i, col in enumerate(subset.columns[:-1]):
    sns.barplot(x=col, y='readmitted', data=subset, ax=axes[i])
    axes[i].set_title(f'Readmission Rate by {col.capitalize()}')
    axes[i].set_xlabel(col.capitalize())
    axes[i].set_ylabel('Readmission Rate')
  # Adjust spacing between subplots
  plt.tight_layout()
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique">'
  return img_html

def graph4A(df):
  crosstab_df = pd.crosstab(df['admission_type'],df['readmitted'])
  crosstab_df.plot(kind='bar', stacked=True)
  plt.title('Admission tyoe VS readmitted')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html

def graph4B(df):
  df_counts = df[['readmitted', 'admission_type']].groupby(['admission_type', 'readmitted']).size().reset_index(name='count')
  # Calculate proportions within each 'admission_type'
  df_counts['proportion'] = df_counts.groupby('admission_type')['count'].transform(lambda x: x / x.sum())
  # Plotting the count plot with proportions
  plt.figure(figsize=(10, 6))
  sns.barplot(x='admission_type', y='proportion', hue='readmitted', data=df_counts, palette='viridis')
  # Adding labels and title
  plt.xlabel('Admission Type')
  plt.ylabel('Proportion of patients')
  plt.title('Proportion of readmitted by Admission Type')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html


def graph4C(df):
  df_counts = df[['A1Cresult', 'admission_type']].groupby(['admission_type', 'A1Cresult']).size().reset_index(name='count')
  # Calculate proportions within each 'admission_type'
  df_counts['proportion'] = df_counts.groupby('admission_type')['count'].transform(lambda x: x / x.sum())
  # Plotting the count plot with proportions
  plt.figure(figsize=(10, 6))
  sns.barplot(x='admission_type', y='proportion', hue='A1Cresult', data=df_counts, palette='viridis')
  # Adding labels and title
  plt.xlabel('Admission Type')
  plt.ylabel('Proportion of patients')
  plt.title('Proportion of A1Cresult by Admission Type')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html

def graph4D(df):
  # Get unique values of 'admission_type'
  admission_types = df['admission_type'].unique()
  # Set up subplots with a 4x2 grid
  fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), sharey=True)
  # Loop through each admission_type and create a count plot in the corresponding subplot
  for i, admission_type in enumerate(admission_types):
      # Calculate the row and column indices
      row_idx, col_idx = divmod(i, 2)
      # Create a subset of the data for the current admission_type
      subset_df = df[df['admission_type'] == admission_type]
      # Calculate proportions manually
      proportions = subset_df.groupby(['A1Cresult', 'readmitted']).size() / len(subset_df)
      proportions = proportions.reset_index(name='proportion')
      # Create the bar plot with normalization
      sns.barplot(x='A1Cresult', y='proportion', hue='readmitted', data=proportions, palette="muted", ax=axes[row_idx, col_idx])
      # Customize the plot
      axes[row_idx, col_idx].set_title(f'Admission Type: "{admission_type}"')
      axes[row_idx, col_idx].set_xlabel('A1Cresult')
      axes[row_idx, col_idx].set_ylabel('Proportion')
      axes[row_idx, col_idx].legend(title="Readmitted", loc="upper right")
  # Adjust layout
  plt.tight_layout()
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html
   

def graph5A(df):
  crosstab_df = pd.crosstab(df['admission_source'],df['readmitted'])
  crosstab_df.plot(kind='bar', stacked=True)
  plt.title('Admission source VS readmitted')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html

def graph5B(df,colonne):
  df_counts = df[[colonne, 'admission_source']].groupby(['admission_source', colonne]).size().reset_index(name='count')
  # Calculate proportions within each 'admission_source'
  df_counts['proportion'] = df_counts.groupby('admission_source')['count'].transform(lambda x: x / x.sum())
  # Plotting the count plot with proportions
  plt.figure(figsize=(10, 6))
  sns.barplot(x='admission_source', y='proportion', hue=colonne, data=df_counts, palette='viridis')
  # Adding labels and title
  plt.xlabel('Admission Source')
  plt.ylabel('Proportion of patients')
  plt.title(f'Proportion of {colonne} by Admission Source')
  # Display the plot
  plt.xticks(rotation=90)
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html

def graph6A(df,colonne):
  df_bar = df[colonne].value_counts()
  df_bar.sort_values(ascending=False).plot.barh(color='purple')
  plt.title(f'number of patient per {colonne}')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html


def graph7A(df):
  # Create a Bokeh data source
  source = ColumnDataSource(df)
  # Create a linear colormap to map 'readmitted' values to colors
  mapper = linear_cmap(field_name='readmitted', palette=Viridis256, low=0, high=1)
  # Create the Bokeh figure
  p = figure(width=800, height=600, tools='pan,box_zoom,reset', title='Bokeh Graph')
  # Add points to the graph using the specified columns
  scatter = p.scatter(x='num_lab_procedures', y='num_medications', size='time_in_hospital',
                      color=mapper, source=source, legend_field='readmitted', alpha=0.6)
  # Add axes and a legend
  p.xaxis.axis_label = 'Number of Lab Procedures'
  p.yaxis.axis_label = 'Number of Medications'
  p.legend.title = 'Readmitted'
  img_html = file_html(p, CDN)
  return img_html


def graph7B(df):
  fig, ax = plt.subplots(figsize=(13, 7))
  sns.kdeplot(data=df, x='num_medications', hue='readmitted', fill=True, alpha=.5, linewidth=0)
  ax.set(xlabel='Number of medications', ylabel='Frequency')
  plt.title('Number of medications and Readmitted Frequency')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html

def graph7C(df):
  sns.boxplot(x='readmitted', y='time_in_hospital', data=df,order=sorted(df['readmitted'].unique()))
  plt.title(f'Distribution of time_in_hospital by readmitted')
  # Affichage de la carte de corrélation
  image_stream = BytesIO()
  plt.savefig(image_stream, format='png',bbox_inches='tight')
  plt.close()
  image_stream.seek(0)
  img_data = base64.b64encode(image_stream.read()).decode('utf-8')
  img_html = f'<img src="data:image/png;base64,{img_data}" alt="Graphique" >'
  return img_html




#appli web

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/graph1')
def graph1():
  plt.clf()
  plot_html=matrice_corr(df2)
  return render_template('graph1.html', plot_html=plot_html)

@app.route('/graph2')
def graph2():
    plt.clf()
    plot_html=graph2A(df1)
    plot_html2=graph2B(df1)
    plot_html3=graph2C(df1)
    return render_template('graph2.html', plot_html=plot_html,plot_html2=plot_html2,plot_html3=plot_html3)

@app.route('/graph3')
def graph3():
   plt.clf()
   plot_html=graph3A(df1)
   plot_html2=graph3B(df1)
   return render_template('graph3.html', plot_html=plot_html,plot_html2=plot_html2)


@app.route('/graph4')
def graph4():
   plt.clf()
   plot_html=graph4A(df1)
   plot_html2=graph4B(df1)
   plot_html3=graph4C(df1)
   plot_html4=graph4D(df1)
   return render_template('graph4.html', plot_html=plot_html,plot_html2=plot_html2,plot_html3=plot_html3,plot_html4=plot_html4)

@app.route('/graph5')
def graph5():
   plt.clf()
   plot_html=graph5A(df1)
   plot_html2=graph5B(df1,"max_glu_serum")
   plot_html3=graph5B(df1,"A1Cresult")
   return render_template('graph5.html', plot_html=plot_html,plot_html2=plot_html2,plot_html3=plot_html3)

@app.route('/graph6')
def graph6():
   plt.clf()
   plot_html=graph6A(df1,'diag_1')
   plot_html2=graph6A(df1,'diag_2')
   plot_html3=graph6A(df1,'diag_3')
   return render_template('graph6.html', plot_html=plot_html,plot_html2=plot_html2,plot_html3=plot_html3)

@app.route('/graph7')
def graph7():
   plt.clf()
   plot_html=graph7A(df1)
   plot_html2=graph7B(df1)
   plot_html3=graph7C(df1)
   return render_template('graph7.html', plot_html=plot_html,plot_html2=plot_html2,plot_html3=plot_html3)


app.run(debug=True)

