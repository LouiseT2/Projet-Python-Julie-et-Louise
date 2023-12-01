# Projet-Python-Julie-et-Louise
Prediction of Readmitted diabet patients in hospital.


omizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import timeit
from sklearn.metrics import confusion_matrix
fichier python flask :  
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




EXEMPLES D’UTILISATION 


fichier python projet : 
il faut run les cellules du notebook et regarder les graphiques/statistiques dans le terminal


fichier python flask : 
ouvrir le dossier “hello_world” sur visual studio code 
il faut run le fichier app.py
cliquer sur l’url dans le terminal (Ctrl+Click) puis naviguer sur les pages html afin de voir les différents graphiques


CONFIGURATION


fichier python projet : 
Importations
chargement des dataset
Traitement du dataset (colonnes, missing values, encodage, réencodage, duplications, valeurs à adapter)
Visualisations (graphiques sur les tendances des différentes variables en fonction de notre objectif afin de mieux cibler les variables intéressantes dans ce projet)
Prédiction (data splitting, normalisation, importations des modèles et des grilles d'hyper paramètres, prédiction et affichage de statistique)


fichier python flask : 
app.py : importations, chargement du dataset nettoyé/préparé, fonctions pour créer des graphiques, code flask
templates : affichage de toutes les pages html du site




CONTRIBUTION


Il manque à notre projet de meilleures optimisations de prédiction.
Nos opérations gridSearch ont été très coûteuses en temps, il est difficile de savoir quels hyperparamètres sont les plus adaptés. 
Une étude approfondie des statistiques serait un point en plus.


LIENS UTILES 


Liens vers les documentations utiles : 
Matplotlib : https://matplotlib.org/stable/index.html
Seaborn : https://seaborn.pydata.org/
Pandas : https://pandas.pydata.org/docs/reference/api/pandas.Series.html
Scikit-learn : https://scikit-learn.org/stable/index.html
Flask : https://pixees.fr/informatiquelycee/n_site/nsi_prem_flask.html
Html : https://developer.mozilla.org/fr/docs/Glossary/HTML


REMERCIEMENTS


Je souhaite remercier Louise, son imperturbable concentration et sa passion pour mener à bien un projet.
Échanger avec elle fut un plaisir du début à la fin.




NOUS CONTACTER


Louise Tiger : louise.tiger@edu.devinci.fr 
Julie Queimado : julie.queimado@edu.devinci.fr














PYTHON PROJECT A4


Louise TIGER
Julie QUEIMADO


OBJECTIVE


Our goal is to use all available information to predict whether a diabetic patient will return to the hospital in order to anticipate all the necessary services for their admission to the facility.


DATASET


For this, we have a dataset: diabetic_data.csv, from the link: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
it contains the records of patients who came to the hospital, as well as other variables:


Blue: numeric columns
The rest: numbered


Column Name: Description
int encounter_id: encounter identifier
int patient_nbr: person identifier
object race: represents the origin
object gender: male, female, or other
object age: represented by age ranges
int admission_type_id: if the admission was urgent, if it was an emergency ...
int discharge_disposition_id: how the person left (home...)
int admission_source_id: where the patient came from (firefighter, hospital transfer...)
int time_in_hospital: fairly intuitive
int num_lab_procedures: Number of lab tests performed during the encounter
int num_procedures: same as above except for non-laboratory tests
int num_medications: number of medications administered during the encounter
int number_outpatient: in the year, the number of times the person came without being a patient
int number emergency: in the year, the number of times the person came as a patient
objects diag_1, diag_2, diag_3: id of diagnoses 1, 2, and 3
int number_diagnoses: how many they have done
object max_glu_serum: range of that result or none if not measured
object same for A1Cresult
object same principle except here it's whether the medication was increased, decreased, or nothing: metformin, repaglinide, nateglinide, chlorpropamide, glimpiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone
object change: if the patient made any changes to their medication intake following the subsequent consultation
object diabetesMed: if diabetic medications were prescribed
object readmitted: if the patient was readmitted within >30 days or <30 days or No


VARIABLE ADDED AND DROP
-> To limit the number of diagnostic categories (more than 700) we can regroup them in only 19 disease types 
-> numerics : only integer informations of our dataset
-> rare_medicine : regoup 13 medicines used by less than 2% of the dataset population. We regrouped them together in rare_medicine=1 if one was taken 0 if none.
-> df_medocs_diabeth : dataset for only patient whom diabeth medicines has been prescribed
->readmitted_nbr : number of patient’s readmissions per patient
-> weight, payer_code, medical_specialty because too much missing values, plus three medicaments prescribed to no one where deleted



REFLEXION AND CONCLUSION
We are choosing categorical model: Logistic regression, Gradient Boosting Classifier and Random Forest. The last two are supposed to be more efficient as they can detect non linear correlation between variables. In our data set there is not a lot of variable linear correlated to our target variable, consequently Gradient Boosting Classifier and Random Forest should provide us better results.
We found a final correlation with Random Forest and  Gradient Boosting Classifier around 78%. We know that two patient with the same info can one be readmitted and the other not. So 78% is a really good prediction. It will allow the hospital to prepare themself witch patient are going to come back.



NECESSARY IMPORTS


Make sure to install these modules:


Python project file:
import pandas as pd
import numpy as np


# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.io import output_notebook


#Prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import timeit
from sklearn.metrics import confusion_matrix


Flask Python file:
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




USAGE EXAMPLES


Python project file:
Run the notebook cells and view the graphs/statistics in the terminal.


Flask Python file:
Open the "hello_world" folder in Visual Studio Code.
Run the app.py file.
Click on the URL in the terminal (Ctrl+Click) and then navigate through the HTML pages to see the different graphs.


CONFIGURATION


Python project file:
- Imports
- Dataset loading
- Dataset processing (columns, missing values, encoding, re-encoding, duplications, values to adapt)
- Visualizations (graphs on trends of different variables based on our goal to better target interesting variables in this project)
- Prediction (data splitting, normalization, model and hyperparameter grid imports, prediction, and display of statistics)


Flask Python file:
- app.py: imports, loading of cleaned/prepared dataset, functions to create graphs, Flask code
- templates: display of all HTML site pages


CONTRIBUTION


Our project lacks better prediction optimizations.
Our gridSearch operations were very time-consuming; it is difficult to know which hyperparameters are the most suitable.
An in-depth study of statistics would be an additional point.


USEFUL LINKS


Links to useful documentation:
- Matplotlib: https://matplotlib.org/stable/index.html 
- Seaborn: https://seaborn.pydata.org/ 
- Pandas: https://pandas.pydata.org/docs/reference/api/pandas.Series.html 
- Scikit-learn: https://scikit-learn.org/stable/index.html 
- Flask: https://pixees.fr/informatiquelycee/n_site/nsi_prem_flask.html 
- HTML: https://developer.mozilla.org/fr/docs/Glossary/HTML 
-Statistics : https://gt2.ariis.fr/les-algorithmes-dexploitation/lapprentissage-supervise/




ACKNOWLEDGEMENTS


I would like to thank Louise for her unwavering focus and passion for successfully completing a project. Exchanging ideas with her was a pleasure from start to finish.

I would like to express my gratitude to Julie for her outstanding dedication and contributions, which have been key to the success of this project.

CONTACT US


Louise Tiger: louise.tiger@edu.devinci.fr 
Julie Queimado: julie.queimado@edu.devinci.fr
