#!/usr/bin/env python
# coding: utf-8

# <h1 style="color: #1F4E79; font-family: 'Verdana', sans-serif; text-align: center; font-weight: bold;">
# 🚢 <b>The Titanic Data Analysis & Survival Prediction Project 🌊</b>
# </h1>

# In[1]:


#importation des bibliothèques essentielles 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


# In[2]:


# Vérifier si le chemin du fichier est correct
path_train = r'C:\Users\HP\Desktop\train.csv'
path_test = r'C:\Users\HP\Desktop\test.csv'

print(os.path.exists(path_train))  
print(os.path.exists(path_test))   
#les données 
train_df = pd.read_csv(r'C:\Users\HP\Desktop\train.csv')
test_df = pd.read_csv(path_test) 


# In[3]:


train_df.head()


# In[4]:


train_df.info()


# In[5]:


#Obtenir le nombre de colonnes
nombre_de_colonnes = train_df.shape[1]


# In[6]:


#obtenir les noms de colonnes 
noms_des_colonnes =  train_df.columns.tolist()
train_df.columns.tolist()


# In[13]:


#Analyse Exploratoire des données 
train_df.describe()
train_df.describe()
train_df.describe()


# **Traitement des variables catégorielles**
# Pour préparer nos données en vue de l'analyse ou des modèles de machine learning, nous allons procéder à l'encodage des variables catégorielles comme suit :
# 
# **Sex :**
# 
# Méthode : Encodage One-Hot
# Description : Nous allons transformer cette variable en variables binaires pour représenter les catégories (male et female).
# **Embarked :**
# 
# Méthode : Encodage One-Hot
# Description : Nous allons convertir cette variable en variables binaires pour représenter les différents ports d'embarquement (S, C, Q).
# **Pclass :**
# 
# Méthode : Encodage Ordinal
# Description : Nous allons encoder cette variable avec des valeurs ordinales basées sur l'ordre des classes (1 pour la première classe, 2 pour la deuxième, 3 pour la troisième).
# Ces transformations nous permettent de convertir les variables catégorielles en formats numériques adaptés aux modèles de machine learning, tout en respectant la nature des données. Les autres variables, principalement numériques, seront utilisées telles quelles ou après une normalisation si nécessaire.
# 
# 

# In[192]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Créer une copie du DataFrame pour la transformation
df_transformed = train_df.copy()
# Configurer l'encodeur One Hot pour 'Sex'
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

# Appliquer l'encodage à la variable 'Sex'
sex_encoded = one_hot_encoder.fit_transform(df_transformed[['Sex']])
# Convertir la matrice encodée en DataFrame
sex_encoded_df = pd.DataFrame(sex_encoded, columns=one_hot_encoder.get_feature_names_out(['Sex']))

# Supprimer la colonne originale 'Sex'
df_transformed = df_transformed.drop(columns=['Sex'])

# Ajouter les nouvelles colonnes encodées au DataFrame
df_transformed = pd.concat([df_transformed, sex_encoded_df], axis=1)
print(df_transformed)


# In[194]:


embarked_categories = train_df['Embarked'].unique()
embarked_categories


# In[196]:


# Appliquer l'encodage à la variable 'Embarked'
embarked_encoded = one_hot_encoder.fit_transform(df_transformed[['Embarked']])

# Convertir la matrice encodée en DataFrame
embarked_encoded_df = pd.DataFrame(embarked_encoded, columns=one_hot_encoder.get_feature_names_out(['Embarked']))

# Supprimer la colonne originale 'Embarked'
df_transformed = df_transformed.drop(columns=['Embarked'])

# Ajouter les nouvelles colonnes encodées au DataFrame
df_transformed = pd.concat([df_transformed, embarked_encoded_df], axis=1)
print(df_transformed)


# In[206]:


# Configurer l'encodeur Ordinal
ordinal_encoder = OrdinalEncoder()

# Appliquer l'encodage ordinal à la variable 'Pclass'
df_transformed[['Pclass']] = ordinal_encoder.fit_transform(df_transformed[['Pclass']])
df_transformed


# In[202]:


df_transformed.columns


# In[208]:


df_transformed['Parch'].unique()


# In[228]:


#Parch est déjà numérique et n'a pas besoin d'encodage. nous pouvons  l'utiliser directement dans les modèles de machine learning.
df_transformed['Parch'].unique() 


# In[23]:


df_transformed['Survived'].unique()


# In[220]:


df_transformed['Pclass']
df_transformed['Pclass'].unique()


# In[222]:


df_transformed.shape[0]


# In[224]:


df_transformed.shape[1]


# In[230]:


df_transformed.info()


# In[35]:


df_transformed.columns 


# In[232]:


#Configurer pandas pour afficher toutes les colonnes du data frame 
pd.set_option('display.max_columns', None)
df_transformed


# In[234]:


#identifier les variables continues 
continuous_vars = df_transformed.select_dtypes(include = ['float64', 'int64']).columns
continuous_vars


# In[236]:


#visualisation des données continues 
import pandas as pd 
import matplotlib.pyplot as plt
#Histgramme
plt.figure(figsize = (10, 5))
sns.histplot(df_transformed['Age'].dropna(), bins = 30, kde = True)
plt.title('Distribution de l\'age')
plt.xlabel('Age')
plt.show()


# In[238]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10, 5))
sns.boxplot(x = df_transformed['Fare'].dropna())
plt.title('Boxplot de fare')
plt.xlabel('Faire')
plt.show()


# In[240]:


#Boxplot
plt.figure(figsize = (10, 5))
sns.boxplot(x = df_transformed['Age'].dropna())
plt.title('Boxplot de l\'Age')
plt.xlabel('Age')
plt.show()


# In[242]:


import plotly.express as px
import pandas as pd 
fig = px.box(df_transformed, x = 'Age', title = 'Boxplot d\'Age')
#personnaliser la couleur du boxplot 
fig.update_traces(marker_color = '#FFB6C1')
fig.show()


# In[244]:


import plotly.express as px 
fig = px.box(df_transformed, x = 'Fare', title = 'Boxplot of fare')
fig.update_traces(marker_color = '#98FF98')
#affichage du graphique 
fig.show()


# In[246]:


#Calcule de la moyenne 
mean = df_transformed['Fare'].mean()
mean


# In[248]:


#Visualisation de la distribution des tarifs pour voir sa forme
import plotly.express as px 
import pandas as pd
fig_fare = px.histogram(df_transformed, x = 'Fare', nbins = 30, title = 'Distribution des tarifs', labels = {'Fare':'Tarifs'}, template = 'plotly_white', color_discrete_sequence=['#3EB489'], opacity=0.8)

# Ajout d'une bordure autour des barres
fig_fare.update_traces(
    marker=dict(
        line=dict(
            width=3,  # Épaisseur de la bordure
            color='black'  # Couleur de la bordure
        )
    )
)
fig_fare.show()


# In[250]:


#Calculer la corrélation entre variable Age et variable Fare 
correlation = df_transformed['Age'].corr(df_transformed['Fare'])
correlation


# **Comment**
# 
# **With a correlation coefficient of 0.096 between Age and Fare, we observe a very weak and almost negligible relationship between these two variables. This indicates that, within this dataset, there is no significant link between an individual’s age and the amount they paid.**

# In[252]:


import plotly.express as px
fig = px.scatter(df_transformed, x = 'Age', y = 'Fare', title = 'Scatter plot of Age vs Fare', labels = {'Age' : 'Age', 'Fare' : 'Fare'})
fig.update_layout(xaxis_title = 'Age', yaxis_title = 'Fare')
fig.show()


# **Comment**
# 
# **"Based on the scatter plot of Age versus Fare, there appears to be a weak positive relationship between the two variables, with a correlation coefficient of 0.096. This indicates that Age has a minimal effect on Fare. The plot shows a broad distribution of points, with no distinct trend or clustering, suggesting that variations in Age do not significantly predict changes in Fare."**

# In[254]:


df_transformed.columns


# In[256]:


df_transformed['SibSp'].dtype


# In[258]:


#Correlation between Age and SibSp
correlation = df_transformed['Age'].corr(df_transformed['SibSp'])
correlation


# **Comment**  
# **The correlation between SibSp and Age is -0.308, indicating a moderate negative relationship between the number of siblings/spouses aboard and the age of passengers. This suggests that, on average, passengers with a higher number of siblings or spouses aboard tend to be younger. However, this relationship is not extremely strong, meaning that while there is a noticeable trend, other factors may also influence age. The negative correlation indicates that as the number of siblings/spouses increases, the age of the passengers tends to decrease, though not strongly.**

# In[260]:


import plotly.express as px
fig = px.scatter(df_transformed, x = 'SibSp', y = 'Age', title = 'Scatter plot of SibSp vs Age')
fig.show()


# **Comment**  
# **The scatter plot illustrates the relationship between Age and SibSp, the number of siblings/spouses aboard. The plot reveals a weak negative correlation of -0.308, indicating that as the number of siblings/spouses increases, there is a slight tendency for the age to decrease. However, the distribution of points is quite dispersed with no strong trend or clustering. This suggests that variations in the number of siblings/spouses aboard do not have a significant predictive effect on the age of passengers.**

# In[262]:


#Correlation between Fare and SibSp
correlation = df_transformed['Fare'].corr(df_transformed['SibSp'])
correlation


# In[264]:


import plotly.express as px 
fig = px.scatter(df_transformed, x = 'Fare', y = 'SibSp', title = 'Scatter plot of SibSp vs Fare', labels = {'SibSp' : 'Siblings/Spouses Aboard', 'Fare' : 'Ticket Price'})
fig.update_layout(xaxis_gridcolor = 'thistle')
fig.update_layout(yaxis_gridcolor = 'thistle')
fig.update_traces(marker = dict(color = 'mediumpurple'))


# In[266]:


import plotly.express as px 
#create the density plot for the variable Age 
fig = px.density_contour(df_transformed, x = 'Age', title = 'KDE plot of Age')
fig.update_traces(contours_coloring="lines") 
fig.show()


# In[268]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assurez-vous que df_transformed est déjà défini et contient la colonne 'Age'

# Créer le graphique de densité avec probabilités
sns.kdeplot(df_transformed['Age'], fill=True, common_norm=True)

# Ajouter des étiquettes et un titre
plt.xlabel('Âge')
plt.ylabel('Probabilité')
plt.title('Graphique de densité des âges')

# Afficher le graphique
plt.show()


# In[270]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assurez-vous que df_transformed est déjà défini et contient la colonne 'Age'

# Créer le graphique de densité avec probabilités
sns.kdeplot(df_transformed['Fare'], fill=True, common_norm=True)

# Ajouter des étiquettes et un titre
plt.xlabel('Fare')
plt.ylabel('Probabilité')
plt.title('Graphique de densité de Fare')

# Afficher le graphique
plt.show()


# In[274]:


import sys
print("Python executable location:", sys.executable)


# **Etape de machine learning**

# In[276]:


#Importer les bibliothèques nécessaires 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm


# In[278]:


df_transformed.columns


# In[280]:


df_transformed.head()


# In[296]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Supposons que df_transformed est déjà défini et contient vos données

# Sélectionner les variables continues
continuous_features = df_transformed[['Age', 'SibSp', 'Parch', 'Fare']]

# Initialiser le scaler
scaler = StandardScaler()

# Appliquer la normalisation
continuous_scaled = scaler.fit_transform(continuous_features)

# Créer un DataFrame à partir des données normalisées
scaled_df = pd.DataFrame(continuous_scaled, columns=continuous_features.columns)

# Remplacer les colonnes d'origine par les versions normalisées
df_transformed[['Age', 'SibSp', 'Parch', 'Fare']] = scaled_df

# Afficher le DataFrame final
print(df_transformed.head())


# In[302]:


df_transformed


# In[306]:


#sélectionner les variables explicatives et la variable cible 
x_train = df_transformed[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]# variables Explicatives 
y_train = df_transformed[['Survived']]
x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]  # Assurez-vous que y_train correspond à x_train


# In[310]:


# Sélectionner les variables explicatives et la variable cible
x_train = df_transformed[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]  # Variables explicatives
y_train = df_transformed[['Survived']]

# Supprimer les valeurs manquantes dans x_train
x_train = x_train.dropna()

# Assurez-vous que y_train correspond à x_train
y_train = y_train.loc[x_train.index]


# In[314]:


from sklearn.linear_model import LogisticRegression

# Créer le modèle de régression logistique
model = LogisticRegression(max_iter=200)  # Augmenter max_iter si nécessaire

# Entraînement du modèle sur les données d'entraînement
model.fit(x_train, y_train.values.ravel())  # Utiliser ravel() pour aplatir y_train

# Prédiction sur les données d'entraînement
y_pred_train = model.predict(x_train)

# Afficher les prédictions
print(y_pred_train)


# In[339]:


# Évaluer l'accuracy du modèle
accuracy = accuracy_score(y_train, y_pred_train)
print(f'Accuracy: {accuracy:.2f}')

# Matrice de confusion
conf_matrix = confusion_matrix(y_train, y_pred_train)
print('Confusion Matrix:')
print(conf_matrix)

# Rapport de classification
class_report = classification_report(y_train, y_pred_train)
print('Classification Report:')
print(class_report)


# **Amélioration du modèle**

# In[346]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sélectionner uniquement les colonnes numériques
numeric_df = df_transformed.select_dtypes(include=['float64', 'int64'])

# Calcul de la matrice de corrélation
correlation_matrix = numeric_df.corr()

# Affichage de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de Corrélation des Variables Numériques')
plt.show()


# In[ ]:




