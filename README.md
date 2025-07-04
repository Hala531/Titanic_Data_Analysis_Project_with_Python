# Titanic Data Analysis Project with Python       




![Titanic Model - Haute résolution](https://cdn11.bigcommerce.com/s-n12pqvjack/images/stencil/1280x1280/products/49743/152991/hbb83420-1700-hobby-boss-rms-titanic-plastic-model-kit-squadron-model-models__53289.1693650953.jpg?c=1)


  
    
 
## Project Overview   
This project explores the famous Titanic dataset, providing a comprehensive data analysis to uncover key trends and insights about passenger survival. It involves data cleaning, visualization, and predictive modeling, using powerful Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn. The goal is to understand the factors that influenced survival rates on the Titanic, such as age, gender, and passenger class.  

## Objectives  

* **Data Importation** :  Load the Titanic dataset and the necessary Python libraries for analysis.
* **Data Exploration** : Analyze the dataset structure, including the number of columns, column names, and the types of data present.  
* **Categorical Variable Processing** : Clean and preprocess categorical variables to make them suitable for analysis.    
* **Data Visualization** : Generate insightful visualizations for continuous variables to identify trends and patterns.  
* **Correlation Analysis** : Calculate and interpret the correlation between continuous variables to understand their relationships.  

## Datasets  

this project uses two main datasets :  
`train.csv` : Contains 891 entries used for training the machine learning model.  
`test.csv` : Contains 418 entries used for model evaluation.  

Both datasets include 12 features describing each passenger:  

* **PassengerId** : Unique ID for each passenger.   
* **Survived** : Passenger class (1 = First, 2 = Second, 3 = Third). 
* **Pclass** : Passenger class (1 = First, 2 = Second, 3 = Third).  
* **Name** : Full name of the passenger    
* **Sex** : Gender of the passenger (male/female).    
* **Age** : Age of the passenger in years.    
* **SibSp** : Number of siblings or spouses aboard the Titanic.   
* **Parch** : Number of parents or children aboard the Titanic.   
* **Ticket** : Ticket number.   
* **Fare** : Fare paid for the ticket.   
* **Cabin** : Cabin number (if available).   
* **Embarked** : Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Steps and Workflow  

1. Import the necessary libraries: Pandas, Matplotlib, Pyplot, NumPy, Seaborn, os.
2. Load the datasets: train_data and test_data.
3. Perform exploratory data analysis (EDA): Check the number of columns, column names, and use functions like train_df.info() and train_df.describe() to understand the data.
4. Process categorical variables: Apply OneHotEncoder to the Embarked and Sex columns, and OrdinalEncoder to the Pclass column.
5. Visualize continuous variables: Use histograms for Age, boxplots for Fare, and interactive boxplots for Age and Fare.
6. Calculate statistical insights: Mean of Fare, correlation between Age and Fare, Age and SibSp, and Fare and SibSp.
7. Create scatter plots for these relationships and density plots for Age and Fare.
8. Build a machine learning model to predict survival outcomes based on the processed data.



