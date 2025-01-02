import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_is_alone(df):
  """
  Creates a new column 'IsAlone' in the DataFrame.

  Args:
    df: Pandas DataFrame containing the Titanic dataset.

  Returns:
    DataFrame with the new 'IsAlone' column.
  """

  df['FamilySize'] = df['SibSp'] + df['Parch'] 
  df['IsAlone'] = 0
  df.loc[df['FamilySize'] == 0, 'IsAlone'] = 1
  return df
    
def has_cabin(df):
    """
    Creates a new column 'HasCabin' in the DataFrame.
    
    Args:
        df: Pandas DataFrame containing the Titanic dataset.
    
    Returns:
        DataFrame with the new 'HasCabin' column.
    """
    
    df['HasCabin'] = 0
    df.loc[df['Cabin'].notnull(), 'HasCabin'] = 1
    return df

def impute_embarked(df):
  """
  Imputes missing values in the 'Embarked' column with the most frequent value.

  Args:
    df: Pandas DataFrame containing the Titanic dataset.

  Returns:
    DataFrame with imputed 'Embarked' values.
  """

  most_frequent_embarked = df['Embarked'].mode()[0]
  df['Embarked'] = df['Embarked'].fillna(most_frequent_embarked)
  return df

def sex_to_boolean(df):
  """
  Converts the 'Sex' column to boolean values (True/False) 
  and then to 1/0 representation.

  Args:
    df: Pandas DataFrame containing the Titanic dataset.

  Returns:
    DataFrame with the converted 'Sex' column.
  """

  # Convert 'Sex' to boolean (True for 'male', False for 'female')
  df['Sex'] = df['Sex'] == 'male' 

  # Convert boolean to 1/0 (1 for 'male', 0 for 'female')
  df['Sex'] = df['Sex'].astype(int) 

  return df

def impute_age(df):
    """
    Imputes missing values in the 'Age' column with the median value.

    Args:
        df: Pandas DataFrame containing the Titanic dataset.

    Returns:
        DataFrame with imputed 'Age' values.
    """
    
    median_age = df['Age'].median().astype(int)
    df['Age'] = df['Age'].fillna(median_age)
    return df

def extract_title(df):
  """
  Extracts titles from passenger names and creates a new 'Title' column.

  Args:
    df: Pandas DataFrame containing the Titanic dataset.

  Returns:
    DataFrame with the new 'Title' column.
  """

  df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
  df['Title'] = df['Title'].replace(['Ms', 'Mlle', 'Mme'], 'Miss')
  df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
  
  le = LabelEncoder()
  df['TitleEncoded'] = le.fit_transform(df['Title'])
  return df

def categorize_age(df):
  """
  Categorizes passenger ages into 'Child', 'Teen', 'Adult', and 'Elderly'.

  Args:
    df: Pandas DataFrame containing the Titanic dataset.

  Returns:
    DataFrame with the new 'Age_Category' column.
  """

  df['AgeCategory'] = pd.cut(df['Age'], bins=[0, 12, 19, 60, 120], labels=['Child', 'Teen', 'Adult', 'Elderly'], right=True)
  le = LabelEncoder()
  df['AgeCategory'] = le.fit_transform(df['AgeCategory'])
  return df

def fare_category(df):
    """
    Categorizes passenger fares into 'Low', 'Medium', 'High' and 'Very High.

    Args:
      df: Pandas DataFrame containing the Titanic dataset.

    Returns:
      DataFrame with the new 'Fare_Category' column.
    """
    
    df['FareCategory'] = pd.cut(df['Fare'], bins=[0, 20, 200, 300, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])
    le = LabelEncoder()
    df['FareCategory'] = le.fit_transform(df['FareCategory'])
    return df

