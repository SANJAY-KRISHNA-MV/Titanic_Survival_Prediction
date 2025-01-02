import pandas as pd
from data_preprocessing import impute_embarked, create_is_alone, sex_to_boolean, impute_age, extract_title, categorize_age, has_cabin, fare_category

def load_and_preprocess_data(path):
  """
  Loads the Titanic dataset, cleans and preprocesses it.

  Returns:
    Cleaned and preprocessed DataFrame.
  """
  train_df = pd.read_csv(path) 
  train_df = has_cabin(train_df)
  train_df = impute_embarked(train_df)
  train_df = create_is_alone(train_df) 
  train_df = sex_to_boolean(train_df)
  train_df = impute_age(train_df)
  train_df = extract_title(train_df)
  train_df = categorize_age(train_df)
  train_df = fare_category(train_df)
  # ... (Apply other cleaning functions) ...
  return train_df