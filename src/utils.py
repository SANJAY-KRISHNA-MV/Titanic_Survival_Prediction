def train_and_save_model():
  """
  Trains a Support Vector Machine (SVM) model using the Titanic dataset and saves the model to a file.
  """
  import pickle
  import model_training as mt
  import data_loader as dl

  df = dl.load_and_preprocess_data(r"..\Titanic_Survival_Prediction\data\Titanic-Dataset.csv")

  svc_model = mt.train_svc_model(df)

  # Save the trained model to a file
  with open('models/trained_svm_model.pkl', 'wb') as model_file:
      pickle.dump(svc_model, model_file)

  # Load the saved model
  with open('models/trained_svm_model.pkl', 'rb') as model_file:
      loaded_model = pickle.load(model_file)

  # Make predictions using the loaded model
  from sklearn.metrics import accuracy_score
  _ , X_test, _, y_test = mt.split_data(df)
  y_pred = svc_model.predict(X_test)
  y_pred_loaded = loaded_model.predict(X_test)

  # Evaluate the loaded model
  accuracy = accuracy_score(y_test, y_pred)
  loaded_model_accuracy = accuracy_score(y_test, y_pred_loaded)
  print(f"Original Model Accuracy: {accuracy}")
  print(f"Loaded Model Accuracy: {loaded_model_accuracy}")

def get_model_input_instructions():
  """
  Provides instructions for preparing input data for the Titanic survival prediction model.

  Returns:
    A string containing the instructions.
  """

  instructions = """
  **Data Input Instructions**

  To use the trained model for predictions, please prepare your data in the following format:

  Data columns (total 12 columns):
  #   Column       Non-Null Count  Dtype  
  ---  ------       --------------  -----  
  0   PassengerId  891 non-null    int64   
  1   Pclass       891 non-null    int64  
  2   Name         891 non-null    object 
  3   Sex          891 non-null    object 
  4   Age          714 non-null    float64
  5   SibSp        891 non-null    int64  
  6   Parch        891 non-null    int64  
  7   Ticket       891 non-null    object 
  8   Fare         891 non-null    float64
  9   Cabin        204 non-null    object 
  10  Embarked     889 non-null    object 
  dtypes: float64(2), int64(4), object(5)

  **Important Notes:**

  1. **Data Format:** Please ensure your data is in a CSV or similar format.
  2. **Feature Names:** Use the exact feature names as listed above.
  3. **Data Types:** Ensure the data types for each feature match the specified types.
  4. **Encoding:**
      - For 'Sex', use 'male' or 'female'.
      - For 'Embarked', use 'S' for Southampton, 'C' for Cherbourg, and 'Q' for Queenstown.
      - For 'Title', use the specified titles (e.g., 'Mr', 'Miss', 'Mrs', 'Master', 'Rare').
  5. **Missing Values:** Handle missing values appropriately (e.g., imputation) before making predictions.
      - you can choose to completely drop the Ticket column as it is not used in the model.
  6. You should call the load_and_preprocess_data() function in the 'data_loader.py' script to load and 
     preprocess your data before making predictions.
     As the function would clean and process the data as required by the model.

  Please refer to the project documentation for more details and any specific requirements.

  """
  return instructions

# Train and save the model
train_and_save_model()

# Get and print the instructions
instructions = get_model_input_instructions()
print(instructions)