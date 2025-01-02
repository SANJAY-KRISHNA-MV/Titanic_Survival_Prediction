def split_data(df):
    """
    Splits the data into training and test data.

    Args:
    df: The dataset.

    Returns:
    X_train, X_test, y_train, y_test.
    """
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['Survived','PassengerId','Name','Age','Ticket','Cabin','Fare','Embarked','Title','FamilySize'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_logistic_regression_model(df):
    """
    Trains an Logistic Regression model on the given training data.

    Args:
    X_train: Training features.
    y_train: Training labels.

    Returns:
    Trained Logistic Regression model.
    """
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = split_data(df)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model

def train_svm_model(df):
    """
    Trains an SVM model on the given training data.

    Args:
    X_train: Training features.
    y_train: Training labels.

    Returns:
    Trained SVM model.
    """
    from sklearn.svm import SVC
    # Create an SVM classifier
    svm_model = SVC(kernel='linear', C=1.0)  # You can experiment with different kernels and C values

    # Train the model
    X_train, X_test, y_train, y_test = split_data(df)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_random_forest_model(df):
    """
    Trains a Random Forest Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained Random Forest model.
    """
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, max_features=3, random_state=42) 
    X_train, X_test, y_train, y_test = split_data(df)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_knn_model(df):
    """
    Trains a K-Nearest Neighbors Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained KNN model.
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=3)
    X_train, X_test, y_train, y_test = split_data(df) 
    knn_model.fit(X_train, y_train)
    return knn_model

def train_xgboost_model(df):
    """
    Trains an XGBoost Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained XGBoost model.
    """
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42) 
    X_train, X_test, y_train, y_test = split_data(df)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_svc_model(df):
    """
    Trains a Support Vector Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained SVC model.
    """
    from sklearn.svm import SVC
    svc_model = SVC(probability=True)
    X_train, X_test, y_train, y_test = split_data(df)
    svc_model.fit(X_train, y_train)
    return svc_model

def train_decision_tree_model(df):
    """
    Trains a Decision Tree Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained Decision Tree model.
    """
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split_data(df)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_ada_boost_model(df):
    """
    Trains an AdaBoost Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained AdaBoost model.
    """
    from sklearn.ensemble import AdaBoostClassifier
    ada_model = AdaBoostClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split_data(df)
    ada_model.fit(X_train, y_train)
    return ada_model

def train_gradient_boosting_model(df):
    """
    Trains a Gradient Boosting Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained Gradient Boosting model.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(random_state=42)
    X_train, X_test, y_train, y_test = split_data(df)
    gb_model.fit(X_train, y_train)
    return gb_model

def evaluate_model(model, df):
    """
    Evaluates the model on the test data.

    Args:
    model: Trained model.
    X_test: Test features.
    y_test: Test labels.

    Returns:
    The accuracy of the model on the test data.
    """
    from sklearn.metrics import accuracy_score
    _ , X_test, _, y_test = split_data(df)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def predict(model, df):
    """
    Predicts the target variable using the trained model.

    Args:
    model: Trained model.
    df: The dataset to predict on.

    Returns:
    The predicted target variable.
    """
    X = df.drop(columns=['Survived','PassengerId','Name','Age','Ticket','Cabin','Fare','Embarked','Title','FamilySize'])
    return model.predict(X)
