import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
d = pd.read_csv('/content/turnover.csv', encoding='latin-1')
d
d.isna().sum()
y = d['event']
X = d.drop('event', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from operator import le
# Import OneHotEncoder
from sklearn.preprocessing import  LabelEncoder
le=LabelEncoder()
d['label_encoed']=le.fit_transform(d['gender'])
d
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = '/content/turnover.csv'  # Replace with the correct file path
data = pd.read_csv(file_path, encoding='latin1')

# Encode categorical variables
categorical_cols = ['gender', 'industry', 'profession', 'traffic', 'coach', 'head_gender', 'greywage', 'way']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    data[col] = label_encoders[col].fit_transform(data[col])

# Separate features and target
X = data.drop(columns=['event'])
y = data['event']

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR(),
    "Multiple Regression": LinearRegression(),
    "Polynomial Regression": LinearRegression()
}

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(X_train)  # Apply to all features
x_test_poly = poly_features.transform(X_test)

# Train and evaluate models
model_accuracies = {}
for model_name, model in models.items():
    if model_name == "Polynomial Regression":
        model.fit(x_train_poly, y_train)
        y_pred = model.predict(x_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred) # This line was previously indented causing the error.
rmse = np.sqrt(mse) # This line and the subsequent lines were incorrectly indented.
r2 = r2_score(y_test, y_pred)
model_accuracies[model_name] = {"MSE": mse, "RMSE": rmse, "R-squared": r2}

print(f"{model_name}:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R-squared: {r2:.4f}")
def predict_turnover():
    """Predict employee turnover based on user input."""
    stag = float(input("Enter Stage value: "))
    gender = input("Enter Gender (m/f): ")
    age = float(input("Enter Age: "))
    industry = input("Enter Industry: ")
    profession = input("Enter Profession: ")
    traffic = input("Enter Traffic source: ")
    coach = input("Has coach (yes/no): ")
    head_gender = input("Head Gender (m/f): ")
    greywage = input("Grey wage (white/grey): ")
    way = input("Enter Way (bus, train, etc.): ")
    extraversion = float(input("Enter Extraversion score: "))
    independ = float(input("Enter Independence score: "))
    selfcontrol = float(input("Enter Self-control score: "))
    anxiety = float(input("Enter Anxiety score: "))
    novator = float(input("Enter Novator score: "))

    # Create input dataframe
    new_data = pd.DataFrame({
        'stag': [stag], 'gender': [gender], 'age': [age], 'industry': [industry],
        'profession': [profession], 'traffic': [traffic], 'coach': [coach],
        'head_gender': [head_gender], 'greywage': [greywage], 'way': [way],
        'extraversion': [extraversion], 'independ': [independ],
        'selfcontrol': [selfcontrol], 'anxiety': [anxiety], 'novator': [novator]
    })

    # Encode categorical data
    for col in categorical_cols:
        # Handle unseen labels by assigning a default value or raising a more informative error
        try:
            new_data[col] = label_encoders[col].transform(new_data[col])
        except ValueError as e:
            print(f"Warning: Unseen label '{e.args[0].split()[-1]}' for column '{col}'. Using most frequent value instead.")
            # Impute with the most frequent value for the column
            most_frequent_value = data[col].mode()[0] 
            new_data[col] = most_frequent_value

    # Scale numerical features
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

    # Predict with the best model
    # Assuming 'best_model_name' and 'best_model' are defined somewhere in your code
    best_model_name = "Polynomial Regression"  # Replace with your actual best model name
    best_model = models[best_model_name]       # Replace with your actual best model
    
    if best_model_name == "Polynomial Regression":
        new_data_poly = poly_features.transform(new_data)
        predicted_turnover_prob = best_model.predict(new_data_poly)[0]
    else:
        predicted_turnover_prob = best_model.predict(new_data)[0]
    
    print("Predicted Turnover Probability:", predicted_turnover_prob) # Print inside the function

# Call the function to make a prediction
predict_turnover()
