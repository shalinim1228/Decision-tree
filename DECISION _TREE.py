import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Sample dataset
data = {
    'Value': ['True', 'True', 'False', 'False', 'False', 'True', 'True', 'True', 'False', 'False'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Cool', 'Cool', 'Cool', 'Hot', 'Hot', 'Cool', 'Cool'],
    'Level': ['High', 'High', 'High', 'Normal', 'Normal', 'High', 'High', 'Normal', 'Normal', 'High'],
    'Classification': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Value'] = label_encoder.fit_transform(df['Value'])
df['Temperature'] = label_encoder.fit_transform(df['Temperature'])
df['Level'] = label_encoder.fit_transform(df['Level'])
df['Classification'] = label_encoder.fit_transform(df['Classification'])

# Split data into features and target
X = df[['Value', 'Temperature', 'Level']]
y = df['Classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a Decision Tree Classifier with entropy criterion
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, clf.predict(X_test)))

# Function to get input from the user and predict the classification
def get_user_input_and_predict():
    # Encode user inputs based on LabelEncoder
    value_map = {'True': 1, 'False': 0}
    temperature_map = {'Hot': 1, 'Cool': 0}
    level_map = {'High': 0, 'Normal': 1}

    # Get user input
    value_input = input("Enter Value (True/False): ")
    temperature_input = input("Enter Temperature (Hot/Cool): ")
    level_input = input("Enter Level (High/Normal): ")

    # Encode the inputs
    value_encoded = value_map.get(value_input, -1)
    temperature_encoded = temperature_map.get(temperature_input, -1)
    level_encoded = level_map.get(level_input, -1)

    # Ensure valid inputs
    if -1 in [value_encoded, temperature_encoded, level_encoded]:
        print("Invalid input. Please enter valid values.")
        return

    # Create input for prediction
    user_data = [[value_encoded, temperature_encoded, level_encoded]]

    # Predict and decode classification
    prediction = clf.predict(user_data)
    class_map = {0: 'No', 1: 'Yes'}
    print(f"Predicted Classification: {class_map[prediction[0]]}")

# Run the function to get user input and predict
get_user_input_and_predict()
