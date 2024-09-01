import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import pickle
import os

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess the data
def preprocess_data(df):
    # drop columns
    df = df.drop(['RowNumber','CustomerId','Surname','Gender'], axis=1)

    # Handle categorical variables
    df['Geography'] = df['Geography'].astype('category')
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Geography'])
    
    return df

# Train a random forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

# Save the model and train features
def save_model(model, train_features, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model, 'train_features': train_features}, f)
  
# Load the model and train features
def load_model(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
        return loaded_data['model'], loaded_data['train_features']

# Make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Main function
def main():

    train_features = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=True)
    args = parser.parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model_file):
        # Load the dataset
        df = load_data(args.train_dataset)
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Split the data into features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_features = X_train.columns

        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model on the test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'\nModel accuracy on test data: {accuracy:.3f}\n')

        # Get feature importances
        feature_importances = model.feature_importances_

        # Print feature importances
        print("\nFeature Importances:\n")
        for feature, importance in zip(train_features, feature_importances):
            print(f"{feature}: {importance:.3f}")

        # Save the model and train features
        save_model(model, train_features, args.model_file)
        
        print('\nModel trained and saved!\n')
    
    while True:
            
        # Load the model and train features
        model, train_features = load_model(args.model_file)
        
        # Get user input
        print(f'\nEnter the following details to predict if the customer will exit the bank:\n')
        CreditScore = int(input('Enter CreditScore: '))
        Geography = input('Enter Geography (France, Spain, Germany): ')
        Age = int(input('Enter Age: '))
        Tenure = int(input('Enter Tenure: '))
        Balance = float(input('Enter Balance: '))
        NumOfProducts = int(input('Enter NumOfProducts: '))
        HasCrCard = int(input('Enter HasCrCard (0 or 1): '))
        IsActiveMember = int(input('Enter IsActiveMember (0 or 1): '))
        EstimatedSalary = float(input('Enter EstimatedSalary: '))
        
        # Create a test dataframe
        test_df = pd.DataFrame({
            'CreditScore': [CreditScore],
            'Geography': [Geography],
            'Age': [Age],
            'Tenure': [Tenure],
            'Balance': [Balance],
            'NumOfProducts': [NumOfProducts],
            'HasCrCard': [HasCrCard],
            'IsActiveMember': [IsActiveMember],
            'EstimatedSalary': [EstimatedSalary]
        })
        
        # Preprocess the test data
        test_df = pd.get_dummies(test_df, columns=['Geography'], drop_first=False)

        # Add missing one-hot encoded columns
        for col in train_features:
            if col not in test_df.columns:
                test_df[col] = 0

        test_df = test_df.reindex(columns=train_features, fill_value=0)
        
        # Make predictions
        predictions = make_predictions(model, test_df)
        
        print('\nPredicted Exited (churn from bank):', predictions[0])
        print()

        # Get feature importances
        feature_importances = model.feature_importances_

        # Print feature importances
        print("\nFeature Importances:\n")
        for feature, importance in zip(train_features, feature_importances):
            print(f"{feature}: {importance:.3f}")

        response = input('\nDo you want to predict for another customer? (y/n): ')
        if response.lower() != 'y':
            break


if __name__ == '__main__':
    main()