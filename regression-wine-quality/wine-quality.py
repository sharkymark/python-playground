import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import argparse
import pickle
import os

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path, sep=',', header=0)

# Preprocess the data
def preprocess_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y

# Train a random forest regressor
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
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
        X, y = preprocess_data(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_features = X_train.columns

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model on the test data
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'\nModel MSE on test data: {mse:.3f}\n')

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

        print("\n1. Enter a single row")
        print("2. Score a test dataset")
        choice = input("Enter your choice: ")

        if choice == '1':
            # Get user input
            fixed_acidity = float(input('Enter fixed acidity: '))
            volatile_acidity = float(input('Enter volatile acidity: '))
            citric_acid = float(input('Enter citric acid: '))
            residual_sugar = float(input('Enter residual sugar: '))
            chlorides = float(input('Enter chlorides: '))
            free_sulfur_dioxide = float(input('Enter free sulfur dioxide: '))
            total_sulfur_dioxide = float(input('Enter total sulfur dioxide: '))
            density = float(input('Enter density: '))
            pH = float(input('Enter pH: '))
            sulphates = float(input('Enter sulphates: '))
            alcohol = float(input('Enter alcohol: '))

            # Create a test dataframe
            test_df = pd.DataFrame({
                'fixed acidity': [fixed_acidity],
                'volatile acidity': [volatile_acidity],
                'citric acid': [citric_acid],
                'residual sugar': [residual_sugar],
                'chlorides': [chlorides],
                'free sulfur dioxide': [free_sulfur_dioxide],
                'total sulfur dioxide': [total_sulfur_dioxide],
                'density': [density],
                'pH': [pH],
                'sulphates': [sulphates],
                'alcohol': [alcohol]
            })

            # Make predictions
            predictions = make_predictions(model, test_df)

            print('\nPredicted quality:', predictions[0])

        elif choice == '2':
            # Check the train dataset and load the corresponding test dataset
            if args.train_dataset.endswith('red.csv'):
                test_df = load_data('./sets/test_wine_quality_red.csv')
            elif args.train_dataset.endswith('white.csv'):
                test_df = load_data('./sets/test_wine_quality_white.csv')
            else:
                print("Invalid train dataset. Please use either red or white.")
                continue

            # Preprocess the test data
            X_test, _ = preprocess_data(test_df)

            # Make predictions
            predictions = make_predictions(model, X_test)

            print('\nPredicted qualities:')
            print(predictions)

            # Get feature importances
            feature_importances = model.feature_importances_

            # Print feature importances
            print("\nFeature Importances:\n")
            for feature, importance in zip(train_features, feature_importances):
                print(f"{feature}: {importance:.3f}")

        else:
            print("Invalid choice. Please choose 1 or 2.")

        response = input('\nDo you want to continue? (y/n): ')
        if response.lower() != 'y':
            break


if __name__ == '__main__':
    main()          