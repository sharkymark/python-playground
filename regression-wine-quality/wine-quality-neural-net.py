import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import argparse
import os

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path, sep=',', header=0)

# Preprocess the data
def preprocess_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y

# Scale the data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train a neural network regressor
def train_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model

# Save the model
def save_model(model, file_path):
    model.save(file_path)

# Load the model
def load_model(file_path):
    return keras.models.load_model(file_path)

# Make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Main function
def main():
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

        # Define the scaler object
        scaler = StandardScaler()

        # Fit the scaler object to your training data
        scaler.fit(X_train)

        # Scale the data
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

        # Train the model
        model = train_model(X_train_scaled, y_train)

        # Evaluate the model on the test data
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred.flatten())
        print(f'\nModel MSE on test data: {mse:.3f}\n')

        # Save the model
        save_model(model, args.model_file)

        print('\nModel trained and saved!\n')

    while True:
        # Load the model
        model = load_model(args.model_file)

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

            # Scale the test data
            scaler = StandardScaler()
            # Fit the scaler object to your training data
            scaler.fit(X_train)
            test_df_scaled = scaler.fit_transform(test_df)

            # Make predictions
            predictions = model.predict(test_df_scaled)

            print('\nPredicted quality:', predictions[0][0])

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

            # Define the scaler object
            scaler = StandardScaler()
            # Fit the scaler object to your training data
            scaler.fit(X_train)
            # Scale the test data
            X_test_scaled = scaler.transform(X_test)

            # Make predictions
            predictions = model.predict(X_test_scaled)

            print('\nPredicted qualities:')
            print(predictions)

        else:
            print("Invalid choice. Please choose 1 or 2.")

        response = input('\nDo you want to continue? (y/n): ')
        if response.lower() != 'y':
            break

if __name__ == '__main__':
    main()
                                    
