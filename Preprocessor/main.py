import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

#This code generates a dummy dataset with 100 rows and 4 columns: two numeric features, one categorical feature, and a binary target variable.
# The dataset includes some missing values and an outlier to simulate real-world data challenges.
# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Display the first few rows of the dummy dataset
print(df_dummy.head())

# These functions encapsulate the core preprocessing tasks, making them reusable across different datasets. They will be applied to our dummy data.
def load_data(df):
    return df

def handle_missing_values(df):
    # Fill numeric columns with their mean
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns with their mode (most frequent value)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        #.mode() returns the most frequent value (i.e. the mode) in a column.
        #mode() can return multiple values if there's a tie. For example:
        # Saying - "For this column, fill any missing values with the most frequent non-null value in the column."
        # This is a good default for categorical columns.


    return df  # For numeric data, fill missing values with the mean

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)].copy()  # copy change = This ensures that df_preprocessed isn't a "view" of the original df, so you can safely mutate it.
    # Remove rows with any outliers

def scale_data(df):
    scaler = StandardScaler()
    # Use .loc with column labels to do the assignment safely and unambiguously:
    # not this --> df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    #Before scaling, explicitly cast the numeric columns to float, like this:
    df[numeric_cols] = df[numeric_cols].astype('float64')  # cast before scaling
    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
    #This says:
    #"In all rows :, for these numeric columns, replace with scaled values."
    #Now Pandas is 100% sure you're modifying the real DataFrame.
    return df

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)

#Apply the preprocessing tool to the dummy data:
def main():
    # Load the data
    df_preprocessed = load_data(df_dummy)
    print("loaded")
    # Handle missing values
    df_preprocessed = handle_missing_values(df_preprocessed)
    print("handled")
    # Remove outliers
    df_preprocessed = remove_outliers(df_preprocessed)
    print("removed")
    # Scale the data
    df_preprocessed = scale_data(df_preprocessed)
    print("scaled")
    # Encode categorical variables
    df_preprocessed = encode_categorical(df_preprocessed, ['Category'])
    print("encoded")
    # Display the preprocessed data
    print(df_preprocessed.head())
    print("displayed")
    # Save the cleaned and preprocessed DataFrame to a CSV file

    # Saving the preprocessed data to a new file ensures that it’s ready for use in training ML models.
    # This step makes it easy to use the cleaned and processed data in future analysis or modeling efforts.
    save_data(df_preprocessed, 'preprocessed_dummy_data.csv')

    print('Preprocessing complete. Preprocessed data saved as preprocessed_dummy_data.csv')

    # This checks that all missing values. have been handled properly.
    # Check for missing values:
    print(df_preprocessed.isnull().sum())
    # This summarizes the dataset and confirms that any extreme values (outliers). have been removed.
    # Verify outlier removal:
    print(df_preprocessed.describe())
    # This ensures that the numeric features have been scaled properly, making them ready for ML algorithms.
    # Inspect scaled data:
    print(df_preprocessed.head())
    # This confirms that the categorical variables have been encoded into numerical values correctly.
    # Check categorical encoding:
    print(df_preprocessed.columns)

if __name__ == "__main__":
    main()