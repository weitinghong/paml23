from optparse import Option
import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt        # pip install matplotlib
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown('# Preprocess Dataset')

#############################################

# Checkpoint 1
def restore_dataset():
    """
    Input: 
    Output: 
    """
    df=None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset',type = ['csv','txt'])
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df

# Checkpoint 3
def remove_features(X,removed_features):
    """
    Input: 
    Output: 
    """
    removed_features = st.multiselect('Select features to be removed', 
                    options= X.columns)
    X = X.drop(removed_features,axis = 'columns')

    return X

# Checkpoint 4
def impute_dataset(X, impute_method):
    """
    Input: 
    Output: 
    """
    # Select feature to impute
    feature_to_impute = st.selectbox('Which feature would you like to impute?', 
    options=X.columns)

    # Print summary statistics of missing values
    num_missing = X[feature_to_impute].isna().sum()
    num_total = len(X)
    num_features_missing = X.isna().any().sum()
    avg_missing_per_feature = X.isna().sum().mean()

    st.write('Summary Statistics:')
    st.write(f'Number of features with missing values: {num_features_missing}')
    st.write(f'Average number of data points missing per feature: {avg_missing_per_feature:.2f}')
    st.write(f'Total number of missing values in the dataset: {num_missing} out of {num_total} ({num_missing/num_total*100:.2f}%)')

    # Choose imputation method
    if impute_method == 'Skip missing values':
        X_imputed = X.dropna(subset=[feature_to_impute])
    elif impute_method == 'Zero':
        X_imputed = X.fillna(0, inplace=False)
    elif impute_method == 'Mean':
        mean_val = X[feature_to_impute].mean()
        X_imputed = X.fillna(value={feature_to_impute: mean_val}, inplace=False)
    elif impute_method == 'Median':
        median_val = X[feature_to_impute].median()
        X_imputed = X.fillna(value={feature_to_impute: median_val}, inplace=False)
    else:
        raise ValueError(f'Invalid imputation method: {impute_method}')

    return X_imputed

    
# Checkpoint 5

def compute_descriptive_stats(X):
    """
    Input: X
    Output: output_str (str), out_dict (dict)
    """
    stats_feature_select = st.multiselect('Select features to summarize', 
                                 options=X.columns)

    stats_select = st.multiselect('Select statistics to report', 
                                  options=['mean', 'median', 'max', 'min'])

    # descriptive statistics
    output_str = ''
    out_dict = {}
    for feature in stats_feature_select:
        out_dict = {}
        if feature in X.columns:
            out_dict['min'] = X[feature].min()
            out_dict['max'] = X[feature].max()
            out_dict['mean'] = X[feature].mean()
            out_dict['median'] = X[feature].median()
            
            #out_dict[feature] = [stats_select].to_dict()
        output_str += str(out_dict)

    st.markdown('### Result of the imputed dataframe')
    return output_str, out_dict

# Checkpoint 6
from sklearn.model_selection import train_test_split

def split_dataset(X):
    """
    Input: X (pandas DataFrame)
    Output: train (pandas DataFrame), test (pandas DataFrame)
    """
    # Collect percentage of data for test dataset
    test_size = st.number_input('percentage of data for test set', min_value=0, max_value=100, step=1, value=20)

    # Split the dataset into training and testing sets
    train, test = train_test_split(X, test_size=test_size/100, random_state=42)

    a = 3
    b = 1

    percent_train = len(train) / len(X) * 100
    percent_test = len(test) / len(X) * 100

    st.write('Dataset Split:')
    st.write(f'Training set: {percent_train:.2f}%')
    st.write(f'Test set: {percent_test:.2f}%')


    return train, test

# Restore Dataset
df = restore_dataset()

if df is not None:

    st.markdown('View initial data with missing values or invalid inputs')

    # Display original dataframe
    st.write(df)

    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values
    #st.markdown('Number of categories with missing values: {0:.2f}'.format())
    ############################################# MAIN BODY #############################################

    #numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Provide the option to select multiple feature to remove using Streamlit multiselect

    # Remove the features using the remove_features function

    # Display updated dataframe

    # Clean dataset
    st.markdown('### Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset
    
    impute_method = st.selectbox('Select an imputation method:',
                             options=['Zero', 'Mean', 'Median'])
    impute_dataset(df,impute_method)

    # Display updated dataframe
    
    # Descriptive Statistics 
    st.markdown('### Summary of Descriptive Statistics')
    a,b = compute_descriptive_stats(df)
    st.write(a)


    # Provide option to select multiple feature to show descriptive statistics using Streamit multiselect

    # Provide option to select multiple descriptive statistics to show using Streamit multiselect
 
    # Compute Descriptive Statistics including mean, median, min, max
    # ... = compute_descriptive_stats(...)
        
    # Display updated dataframe

    # Split train/test
    st.markdown('### Enter the percentage of test data to use for training the model')

    # Compute the percentage of test and training data
    train, test = split_dataset(df)

    # Print dataset split result
    st.write("Trainig dataset")
    st.write(train)
    st.write()
    st.write("Testing dataset")
    st.write(test)

    # Save state of train and test split
    df_train = train
    df_test = test
