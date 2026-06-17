import pandas as pd
from colorama import Fore,init
# Initialize colorama for colored console output
init(autoreset=True)
import time
from warnings import filterwarnings
# Suppress specific warnings from scikit-learn and matplotlib to keep output clean
filterwarnings('ignore',category=UserWarning,module='sklearn')
filterwarnings('ignore',category=UserWarning,module='matplotlib')

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline # Using imblearn's Pipeline for SMOTE integration
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(filepath = 'new_masage_dataset'):
    '''
    Loads the dataset, preprocesses labels, and splits data into training and testing sets.

    Args:
        filepath (str): The path to the CSV dataset file.

    Returns:
        tuple: x_train, x_test, y_train, y_test (pandas DataFrames/Series and numpy arrays)
               The split features and labels for training and testing.
    '''

    df = pd.read_csv(filepath)
    x=df.drop(columns='label')
    y=df['label']

    label_encoding = LabelEncoder()
    
    # Split the data into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                     shuffle=True,stratify=y,
                                                     test_size=0.2,random_state=42)
    # Encode the target labels (e.g., 'spam'/'ham' to 0/1)
    y_train = label_encoding.fit_transform(y_train)
    y_test = label_encoding.transform(y_test)

    return x_train,x_test,y_train,y_test


def get_model_pipeline():
    '''
    Creates and returns a machine learning pipeline for text classification.
    The pipeline includes text vectorization, feature scaling, SMOTE for imbalance, and an LGBMClassifier.

    Returns:
        imblearn.pipeline.Pipeline: The configured machine learning pipeline.
    '''
    # FeatureUnion to apply CountVectorizer and TfidfVectorizer in parallel on the 'text' column
    vector_trans = FeatureUnion([                                   #use a FeatureUnion for apply parallel processing and take output in one time.
        ('words_count',CountVectorizer(analyzer='word')),
        ('vectorized',TfidfVectorizer(max_features=5000,
                                       ngram_range=(1,2),
                                       min_df = 2))
    ])
    
    # Pipeline for numerical feature scaling
    scaler_pipe = Pipeline([
        ('scale',StandardScaler())])
    
    # ColumnTransformer to apply different preprocessing steps to different columns
    process = ColumnTransformer(transformers=[
        ('words_pre_process',vector_trans,'text'), # Apply text vectorizers to the 'text' column
        ('scaler',scaler_pipe,['char_count', 'digit_count', 'uppercase_words']) # Apply scaler to numerical features
    ],remainder='passthrough')
    
    # Main model pipeline including preprocessing, sampling, and the classifier
    model_pipe = Pipeline([
        ('processors',process), # Apply the ColumnTransformer for feature processing
        ('sampling',SMOTE(random_state=42)), # Apply SMOTE to handle class imbalance
        ('model',LGBMClassifier(random_state=42)) # The LightGBM classifier
    ])

    return model_pipe


def train_and_save(model_path = 'massage_clf.joblib'):
    '''
    Trains the full machine learning pipeline using the loaded data and saves the fitted model.
    It also evaluates the model performance on the test set.

    Args:
        model_path (str): The filename to save the trained model.
    '''
    # Load and preprocess the data
    x_train,x_test,y_train,y_test = load_and_preprocess_data()
    
    # Initialize the model pipeline
    print(f'{Fore.GREEN} initializing a pipeline')
    pipeline = get_model_pipeline()
    
    # Simulate a delay for user experience
    time.sleep(3)
    print(f'{Fore.RED} training the model')
    # Train the pipeline on the training data
    pipeline.fit(x_train,y_train)
    
    # Simulate a delay
    time.sleep(3)
    # Evaluate model performance on the test set
    y_pred = pipeline.predict(x_test)
    class_report = classification_report(y_test,y_pred)
    confusion_mat = confusion_matrix(y_test,y_pred)
    
    # Print performance metrics
    print(f'{Fore.YELLOW} model performance')
    print(class_report)
    print(confusion_mat)
    
    # Save the trained pipeline to disk
    time.sleep(4)
    print(f'{Fore.LIGHTCYAN_EX} saving the model')
    joblib.dump(pipeline,model_path)
    print(f'{Fore.LIGHTMAGENTA_EX} model saved successfully')


if __name__ =='__main__': # Entry point of the script
    train_and_save()
