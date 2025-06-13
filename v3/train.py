import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

import create_aggregated_features
import preprocess_features

def load_and_merge_data():
    """Load training labels and detailed flow data"""
    print("Loading training data...")

    labels_df = pd.read_csv('flowkeys_training_labeled_enc.csv')
    flows_df = pd.read_csv('flowkeys_training_detailed.csv')

    # Merge on flow key
    merge_keys = [
        'sourceIPAddress', 'destinationIPAddress',
        'sourceTransportPort', 'destinationTransportPort',
        'flowStartMilliseconds'
    ]

    merged_df = pd.merge(labels_df, flows_df, on=merge_keys, how='inner')
    print(f"Merged dataset shape: {merged_df.shape}")
    return merged_df


def build_and_tune_pipeline():
    """Builds a pipeline and sets up a GridSearchCV for tuning."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(
            class_weight='balanced',
            random_state=42,
            learning_rate=0.1,
            max_leaf_nodes=100,
        ))
    ])

    # Define the parameter grid to search
    # Start with a small grid, expand if you have time
    param_grid = {
        'classifier__learning_rate': [0.1, 0.2],
        'classifier__max_leaf_nodes': [50, 100, 500],
        'classifier__l2_regularization': [0.0, 1.0]
    }

    # Set up the grid search with cross-validation
    # cv=3 means 3-fold cross-validation. n_jobs=-1 uses all CPU cores.
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    return search


def build_pipeline():
    """Pipeline with scikit-learn HistGradientBoostingClassifier"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(
            class_weight='balanced',
            max_leaf_nodes=100,
            learning_rate=0.05,
            max_iter=200,
            random_state=42
        ))
    ])

def main():
    print("=== NetSec Competition Model Training ===")

    df = load_and_merge_data()
    df = create_aggregated_features.agg_features(df)
    X, y = preprocess_features.preproc(df, True)

    print(f"\nDataset info:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Attack distribution:\n{y.value_counts()}")

    label_map = {
        'Normal': 0,
        'C1': 1,
        'C2': 2,
        'C3': 3,
        'C4': 4,
        'C5': 5,
        'C6': 6,
        'C7': 7,
    }
    y_encoded = y.map(label_map)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Build and train pipeline
    print(f"\nTraining model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    print(f"\nEvaluating model...")
    y_pred = pipeline.predict(X_val)

    reverse_label_map = {v: k for k, v in label_map.items()}
    target_names_list = [reverse_label_map[i] for i in sorted(reverse_label_map)]

    print("\nClassification Report:")
    # Use the 'labels' parameter to force the report to have 8 rows
    print(classification_report(
        y_val,
        y_pred,
        target_names=target_names_list,
        labels=list(range(len(label_map))),
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    joblib.dump(pipeline, 'flow_classifier.pkl')
    joblib.dump(reverse_label_map, 'label_mapping.pkl')

    print(f"\nModel and label mapping saved successfully!")
    print("Files created:")
    print("- flow_classifier.pkl")
    print("- label_mapping.pkl")


if __name__ == '__main__':
    main()