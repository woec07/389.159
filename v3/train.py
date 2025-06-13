import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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


def build_pipeline():
    """Pipeline with scikit-learn’s HistGradientBoostingClassifier for top performance."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,  # Reduced for faster training
            class_weight='balanced_subsample',
            max_depth=12,  # Reduced depth
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        ))
    ])

def main():
    print("=== NetSec Competition Model Training (Optimized) ===")

    # Load and merge data
    df = load_and_merge_data()

    # Create aggregated features (fast version)
    df = create_aggregated_features.agg_features(df)

    # Preprocess features
    X, y = preprocess_features.preproc(df, True)

    print(f"\nDataset info:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Attack distribution:\n{y.value_counts()}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\nLabel encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

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

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Save model and encoder
    joblib.dump(pipeline, 'flow_classifier.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print(f"\nModel saved successfully!")
    print("Files created:")
    print("- flow_classifier.pkl")
    print("- label_encoder.pkl")


if __name__ == '__main__':
    main()