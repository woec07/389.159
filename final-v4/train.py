import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import  StandardScaler
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

def relabel_c3_data(df):
    pair_stats = df.groupby(['sourceIPAddress', 'destinationIPAddress']).agg({
        'destinationTransportPort': 'nunique',
        'flowStartMilliseconds': 'count'
    }).rename(columns={
        'destinationTransportPort': 'unique_ports',
        'flowStartMilliseconds': 'total_flows'
    }).reset_index()

    valid_pairs = pair_stats[
        (pair_stats['unique_ports'] >= 50) &
        (pair_stats['total_flows'] >= 50)
    ]
    valid_pairs_set = set(zip(valid_pairs['sourceIPAddress'], valid_pairs['destinationIPAddress']))

    # Build the correct C3 flow mask
    c3_mask_base = (
        (df['flowDurationMilliseconds'] == 0) &
        (df['packetTotalCount'] == 1) &
        (df['octetTotalCount'] == 44) &
        (df['protocolIdentifier'] == 6) &
        (df['_tcpFlags'] == 'S') 
    )
    
    # mask of tuples
    pair_mask = df.apply(lambda row: (row['sourceIPAddress'], row['destinationIPAddress']) in valid_pairs_set, axis=1)
    # Combine both masks
    full_c3_mask = c3_mask_base & pair_mask
    # relabel
    df.loc[full_c3_mask, 'Attack_Type_enc'] = 'C3'
    return df


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
    
    # needed as ground truth is wrong for C3 (-> relabeling for training needed)
    df = relabel_c3_data(df)
    df = create_aggregated_features.agg_features(df)
    X, y = preprocess_features.preproc(df, True)

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
        'C8': 8,
    }
    y_encoded = y.map(label_map)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
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

    print("Files created:")
    print("- flow_classifier.pkl")
    print("- label_mapping.pkl")


if __name__ == '__main__':
    main()