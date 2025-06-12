import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import socket
import struct


def ip_to_int(ip_str):
    """Convert IP address string to integer"""
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip_str)))[0]
    except:
        return 0


def parse_tcp_flags(tcp_flags_str):
    """Parse TCP flags string (e.g., 'SA', 'S', 'RA') into individual flag presence"""
    if pd.isna(tcp_flags_str) or tcp_flags_str == '':
        return {
            'has_syn': False,
            'has_ack': False,
            'has_fin': False,
            'has_rst': False,
        }

    tcp_flags_str = str(tcp_flags_str).upper()
    return {
        'has_syn': 'S' in tcp_flags_str,
        'has_ack': 'A' in tcp_flags_str,
        'has_fin': 'F' in tcp_flags_str,
        'has_rst': 'R' in tcp_flags_str,
    }


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


def create_aggregated_features_fast(df):
    """Create aggregated features using efficient pandas operations"""
    df = df.copy()

    # Convert IP addresses to integers for processing
    df['src_ip_int'] = df['sourceIPAddress'].apply(ip_to_int)
    df['dst_ip_int'] = df['destinationIPAddress'].apply(ip_to_int)

    print("Computing aggregated features (fast version)...")

    # C2 Detection: Horizontal scanning (one source -> many destinations)
    src_stats = df.groupby('src_ip_int').agg({
        'dst_ip_int': 'nunique',
        'destinationTransportPort': 'nunique',
        'packetTotalCount': 'sum',
        'octetTotalCount': 'sum'
    }).rename(columns={
        'dst_ip_int': 'unique_dst_per_src',
        'destinationTransportPort': 'dst_port_diversity',
        'packetTotalCount': 'total_packets_from_src',
        'octetTotalCount': 'total_bytes_from_src'
    })

    # C1 Detection: DDoS (many sources -> one destination)
    dst_stats = df.groupby('dst_ip_int').agg({
        'src_ip_int': 'nunique',
        'packetTotalCount': 'sum',
        'octetTotalCount': 'sum'
    }).rename(columns={
        'src_ip_int': 'unique_src_per_dst',
        'packetTotalCount': 'total_packets_to_dst',
        'octetTotalCount': 'total_bytes_to_dst'
    })

    # Merge back to original dataframe
    df = df.merge(src_stats, left_on='src_ip_int', right_index=True, how='left')
    df = df.merge(dst_stats, left_on='dst_ip_int', right_index=True, how='left')

    # Fill NaN values (for single-flow sources/destinations)
    df['unique_dst_per_src'] = df['unique_dst_per_src'].fillna(1)
    df['unique_src_per_dst'] = df['unique_src_per_dst'].fillna(1)
    df['dst_port_diversity'] = df['dst_port_diversity'].fillna(1)
    df['total_packets_from_src'] = df['total_packets_from_src'].fillna(df['packetTotalCount'])
    df['total_bytes_from_src'] = df['total_bytes_from_src'].fillna(df['octetTotalCount'])
    df['total_packets_to_dst'] = df['total_packets_to_dst'].fillna(df['packetTotalCount'])
    df['total_bytes_to_dst'] = df['total_bytes_to_dst'].fillna(df['octetTotalCount'])

    return df


def preprocess_features(df):
    """Select and preprocess features for training"""

    # Available go-flows features
    base_features = [
        'flowDurationMilliseconds',
        'packetTotalCount',
        'octetTotalCount',
        'protocolIdentifier',
        'destinationTransportPort',
        'sourceTransportPort'
    ]

    # Simplified aggregated features (much faster to compute)
    agg_features = [
        'unique_dst_per_src',  # C2: horizontal scanning
        'unique_src_per_dst',  # C1: DDoS
        'dst_port_diversity',  # C3: vertical scanning
        'total_packets_from_src',  # C2: scanning volume
        'total_bytes_from_src',  # C2: scanning volume
        'total_packets_to_dst',  # C1: DDoS volume
        'total_bytes_to_dst'  # C1: DDoS volume
    ]

    # Combine all features
    all_features = base_features + agg_features

    # Select available features
    available_features = [f for f in all_features if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = df[available_features].fillna(0)
    y = df['Attack_Type_enc']

    # Create derived features
    X['avg_packet_size'] = X['octetTotalCount'] / (X['packetTotalCount'] + 1)
    X['packets_per_second'] = X['packetTotalCount'] / (X['flowDurationMilliseconds'] + 1) * 1000
    X['bytes_per_second'] = X['octetTotalCount'] / (X['flowDurationMilliseconds'] + 1) * 1000

    # Scanning intensity ratios
    X['dst_per_src_ratio'] = X['unique_dst_per_src'] / (X['total_packets_from_src'] + 1)
    X['src_per_dst_ratio'] = X['unique_src_per_dst'] / (X['total_packets_to_dst'] + 1)

    # TCP flag analysis (simplified)
    if '_tcpFlags' in df.columns:
        tcp_flag_features = df['_tcpFlags'].apply(parse_tcp_flags)
        tcp_flag_df = pd.DataFrame(tcp_flag_features.tolist())

        # Add only the most important TCP flag features
        X['has_syn'] = tcp_flag_df['has_syn'].astype(int)
        X['has_rst'] = tcp_flag_df['has_rst'].astype(int)
        X['is_syn_only'] = (tcp_flag_df['has_syn'] & (~tcp_flag_df['has_ack'])).astype(int)

    # Port analysis (simplified)
    X['is_well_known_port'] = (X['destinationTransportPort'] < 1024).astype(int)
    X['is_http_port'] = (X['destinationTransportPort'].isin([80, 443])).astype(int)
    X['is_ssh_port'] = (X['destinationTransportPort'] == 22).astype(int)

    # Protocol analysis
    X['is_tcp'] = (X['protocolIdentifier'] == 6).astype(int)
    X['is_udp'] = (X['protocolIdentifier'] == 17).astype(int)
    X['is_icmp'] = (X['protocolIdentifier'] == 1).astype(int)

    return X, y


def build_pipeline():
    """Build machine learning pipeline"""
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
    df = create_aggregated_features_fast(df)

    # Preprocess features
    X, y = preprocess_features(df)

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