import sys

import pandas as pd
import numpy as np
import joblib
import socket
import struct

def ip_to_int(ip_str):
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip_str)))[0]
    except:
        return 0


def parse_tcp_flags(tcp_flags_str):
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


def create_aggregated_features(df):
    df = df.copy()

    df['src_ip_int'] = df['sourceIPAddress'].apply(ip_to_int)
    df['dst_ip_int'] = df['destinationIPAddress'].apply(ip_to_int)

    print("Computing aggregated features...")

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

    dst_stats = df.groupby('dst_ip_int').agg({
        'src_ip_int': 'nunique',
        'packetTotalCount': 'sum',
        'octetTotalCount': 'sum'
    }).rename(columns={
        'src_ip_int': 'unique_src_per_dst',
        'packetTotalCount': 'total_packets_to_dst',
        'octetTotalCount': 'total_bytes_to_dst'
    })

    df = df.merge(src_stats, left_on='src_ip_int', right_index=True, how='left')
    df = df.merge(dst_stats, left_on='dst_ip_int', right_index=True, how='left')

    df['unique_dst_per_src'] = df['unique_dst_per_src'].fillna(1)
    df['unique_src_per_dst'] = df['unique_src_per_dst'].fillna(1)
    df['dst_port_diversity'] = df['dst_port_diversity'].fillna(1)
    df['total_packets_from_src'] = df['total_packets_from_src'].fillna(df['packetTotalCount'])
    df['total_bytes_from_src'] = df['total_bytes_from_src'].fillna(df['octetTotalCount'])
    df['total_packets_to_dst'] = df['total_packets_to_dst'].fillna(df['packetTotalCount'])
    df['total_bytes_to_dst'] = df['total_bytes_to_dst'].fillna(df['octetTotalCount'])

    return df


def preprocess_features(df):
    base_features = [
        'flowDurationMilliseconds',
        'packetTotalCount',
        'octetTotalCount',
        'protocolIdentifier',
        'destinationTransportPort',
        'sourceTransportPort'
    ]

    agg_features = [
        'unique_dst_per_src',
        'unique_src_per_dst',
        'dst_port_diversity',
        'total_packets_from_src',
        'total_bytes_from_src',
        'total_packets_to_dst',
        'total_bytes_to_dst'
    ]

    all_features = base_features + agg_features
    available_features = [f for f in all_features if f in df.columns]

    X = df[available_features].fillna(0)

    X['avg_packet_size'] = X['octetTotalCount'] / (X['packetTotalCount'] + 1)
    X['packets_per_second'] = X['packetTotalCount'] / (X['flowDurationMilliseconds'] + 1) * 1000
    X['bytes_per_second'] = X['octetTotalCount'] / (X['flowDurationMilliseconds'] + 1) * 1000

    X['dst_per_src_ratio'] = X['unique_dst_per_src'] / (X['total_packets_from_src'] + 1)
    X['src_per_dst_ratio'] = X['unique_src_per_dst'] / (X['total_packets_to_dst'] + 1)

    if '_tcpFlags' in df.columns:
        tcp_flag_features = df['_tcpFlags'].apply(parse_tcp_flags)
        tcp_flag_df = pd.DataFrame(tcp_flag_features.tolist())

        X['has_syn'] = tcp_flag_df['has_syn'].astype(int)
        X['has_rst'] = tcp_flag_df['has_rst'].astype(int)
        X['is_syn_only'] = (tcp_flag_df['has_syn'] & (~tcp_flag_df['has_ack'])).astype(int)

    X['is_well_known_port'] = (X['destinationTransportPort'] < 1024).astype(int)
    X['is_http_port'] = (X['destinationTransportPort'].isin([80, 443])).astype(int)
    X['is_ssh_port'] = (X['destinationTransportPort'] == 22).astype(int)

    X['is_tcp'] = (X['protocolIdentifier'] == 6).astype(int)
    X['is_udp'] = (X['protocolIdentifier'] == 17).astype(int)
    X['is_icmp'] = (X['protocolIdentifier'] == 1).astype(int)

    return X


def main():
    if len(sys.argv) == 3:
        # correct number of arguments provided (e.g., by run.sh)
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 1:
        # defaults
        input_file = "flowkeys_training_detailed.csv"
        output_file = "output.csv"
        print(f"No arguments provided. Using defaults: input='{input_file}', output='{output_file}'")
    else:
        print("Usage: python3 classifier.py <input_csv> <output_csv>")
        print("Or run without arguments to use default filenames.")
        sys.exit(1)

    print(f"Loading flows from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} flows")

    pipeline = joblib.load('flow_classifier.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    df = create_aggregated_features(df)

    print("Preprocessing features...")
    X = preprocess_features(df)

    print("Making predictions...")
    y_pred = pipeline.predict(X)

    predictions = label_encoder.inverse_transform(y_pred)

    output_df = df[[
        'flowStartMilliseconds',
        'sourceIPAddress',
        'destinationIPAddress',
        'sourceTransportPort',
        'destinationTransportPort'
    ]].copy()

    output_df['Binary_Label'] = (predictions != 'Normal').astype(int)
    output_df['prediction'] = predictions

    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    pred_counts = pd.Series(predictions).value_counts()
    print(f"\nPrediction summary:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count}")


if __name__ == '__main__':
    main()