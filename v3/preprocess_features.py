#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:51:09 2025

@author: matthias
"""
import pandas as pd

def parse_tcp_flags(tcp_flags_str: str) -> dict:
    """Parse TCP flags string (e.g., 'SA', 'S', 'RA') into individual flag presence"""
    if pd.isna(tcp_flags_str) or tcp_flags_str == '':
        return {
            'has_syn': False,
            'has_ack': False,
            'has_fin': False,
            'has_rst': False,
            'has_psh': False,
            'has_urg': False,
        }

    f = str(tcp_flags_str).upper()
    return {
        'has_syn': 'S' in f,
        'has_ack': 'A' in f,
        'has_fin': 'F' in f,
        'has_rst': 'R' in f,
        'has_psh': 'P' in f,
        'has_urg': 'U' in f,
    }

def preproc(df: pd.DataFrame, flag=True) -> (pd.DataFrame, pd.DataFrame):
    """Select and preprocess features for training"""

    # go-flows features
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
        'total_bytes_to_dst',       
        'src_scan_reputation',      
        'c3_dst_port_diversity',
        'c3_flow_count',
        'c3_successful_conns',
        'c3_failed_conns',
        'c3_scan_intensity',
        'c3_port_diversity_ratio',
        'c3_failure_ratio',
        'c4_flows_from_src',
        'c4_long_flows_from_src',
        'c4_big_payload_flows_from_src',
        'c4_long_ratio_src',
        'c4_big_payload_ratio_src',
    ]

    all_features = base_features + agg_features

    available_features = [f for f in all_features if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = df[available_features].fillna(0)

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

    if flag:
        y = df['Attack_Type_enc']
        return X, y
    else:
        return X
