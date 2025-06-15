#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:39:14 2025

@author: matthias
"""
import socket
import struct
import pandas as pd

def ip_to_int(ip_str: str) -> int:
    """Convert IP address string to integer"""
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip_str)))[0]
    except:
        return 0
    
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

# C1: Distributed resource exhaustion attack
def c1_detection(df: pd.DataFrame) -> pd.DataFrame:
    # C1 Detection: DDoS (many sources -> one destination)
    c1_stats = df.groupby('dst_ip_int').agg({
        'src_ip_int': 'nunique',
        'packetTotalCount': 'sum',
        'octetTotalCount': 'sum'
    }).rename(columns={
        'src_ip_int': 'unique_src_per_dst',
        'packetTotalCount': 'total_packets_to_dst',
        'octetTotalCount': 'total_bytes_to_dst'
    })

    return c1_stats

# C2: Horizontal network probing
def c2_detection(df: pd.DataFrame) -> pd.DataFrame:
    # C2 Detection: Horizontal scanning (one source -> many destinations, one port)
    c2_stats = df.groupby('src_ip_int').agg({
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

    return c2_stats

# C3: Vertical service enumeration
def c3_detection(df: pd.DataFrame) -> pd.DataFrame:
    # C3 Detection: Vertical scanning (one source -> one destination, many ports)
    grouped_main = df.groupby(['src_ip_int', 'dst_ip_int'])
    c3_stats = grouped_main.agg({
        'destinationTransportPort': 'nunique',
        'flowStartMilliseconds': 'count',
        'is_successful_connection': 'sum',
        'is_failed_connection': 'sum'
    }).rename(columns={
        'destinationTransportPort': 'c3_dst_port_diversity',
        'flowStartMilliseconds': 'c3_flow_count',
        'is_successful_connection': 'c3_successful_conns',
        'is_failed_connection': 'c3_failed_conns'
    })
    # extra feature 1: how many diverse ports are scanned
    c3_stats['c3_port_diversity_ratio'] = c3_stats['c3_dst_port_diversity'] / (c3_stats['c3_flow_count'] + 1)
    # extra feature 2: how the ratio failed/successfull connection is
    c3_stats['c3_failure_ratio'] = c3_stats['c3_failed_conns'] / (c3_stats['c3_successful_conns'] + 1)
    return c3_stats

# C4: Remote code execution via service vulnerability
def c4_detection(df: pd.DataFrame) -> pd.DataFrame:
    # C4 Detection: attacker tries to run code using a service port
    # service port is attacked, long flow duration, big payload
    vuln_ports = {
        21, 22, 23, 25, 53, 67, 68, 69,
        80, 110, 123, 137, 138, 139, 143,
        161, 162, 389, 443, 445, 512, 513,
        514, 2049, 3306, 3389, 5432, 5900,
        5901, 5902, 6379, 27017, 9200,
        1433, 1434, 1521, 2375, 6443
    }

    # prefiltering
    df['is_c4_candidate'] = df['destinationTransportPort'].isin(vuln_ports)
    df['is_long_flow'] = df['flowDurationMilliseconds'] > 10_000  # >10 s
    df['is_big_payload'] = df['octetTotalCount'] > df['octetTotalCount'].quantile(0.75)
    
    c4_stats = df[df['is_c4_candidate']].groupby('src_ip_int').agg({
        'is_c4_candidate': 'count',
        'is_long_flow': 'sum',
        'is_big_payload': 'sum',
    }).rename(columns={
        'is_c4_candidate': 'c4_flows_from_src',
        'is_long_flow': 'c4_long_flows_from_src',
        'is_big_payload': 'c4_big_payload_flows_from_src',
    })
    # extra feature 1: how many flows very long flows from this source
    c4_stats['c4_long_ratio_src'] = c4_stats['c4_long_flows_from_src'] / (c4_stats['c4_flows_from_src'] + 1)
    # extra feature 2: how many flows very big packet flows from this source
    c4_stats['c4_big_payload_ratio_src'] = c4_stats['c4_big_payload_flows_from_src'] / (c4_stats['c4_flows_from_src'] + 1)

    return c4_stats

# C5: Amplified overload using exposed service
# ----------------------------------------------------------------------------
# C5 Detection: Attacker spoofes victims IP address -> sends small request
# to a public service -> public service answers with large replies, which
# flood the victim
# ----------------------------------------------------------------------------
# C5 Detection is not implemented as groundtruth.csv does not contain any C5 cases,
# hence it is not possible to train our model for C5 data


# C6: Connection slot saturation via slow requests
# ----------------------------------------------------------------------------
# C6 Detection: 
# ----------------------------------------------------------------------------
# no C6 detection implemented as the relative part of C6 errors is relatively 
# small, and it would use the same feature as C1 does -> induces wrong labeling
# between C1 and C6


# C7: Malicious input to extract or manipulate data
# ----------------------------------------------------------------------------
# C7 Detection: C7 attacks target application-layer services (e.g., HTTP, SQL, LDAP)
# by sending requests to extract or manipulate data (e.g., SQL injection).
# how to find: only at service ports (80, 8080); medium packet size (15- 20),
# short flow duration
# ----------------------------------------------------------------------------
# no C7 detection implemented out of time reasons (and model is already good), 
# as a relatively small percentage is covered by C7 faults

# C8: Remote code execution via file-transfer service vulnerability
# ----------------------------------------------------------------------------
# C8 Detection: long flow duration, high payload -> protocols like FTP, SMB, etc.
# ----------------------------------------------------------------------------
# C8 Detection is not implemented as groundtruth.csv does not contain any C5 
# cases, hence it is not possible to train our model for C5 data

def agg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated features"""

    print("Computing aggregated features...")
    df = df.copy()

    # Convert IP addresses to integers for processing
    df['src_ip_int'] = df['sourceIPAddress'].apply(ip_to_int)
    df['dst_ip_int'] = df['destinationIPAddress'].apply(ip_to_int)
    
    # Parse flags directly for aggregation
    flag_parsed = df['_tcpFlags'].apply(parse_tcp_flags)
    flag_df = pd.DataFrame(flag_parsed.tolist())
    
    df['has_syn'] = flag_df['has_syn']
    df['has_ack'] = flag_df['has_ack']
    df['is_successful_connection'] = df['has_syn'] & df['has_ack']
    df['is_failed_connection'] = df['has_syn'] & (~df['has_ack'])

    c1_features = c1_detection(df)
    c2_features = c2_detection(df)
    c3_features = c3_detection(df)
    c4_features = c4_detection(df)

    # Merge back to original dataframe
    df = df.merge(c1_features, left_on='dst_ip_int', right_index=True, how='left')
    df = df.merge(c2_features, left_on='src_ip_int', right_index=True, how='left')
    # Fill NaN values (for single-flow sources/destinations)
    df['unique_dst_per_src'] = df['unique_dst_per_src'].fillna(1)
    df['unique_src_per_dst'] = df['unique_src_per_dst'].fillna(1)
    df['dst_port_diversity'] = df['dst_port_diversity'].fillna(1)
    df['total_packets_from_src'] = df['total_packets_from_src'].fillna(df['packetTotalCount'])
    df['total_bytes_from_src'] = df['total_bytes_from_src'].fillna(df['octetTotalCount'])
    df['total_packets_to_dst'] = df['total_packets_to_dst'].fillna(df['packetTotalCount'])
    df['total_bytes_to_dst'] = df['total_bytes_to_dst'].fillna(df['octetTotalCount'])
    df['src_scan_reputation'] = df['unique_dst_per_src'] * df['dst_port_diversity']
    
    df = df.merge(c3_features, left_on=['src_ip_int', 'dst_ip_int'], right_index=True, how='left').fillna({
        'c3_dst_port_diversity': 0,
        'c3_flow_count': 0,
        'c3_successful_conns': 0,
        'c3_failed_conns': 0,
        'c3_port_diversity_ratio': 0,
        'c3_failure_ratio': 0,
    })
    
    df = df.merge(c4_features, left_on='src_ip_int', right_index=True, how='left').fillna({
        'c4_flows_from_src': 0,
        'c4_long_flows_from_src': 0,
        'c4_big_payload_flows_from_src': 0,
        'c4_long_ratio_src': 0,
        'c4_big_payload_ratio_src': 0,
    })
    
    return df
