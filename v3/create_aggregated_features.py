#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:39:14 2025

@author: matthias
"""
import socket
import struct
import pandas as pd


def ip_to_int(ip_str):
    """Convert IP address string to integer"""
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip_str)))[0]
    except:
        return 0

def c1_detection(df):
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
        
    return dst_stats

def c2_detection(df):
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
    
    return src_stats


def c3_detection(df):

    df['is_banner_grab'] = (
        (df['flowDurationMilliseconds'] < 100)      # very brief
        & (df['packetTotalCount'] == 1)             # exactly one packet
        & (df['octetTotalCount']  < 50)            # tiny payload
    )
    # C3 Detection: Vertical scanning (one source -> one destination, many different ports)
    c3_stats = df[df['is_banner_grab']].groupby(['src_ip_int','dst_ip_int']).agg({
        'destinationTransportPort': 'nunique',
        'packetTotalCount': 'sum',
        'octetTotalCount': 'sum',
        'flowStartMilliseconds': 'count',
    }).rename(columns={
        'destinationTransportPort': 'c3_ports_per_scan',
        'packetTotalCount': 'c3_total_packets_per_scan',
        'octetTotalCount': 'c3_total_bytes_per_scan',
        'flowStartMilliseconds': 'c3_flows_per_scan',
    })
        
    c3_stats['c3_ports_per_flow'] = (
        c3_stats['c3_ports_per_scan'] 
        / c3_stats['c3_flows_per_scan'].replace(0, 1)
    )
    return c3_stats

def c4_detection(df):
    # C4 Detection:
    vuln_ports = {
    21, 22, 23, 25, 53, 67, 68, 69,
    80, 110, 123, 137, 138, 139, 143,
    161, 162, 389, 443, 445, 512, 513,
    514, 2049, 3306, 3389, 5432, 5900,
    5901, 5902, 6379, 27017, 9200,
    1433, 1434, 1521, 2375, 6443
    }
    
    df['is_c4_candidate'] = df['destinationTransportPort'].isin(vuln_ports)
    df['is_long_flow']    = df['flowDurationMilliseconds'] > 10_000  # >10 s
    df['is_big_payload']  = df['octetTotalCount'] > df['octetTotalCount'].quantile(0.75)

    # Per-source C4 patterns
    c4_src = df[df['is_c4_candidate']].groupby('src_ip_int').agg({
        'is_c4_candidate': 'count',
        'is_long_flow':    'sum',
        'is_big_payload':  'sum',
    }).rename(columns={
        'is_c4_candidate': 'c4_flows_from_src',
        'is_long_flow':    'c4_long_flows_from_src',
        'is_big_payload':  'c4_big_payload_flows_from_src',
    })
    c4_src['c4_long_ratio_src']       = c4_src['c4_long_flows_from_src'] / (c4_src['c4_flows_from_src'] + 1)
    c4_src['c4_big_payload_ratio_src']= c4_src['c4_big_payload_flows_from_src'] / (c4_src['c4_flows_from_src'] + 1)

    return c4_src
    

def agg_features(df):
    """Create aggregated features using efficient pandas operations"""
    
    print("Computing aggregated features (fast version)...")
    df = df.copy()

    # Convert IP addresses to integers for processing
    df['src_ip_int'] = df['sourceIPAddress'].apply(ip_to_int)
    df['dst_ip_int'] = df['destinationIPAddress'].apply(ip_to_int)
    
    c1_features = c1_detection(df)
    c2_features = c2_detection(df)
    #c3_features = c3_detection(df)
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
    #df = df.merge(c3_features, left_on=['src_ip_int','dst_ip_int'], right_index=True, how='left'
    #).fillna({
    #    'c3_ports_per_scan': 0,
    #    'c3_total_packets_per_scan': 0,
    #    'c3_total_bytes_per_scan': 0,
    #    'c3_flows_per_scan': 0,
    #    'c3_ports_per_flow': 0,   
    #})
    df = df.merge(c4_features, left_on='src_ip_int', right_index=True, how='left').fillna({
        'c4_flows_from_src': 0,
        'c4_long_flows_from_src':0,
        'c4_big_payload_flows_from_src':0,
        'c4_long_ratio_src':0,
        'c4_big_payload_ratio_src':0,
    })
    

    return df

