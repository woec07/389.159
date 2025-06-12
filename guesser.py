#!/usr/bin/env python3
import sys
import pandas as pd


def classifier_C1(df):
    # Group by destination IP and count distinct source IPs
    result = (
        df.groupby('destinationIPAddress')['sourceIPAddress']
          .nunique()
          .reset_index(name='distinctSourceCount')
    )
    # Compute global mean and standard deviation
    mean = result['distinctSourceCount'].mean()
    std = result['distinctSourceCount'].std()
    
    threshold = mean + 35 * std

    # 4) Identify attacked destination IPs
    attack_ips = set(
        result.loc[result['distinctSourceCount'] > threshold, 'destinationIPAddress']
    )
    
    # 5) Annotate the full df
    df['Binary_Label'] = df['destinationIPAddress']\
        .isin(attack_ips)\
        .astype(int)
    df['prediction']   = df['Binary_Label']\
        .apply(lambda x: 'C1' if x == 1 else '')
        
    return df

def classifier_C2(df):
    
    result = (
        df.groupby(['sourceIPAddress', 'destinationTransportPort'])['destinationIPAddress']
          .nunique()
          .reset_index(name='distinctDestCount')
    )
    # Compute global mean and standard deviation
    mean = result['distinctDestCount'].mean()
    std = result['distinctDestCount'].std()
    threshold = mean + 1 * std
    print(mean)
    print(threshold)
    
    filtered = result[result['distinctDestCount'] > threshold]
    
    print(filtered)
    print(filtered['distinctDestCount'].sum())

    
def classifier_C3(df):
    result = (
        df.groupby(['sourceIPAddress', 'destinationIPAddress'])['destinationTransportPort']
          .nunique()
          .reset_index(name='distinctDestPortCount')
    )
    # Compute global mean and standard deviation
    mean = result['distinctDestPortCount'].mean()
    std = result['distinctDestPortCount'].std()
    threshold = mean + 50 * std
    print(mean)
    print(threshold)
    
    filtered = result[result['distinctDestPortCount'] > threshold]
    
    print(filtered)
    print(filtered['distinctDestPortCount'].sum())


if __name__ == '__main__':
    data = pd.read_csv("test_clean_mod.csv")
    
    data = classifier_C1(data) # apply classifier C1
    data = classifier_C3(data)
    
    print
    print("Sample of annotated flows:")
    print(data.head(10))

