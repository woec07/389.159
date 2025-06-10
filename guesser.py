# -*- coding: utf-8 -*-
# guesser.py

import os
import pandas as pd
import ipaddress
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

MODEL_PATH         = 'nn_model.pkl'
SCALER_PATH        = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

def prepare_data(df: pd.DataFrame):
    # Convert dotted IPs → 32-bit ints
    df['src_ip_int'] = df['sourceIPAddress'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
    df['dst_ip_int'] = df['destinationIPAddress'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))

    # Build feature matrix (drop flow keys + label columns)
    X = (
        df
        .drop(columns=[
            'flowStartMilliseconds',
            'sourceIPAddress', 'destinationIPAddress',
            'Binary_Label', 'Attack_Type_enc'
        ], errors='ignore')
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )

    # Encode target labels
    y = df['Attack_Type_enc']
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, scaler, label_encoder

def transform_data(df: pd.DataFrame, scaler: StandardScaler):
    df2 = df.copy()
    df2['src_ip_int'] = df2['sourceIPAddress'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
    df2['dst_ip_int'] = df2['destinationIPAddress'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))

    X = (
        df2
        .drop(columns=[
            'flowStartMilliseconds',
            'sourceIPAddress', 'destinationIPAddress'
        ], errors='ignore')
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )

    return scaler.transform(X)

def train_nn(X, y):
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=500,
        verbose=True,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

if __name__ == "__main__":
    # 1) Load datasets
    train_df = pd.read_csv("flowkeys_training_labeled_enc.csv")
    test_df  = pd.read_csv("test_clean_mod.csv")

    # 2) Either load or train model + preprocessors
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print("Found existing model artifacts, loading...")
        model         = joblib.load(MODEL_PATH)
        scaler        = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        print("No existing artifacts found, training new model...")
        X_train, y_train, scaler, label_encoder = prepare_data(train_df)
        model = train_nn(X_train, y_train)
        # save for next time
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        print(f"Saved new artifacts: {MODEL_PATH}, {SCALER_PATH}, {LABEL_ENCODER_PATH}")

    # 3) Predict on test set
    X_test       = transform_data(test_df, scaler)
    y_pred_enc   = model.predict(X_test)
    y_pred_str   = label_encoder.inverse_transform(y_pred_enc)
    y_pred_labels = ['C0' if lbl == 'Normal' else lbl for lbl in y_pred_str]

    # 4) Generate binary predictions: 0 for C0, 1 for any other class
    y_pred_bin = [0 if lbl == 'C0' else 1 for lbl in y_pred_labels]

    # 5) Write both columns into output.csv
    test_df['Binary_Label']     = y_pred_bin
    test_df['prediction'] = y_pred_labels
    test_df.to_csv("output.csv", index=False)
    print("Saved predictions and binary labels to output.csv")

    # 6) Report attack rate
    attack_rate = test_df['Binary_Label'].mean()
    print(f"{attack_rate:.4f} fraction → {attack_rate*100:.2f}% of flows are attacks")
