import sys

import pandas as pd
import joblib

import create_aggregated_features
import preprocess_features

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

    df = create_aggregated_features.agg_features(df)

    print("Preprocessing features...")
    X = preprocess_features.preproc(df, False)

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