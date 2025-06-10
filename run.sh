#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input-test.pcap>"
  exit 1
fi

PCAP="$1"
CSV_OUT="output-test.csv"
TEST_CSV="test_clean_mod.csv"

echo "Extracting flows from '$PCAP' into '$CSV_OUT'..."
go-flows run features 4tuple_bidi.json export csv "$CSV_OUT" source libpcap "$PCAP"

echo "Preparing test CSV for Python..."
mv -f "$CSV_OUT" "$TEST_CSV"

echo "Running classifier..."
python3 guesser.py

echo "Done.  Class predictions written to output.csv"
