#!/bin/bash

PCAP_FILE=$1

./go-flows run features detailed_bidi.json export csv flows.csv source libpcap -- "$PCAP_FILE"

python3 classify.py flows.csv output.csv

rm -f flows.csv