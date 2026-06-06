#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setting up data directory..."
mkdir -p data/data-multi-subject
mkdir -p data/cache
mkdir -p data/metrics

# Only download participants.tsv if not already present
if [ ! -f "data/data-multi-subject/participants.tsv" ]; then
    echo "Fetching participants metadata..."
    curl -L \
      "https://raw.githubusercontent.com/spine-generic/data-multi-subject/master/participants.tsv" \
      -o data/data-multi-subject/participants.tsv
    echo "Metadata ready."
else
    echo "Metadata already present."
fi

echo ""
echo "   Setup complete."
echo "   MRI files will be downloaded on demand when a subject is selected."
echo "   Metric CSV/TSV files can be uploaded in the app or placed in data/metrics."
echo "   Starting app..."
