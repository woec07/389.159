#!/bin/bash

# --- Configuration ---
# List of all files required for the submission
FILES_TO_PACKAGE=(
    "run.sh"
    "classify.py"
    "detailed_bidi.json"
    "go-flows"
    "flow_classifier.pkl"
    "label_encoder.pkl"
)
# The name of the final zip file
ZIP_NAME="v3.zip"

# --- Color Codes for Logging ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting submission packaging process...${NC}"

# --- 1. Verify all required files exist ---
echo "Verifying required files..."
HAS_ERROR=0
for file in "${FILES_TO_PACKAGE[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "  ${RED}Error: Required file '$file' not found!${NC}"
        HAS_ERROR=1
    else
        echo -e "  ${GREEN}Found: $file${NC}"
    fi
done

# Specific check for .pkl files with helpful message
if [ ! -f "flow_classifier.pkl" ] || [ ! -f "label_encoder.pkl" ]; then
    echo -e "${YELLOW}Hint: Model files are missing. Run 'python3 train_model.py' to generate them.${NC}"
fi

# --- 2. Verify run.sh is executable ---
echo "Verifying permissions..."
if [ -f "run.sh" ] && [ ! -x "run.sh" ]; then
    echo -e "  ${RED}Error: 'run.sh' is not executable!${NC}"
    echo -e "${YELLOW}Hint: Run 'chmod +x run.sh' to fix this.${NC}"
    HAS_ERROR=1
else
    echo -e "  ${GREEN}OK: run.sh is executable.${NC}"
fi

# --- 3. Exit if any errors were found ---
if [ $HAS_ERROR -ne 0 ]; then
    echo -e "\n${RED}Packaging failed due to missing files or incorrect permissions. Please fix the errors above.${NC}"
    exit 1
fi

# --- 4. Create the ZIP archive ---
echo -e "\nAll checks passed. Creating ZIP archive..."
# Remove old zip file to ensure a clean package
rm -f "$ZIP_NAME"

zip "$ZIP_NAME" "${FILES_TO_PACKAGE[@]}"

# --- 5. Final Verification ---
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Success! Submission package '$ZIP_NAME' created.${NC}"
    echo "It contains the following files:"
    unzip -l "$ZIP_NAME"
    echo -e "\n${YELLOW}You can now upload '$ZIP_NAME' to the competition webpage.${NC}"
else
    echo -e "\n${RED}Error: Failed to create '$ZIP_NAME'.${NC}"
    exit 1
fi