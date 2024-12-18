#!/bin/bash
set -e 

# Install required packages
echo "Platform:" $(uname )
echo "Python version: $(python --version)"

# If cuda is installed, check the version
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "Found GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Found CUDA version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"

    FOUND_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | sed 's/\.//g')
    if (( $FOUND_CUDA >= 124 )); then
        CUDA_VERSION="cu124"
    elif (( $FOUND_CUDA >= 121 )); then
        CUDA_VERSION="cu121"
    elif (( $FOUND_CUDA >= 118 )); then
        CUDA_VERSION="cu118"
    else
      CUDA_VERSION="cpu"
      echo "CUDA version ${FOUND_CUDA} is not compatible. Installing PyTorch without CUDA support."
    fi
else
    echo "No GPU or CUDA installation found. Installing PyTorch without CUDA support."
    CUDA_VERSION="cpu"
fi

echo "Updating pip ..."
pip install --upgrade pip --quiet

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing PyTorch 2.2.0"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --quiet
else
    # Install pytorch
    echo "Installing PyTorch 2.2.0+${CUDA_VERSION}"
    # Install PyTorch with CUDA support
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION} --quiet
fi

echo "Installing other required packages ..."
pip install -r requirements.txt --quiet

# Install other required packages and modules
echo "Finalizing installation ..."
pip install -e .
echo "Done !"
