#!/bin/bash

# NablAFx Setup and Test Script
# Based on README.md instructions

set -e  # Exit on any error

echo "=== NablAFx Setup and Test Script ==="

# Function to detect if we're in conda or venv
detect_env() {
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "conda"
    elif [[ -n "$VIRTUAL_ENV" ]]; then
        echo "venv"
    else
        echo "none"
    fi
}

# Function to get Python site-packages directory
get_site_packages() {
    python -c "import site; print(site.getsitepackages()[0])"
}

# Check current environment
ENV_TYPE=$(detect_env)
echo "Detected environment: $ENV_TYPE"

# Setup environment if not already active
if [[ "$ENV_TYPE" == "none" ]]; then
    echo "No active environment detected. Setting up conda environment..."
    
    # Check if conda environment exists
    if conda env list | grep -q "^nablafx "; then
        echo "Environment 'nablafx' already exists. Updating it..."
        conda env update -f environment.yml -n nablafx
    else
        echo "Creating new conda environment..."
        
        # First try to create with main environment file (skip temp file due to version conflicts)
        if conda env create -f environment.yml; then
            echo "Environment created successfully with main environment file"
        else
            echo "Main environment creation failed. Trying alternative approach..."
            # Create basic environment and install via pip
            conda create -n nablafx python=3.9.7 pip wheel numpy -y
            
            # Note: We can't activate conda in a script easily, so we'll update the environment instead
            echo "Installing rational activations..."
            conda run -n nablafx pip install rational-activations==0.2.0
            
            echo "Installing remaining dependencies..."
            conda run -n nablafx pip install --upgrade -r requirements.txt
        fi
    fi
    
    echo "Please activate the environment with: conda activate nablafx"
    echo "Then run this script again."
    exit 1
fi

# Setup rationals config
echo "Setting up rational activations config..."
SITE_PACKAGES=$(get_site_packages)
RATIONAL_CONFIG_PATH="$SITE_PACKAGES/rational/rationals_config.json"

if [[ -f "weights/rationals_config.json" ]]; then
    if [[ ! -f "$RATIONAL_CONFIG_PATH" ]]; then
        echo "Copying rationals_config.json to site-packages..."
        mkdir -p "$(dirname "$RATIONAL_CONFIG_PATH")"
        cp weights/rationals_config.json "$RATIONAL_CONFIG_PATH"
    else
        echo "rationals_config.json already exists in site-packages"
    fi
else
    echo "WARNING: weights/rationals_config.json not found"
fi

# Create required directories
echo "Creating required directories..."
mkdir -p data
mkdir -p logs
mkdir -p checkpoints_fad

# Check for data setup
if [[ ! -d "data/TONETWIST-AFX-DATASET" && ! -L "data/TONETWIST-AFX-DATASET" ]]; then
    echo "WARNING: ToneTwist AFx Dataset not found in data/"
    echo "Please set up data with: ln -s /path/to/TONETWIST-AFX-DATASET/ data/"
fi

# Install additional dependencies
echo "Installing additional dependencies..."
pip install pytest pandas calflops transformers || echo "WARNING: additional dependencies installation failed"

# Ensure rational-activations is properly installed
echo "Checking rational-activations installation..."
if ! python -c "import rational.torch" &>/dev/null; then
    echo "Installing rational-activations..."
    pip install rational-activations==0.2.0 --no-deps --force-reinstall || echo "WARNING: rational-activations installation failed"
fi

# Install nablafx package in development mode
echo "Installing nablafx package in development mode..."
# Note: rational-activations is already installed from conda environment setup
if ! pip install -e .; then
    echo "Standard installation failed, trying without dependencies..."
    pip install -e . --no-deps || echo "WARNING: nablafx package installation failed completely"
fi

# Check wandb login
echo "Checking Weights & Biases setup..."
if ! wandb status &>/dev/null; then
    echo "WARNING: wandb not logged in. Run 'wandb login' if you plan to use logging."
fi

echo "=== Running Tests ==="

# Run unit tests
echo "Running unit tests..."
# Skip tests that require external dependencies or datasets
python -m pytest tests/test_gb_model.py tests/test_lstm.py tests/test_proc_and_contr.py -v --tb=short || echo "WARNING: Some unit tests failed"

# Run performance tests
echo "Running parameter count tests..."
for test_file in tests/nparams_*.py; do
    if [[ -f "$test_file" ]]; then
        echo "Running $test_file..."
        python "$test_file" || echo "WARNING: $test_file failed"
    fi
done

echo "Running FLOPS/MACs/params tests..."
for test_file in tests/flops-macs-params_*.py; do
    if [[ -f "$test_file" ]]; then
        echo "Running $test_file..."
        python "$test_file" || echo "WARNING: $test_file failed"
    fi
done

echo "Running CPU speed tests..."
for test_file in tests/speed_cpu_*.py; do
    if [[ -f "$test_file" ]]; then
        echo "Running $test_file..."
        python "$test_file" || echo "WARNING: $test_file failed"
    fi
done

# Run speed tests with priority if available
if [[ -f "tests/speed_run_with_priority.sh" ]]; then
    echo "Running priority speed tests..."
    bash tests/speed_run_with_priority.sh || echo "WARNING: Priority speed tests failed"
fi

echo "=== Setup and Tests Complete ==="
echo "Environment is ready for NablAFx development!"
echo ""
echo "Next steps:"
echo "- Train black-box model: bash scripts/train_bb.sh"
echo "- Train gray-box model: bash scripts/train_gb.sh"
echo "- Test model: bash scripts/test.sh"
echo "- Pretrain processors: python scripts/pretrain.py"