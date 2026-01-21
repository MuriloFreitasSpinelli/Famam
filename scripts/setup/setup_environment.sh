#!/bin/bash
# ============================================
# Famam Complete Environment Setup (Linux/Mac)
# ============================================
# Sets up conda environment with TensorFlow GPU support (CUDA 11.2)
# Run this ONCE to set up everything, then use run_training.sh
#
# Requirements:
#   - Miniconda or Anaconda installed
#   - NVIDIA GPU with drivers installed (optional, for GPU training)
#
# Usage: chmod +x setup_environment.sh && ./setup_environment.sh

set -e

ENV_NAME="famam_gpu"
PYTHON_VERSION="3.10"

echo ""
echo "============================================"
echo "  Famam Environment Setup"
echo "  Platform: $(uname -s)"
echo "============================================"
echo ""

# ============================================
# Step 1: Find Conda
# ============================================
echo "[1/8] Locating conda installation..."

CONDA_EXE=""
for path in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda" "/usr/local/conda/bin/conda" "$HOME/miniforge3/bin/conda"; do
    if [ -f "$path" ]; then
        CONDA_EXE="$path"
        break
    fi
done

if [ -z "$CONDA_EXE" ] && command -v conda &> /dev/null; then
    CONDA_EXE=$(which conda)
fi

if [ -z "$CONDA_EXE" ]; then
    echo ""
    echo "ERROR: Conda not found!"
    echo ""
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname -s)-$(uname -m).sh"
    echo "  bash Miniconda3-latest-*.sh"
    echo "  source ~/.bashrc  # or ~/.zshrc"
    echo "  # Then run this script again"
    exit 1
fi

echo "       Found: $CONDA_EXE"

# Initialize conda for this shell
eval "$($CONDA_EXE shell.bash hook)"

# ============================================
# Step 2: Check for existing environment
# ============================================
echo ""
echo "[2/8] Checking for existing environment..."

if $CONDA_EXE env list | grep -q "^$ENV_NAME "; then
    echo "       Environment '$ENV_NAME' already exists."
    read -p "       Recreate it? (y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "       Removing existing environment..."
        $CONDA_EXE env remove -n $ENV_NAME -y > /dev/null 2>&1 || true
    else
        echo "       Keeping existing environment. Updating packages..."
        SKIP_CREATE=true
    fi
fi

# ============================================
# Step 3: Create conda environment
# ============================================
if [ "$SKIP_CREATE" != "true" ]; then
    echo ""
    echo "[3/8] Creating conda environment with Python $PYTHON_VERSION..."
    $CONDA_EXE create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# ============================================
# Step 4: Install CUDA toolkit
# ============================================
echo ""
echo "[4/8] Installing CUDA Toolkit 11.2 and cuDNN 8.1..."
echo "       (This downloads ~1.2GB, may take a while)"
$CONDA_EXE install -n $ENV_NAME -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y || echo "WARNING: CUDA installation had issues"

# ============================================
# Step 5: Install TensorFlow
# ============================================
echo ""
echo "[5/8] Installing TensorFlow 2.10.1..."
$CONDA_EXE run -n $ENV_NAME pip install tensorflow==2.10.1 "numpy<2"

# ============================================
# Step 6: Install core dependencies
# ============================================
echo ""
echo "[6/8] Installing core project dependencies..."
$CONDA_EXE run -n $ENV_NAME pip install \
    h5py \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    pretty_midi \
    music21 \
    muspy \
    mido \
    requests

# ============================================
# Step 7: Install additional dependencies
# ============================================
echo ""
echo "[7/8] Installing additional dependencies..."
$CONDA_EXE run -n $ENV_NAME pip install \
    jupyterlab \
    datasets \
    scikit-optimize \
    midi2audio || echo "WARNING: Some optional deps failed (midi2audio needs FluidSynth)"

# ============================================
# Step 8: Setup activation scripts and verify
# ============================================
echo ""
echo "[8/8] Configuring environment and verifying GPU..."

# Get conda env path
ENV_PATH=$($CONDA_EXE run -n $ENV_NAME python -c "import sys; print(sys.prefix)")

# Create activation script for CUDA library path
mkdir -p "$ENV_PATH/etc/conda/activate.d"
cat > "$ENV_PATH/etc/conda/activate.d/cuda_path.sh" << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
EOF
chmod +x "$ENV_PATH/etc/conda/activate.d/cuda_path.sh"

# Verify installation
echo ""
echo "Verifying installation..."
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
$CONDA_EXE run -n $ENV_NAME python -c "
import tensorflow as tf
print('  TensorFlow:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('  GPUs found:', len(gpus))
for g in gpus:
    print('    -', g.name)
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Environment: $ENV_NAME"
echo ""
echo "Quick commands:"
echo "  conda activate $ENV_NAME     - Activate the environment"
echo "  ./run_training.sh            - Run model training"
echo "  jupyter lab                  - Start JupyterLab"
echo ""
echo "Installed packages:"
echo "  - TensorFlow 2.10.1 (GPU support via CUDA 11.2)"
echo "  - NumPy, Pandas, Matplotlib, Scikit-learn"
echo "  - Music21, MusPy, Pretty-MIDI, Mido"
echo "  - JupyterLab, Scikit-optimize, HuggingFace Datasets"
echo ""
