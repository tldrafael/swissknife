#!/usr/bin/env bash
#
# Bare-metal installer that reproduces the `workbench_cu13_py312` Docker image
# directly on an Ubuntu 24.04 host (no containers involved): CUDA toolkit,
# Node.js + build tooling, a uv-managed Python 3.12 venv with the full
# PyTorch/ML stack (incl. Flash Attention and Detectron2 built from source),
# and an OpenSSH server.
#
# Every version/tunable can be overridden via environment variables, see the
# "Configuration" section below.
#
set -euo pipefail

usage() {
    cat <<EOF
Usage: sudo $0 [options]

Options:
  --prefix PATH       Confine every *relocatable* piece of the environment
                       (venv, uv/pip/HF/torch/matplotlib caches, Jupyter
                       config, build scratch dir, the uv binary + its
                       python/pip symlinks, the launcher, and a generated
                       env.sh with all resulting env vars) under PATH instead
                       of system-wide locations. --venv/--build-tmpdir still
                       win over --prefix-derived defaults. apt-installed
                       packages (CUDA toolkit, build-essential, nodejs,
                       openssh-server, ...) are NOT affected: apt/dpkg always
                       installs into normal system paths on bare metal (use
                       the workbench_cu13_py312 Dockerfile for full isolation).
  --venv PATH         Python virtualenv path
                       (default: /opt/venv, or <prefix>/venv with --prefix)
  --skip-cuda         Never apt-get install a CUDA toolkit. A compatible
                       existing toolkit is always detected/reused regardless
                       of this flag; pass it to require that reuse and fail
                       instead of falling back to an apt install.
  --skip-flash-attn   Skip building Flash Attention (by far the slowest step,
                       and needs a matching nvcc)
  --skip-ssh          Skip installing/configuring OpenSSH server
  --allow-root-ssh    Enable root login + password auth in sshd_config
                       (matches the container's defaults; not recommended on
                       a real host unless you know why you need it)
  --build-tmpdir PATH Scratch dir for the Flash Attention/Detectron2 source
                       builds (default: \$TMPDIR or /tmp, or <prefix>/tmp).
                       Flash Attention alone can need 25+GB; point this at a
                       roomier disk (or tmpfs backed by enough RAM) if /tmp
                       is small.
  -h, --help          Show this help and exit
EOF
}

# ----------------------------------------------------------------------------
# Configuration (mirrors the ARGs in the workbench_cu13_py312 Dockerfile)
# ----------------------------------------------------------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
NODE_MAJOR="${NODE_MAJOR:-24}"

CUDA_TOOLKIT_APT_VERSION="${CUDA_TOOLKIT_APT_VERSION:-13-0}"   # apt package: cuda-toolkit-13-0
PYTORCH_CUDA="${PYTORCH_CUDA:-cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"

DETECTRON2_REF="${DETECTRON2_REF:-main}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
# Restrict which GPU architectures Flash Attention compiles for (e.g.
# "8.0;9.0;10.0"). Leave unset for its own default detection. Useful if the
# build fails with "Unsupported gpu architecture": newer CUDA majors
# periodically drop support for compute capabilities a pinned flash-attn
# release's setup.py may still hardcode.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"

PREFIX="${PREFIX:-}"
VIRTUAL_ENV="${VIRTUAL_ENV:-}"
MAX_JOBS_EXPLICIT="${MAX_JOBS:-}"
BUILD_TMPDIR="${BUILD_TMPDIR:-}"

SKIP_CUDA_INSTALL=0
SKIP_FLASH_ATTN=0
SKIP_SSH_INSTALL=0
ALLOW_ROOT_SSH=0

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die() { printf '\n\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --venv) VIRTUAL_ENV="$2"; shift 2 ;;
        --skip-cuda) SKIP_CUDA_INSTALL=1; shift ;;
        --skip-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
        --skip-ssh) SKIP_SSH_INSTALL=1; shift ;;
        --allow-root-ssh) ALLOW_ROOT_SSH=1; shift ;;
        --build-tmpdir) BUILD_TMPDIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

[[ $EUID -eq 0 ]] || die "This script must be run as root (try: sudo $0)."

ARCH="$(dpkg --print-architecture)"
[[ "$ARCH" == "amd64" ]] || die "Only amd64 is supported (Flash Attention is only built/tested on amd64), found: $ARCH"

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
        warn "This script targets Ubuntu 24.04 (found: ${PRETTY_NAME:-unknown}). Continuing anyway, but things may break."
    fi
else
    warn "Could not detect OS release info; assuming Ubuntu 24.04 compatible."
fi

export DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# 0. --prefix resolution
# ----------------------------------------------------------------------------
# Everything relocatable (venv, uv/pip/HF/torch/matplotlib caches, Jupyter
# dirs, build scratch space, the uv binary + its python/pip symlinks, the
# launcher, and a generated env.sh) is confined under $PREFIX instead of
# system-wide locations. This does NOT (and, short of a chroot/container,
# cannot) affect apt/dpkg, which always installs into /usr, /etc,
# /var/lib/dpkg regardless of --prefix.
BIN_DIR=/usr/local/bin
ENV_FILE=/etc/profile.d/cuda.sh
if [[ -n "$PREFIX" ]]; then
    mkdir -p "$PREFIX"
    PREFIX="$(cd "$PREFIX" && pwd)"

    : "${VIRTUAL_ENV:=${PREFIX}/venv}"
    : "${BUILD_TMPDIR:=${PREFIX}/tmp}"
    BIN_DIR="${PREFIX}/bin"
    ENV_FILE="${PREFIX}/env.sh"

    CACHE_DIR="${PREFIX}/cache"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_DIR}/uv}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_DIR}/pip}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}}"
    export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
    export TORCH_HOME="${TORCH_HOME:-${CACHE_DIR}/torch}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_DIR}/matplotlib}"
    export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-${PREFIX}/jupyter/config}"
    export JUPYTER_DATA_DIR="${JUPYTER_DATA_DIR:-${PREFIX}/jupyter/data}"
    export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-${PREFIX}/jupyter/runtime}"

    mkdir -p "$BIN_DIR" "$BUILD_TMPDIR" \
        "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
        "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
    export PATH="${BIN_DIR}:${PATH}"

    # Fresh on every run; pin_cuda_env() appends CUDA_HOME/PATH/LD_LIBRARY_PATH
    # to this same file below instead of writing anything under /etc.
    cat > "$ENV_FILE" <<EOF
export PATH="${BIN_DIR}:\${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR}"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR}"
export JUPYTER_DATA_DIR="${JUPYTER_DATA_DIR}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR}"
EOF

    log "Using prefix ${PREFIX}: venv=${VIRTUAL_ENV}, build-tmpdir=${BUILD_TMPDIR}," \
        "caches/bin/jupyter under ${PREFIX}. Source ${ENV_FILE} in new shells to pick this up" \
        "(apt-installed system packages are NOT affected, see --help)."
fi
: "${VIRTUAL_ENV:=/opt/venv}"

# Needed this early because step 1 (CUDA keyring download) uses it too, not
# just the Flash Attention/Detectron2 builds in step 8.
EFFECTIVE_TMPDIR="${BUILD_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$EFFECTIVE_TMPDIR"
export TMPDIR="$EFFECTIVE_TMPDIR"

log "Installing base prerequisites..."
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

# ----------------------------------------------------------------------------
# 1. CUDA toolkit (replaces the nvidia/cuda:13.0.0-devel base image), unless
#    a PyTorch-compatible one is already present on the host
# ----------------------------------------------------------------------------
# Desired major.minor, e.g. "13-0" -> "13.0". Used as the fallback install
# target if no existing, PyTorch-compatible toolkit is found on the host.
DESIRED_CUDA_VERSION="${CUDA_TOOLKIT_APT_VERSION//-/.}"

# CUDA major.minor -> "wheel_tag:torch_version:torchvision_version:torchaudio_version"
# for the newest PyTorch release that ships a prebuilt Linux x86_64 wheel for
# that CUDA version. This lets the script adapt to whatever CUDA toolkit is
# already on the host instead of being tied to a single pinned
# torch==${TORCH_VERSION}: versions still covered by that pin (12.6/12.8/13.0)
# just reuse it, older ones fall back to the last torch release supporting
# them. See https://pytorch.org/get-started/previous-versions/ - update this
# map to support additional/newer CUDA versions.
declare -A CUDA_PYTORCH_MATRIX=(
    ["11.8"]="cu118:2.7.1:0.22.1:2.7.1"
    ["12.1"]="cu121:2.5.1:0.20.1:2.5.1"
    ["12.4"]="cu124:2.6.0:0.21.0:2.6.0"
    ["12.6"]="cu126:${TORCH_VERSION}:${TORCHVISION_VERSION}:${TORCHAUDIO_VERSION}"
    ["12.8"]="cu128:${TORCH_VERSION}:${TORCHVISION_VERSION}:${TORCHAUDIO_VERSION}"
    ["12.9"]="cu129:2.8.0:0.23.0:2.8.0"
    ["13.0"]="cu130:${TORCH_VERSION}:${TORCHVISION_VERSION}:${TORCHAUDIO_VERSION}"
)

# Newest CUDA toolkit's nvcc visible on this host: via PATH first, else by
# scanning /usr/local/cuda-* directly (covers the case where a toolkit is
# installed but /etc/profile.d/cuda.sh hasn't been sourced yet).
detected_nvcc_bin() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return
    fi
    local candidate
    candidate="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    echo "$candidate"
}

detected_nvcc_version() {
    local nvcc_bin
    nvcc_bin="$(detected_nvcc_bin)" || return 1
    "$nvcc_bin" --version | grep -oP 'release \K[0-9]+\.[0-9]+' || return 1
}

detected_nvcc_home() {
    local nvcc_bin
    nvcc_bin="$(detected_nvcc_bin)" || return 1
    dirname "$(dirname "$nvcc_bin")"
}

# Pins CUDA_HOME/PATH/LD_LIBRARY_PATH to an explicit, versioned install dir
# (never the ambiguous /usr/local/cuda symlink) so a pre-existing, different
# CUDA install on the host can never shadow the version we actually need.
# Called at most once per run: overwrites ENV_FILE on a plain run (fresh
# /etc/profile.d/cuda.sh), or appends to it with --prefix (where step 0
# already wrote the cache/PATH exports there).
pin_cuda_env() {
    local cuda_home="$1"
    [[ -d "$cuda_home" ]] || die "Expected CUDA install dir not found: $cuda_home"

    local redirect=">"
    [[ -n "$PREFIX" ]] && redirect=">>"
    eval "cat $redirect \"\$ENV_FILE\"" <<EOF
export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:\${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/lib64:\${LD_LIBRARY_PATH:-}"
EOF

    export CUDA_HOME="$cuda_home"
    export PATH="${cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"
}

install_cuda_toolkit() {
    local found_version found_home entry
    found_version="$(detected_nvcc_version || true)"

    if [[ -n "$found_version" ]]; then
        found_home="$(detected_nvcc_home || true)"
        entry="${CUDA_PYTORCH_MATRIX[$found_version]:-}"

        if [[ -n "$entry" ]]; then
            local tag torch_ver tv_ver ta_ver
            IFS=':' read -r tag torch_ver tv_ver ta_ver <<< "$entry"
            log "Found existing CUDA ${found_version} toolkit at ${found_home:-unknown}; reusing it" \
                "instead of installing CUDA ${DESIRED_CUDA_VERSION} (torch==${torch_ver}, PYTORCH_CUDA=${tag})."
            DESIRED_CUDA_VERSION="$found_version"
            PYTORCH_CUDA="$tag"
            TORCH_VERSION="$torch_ver"
            TORCHVISION_VERSION="$tv_ver"
            TORCHAUDIO_VERSION="$ta_ver"
            pin_cuda_env "${found_home:-/usr/local/cuda-${found_version}}"
            return
        fi

        warn "Found nvcc ${found_version} on this host, but no known PyTorch release ships a" \
             "prebuilt wheel for it (supported: ${!CUDA_PYTORCH_MATRIX[*]})."
    fi

    if [[ $SKIP_CUDA_INSTALL -eq 1 ]]; then
        die "No usable existing CUDA toolkit found (need one of: ${!CUDA_PYTORCH_MATRIX[*]}), but" \
            "--skip-cuda was passed so this script won't install one. Install a supported CUDA" \
            "version yourself and make sure nvcc is on PATH (or under /usr/local/cuda-<ver>/bin)," \
            "or drop --skip-cuda to let this script install CUDA ${DESIRED_CUDA_VERSION} for you."
    fi

    log "Installing CUDA ${DESIRED_CUDA_VERSION} toolkit from NVIDIA's apt repo..."
    local keyring_deb="${EFFECTIVE_TMPDIR}/cuda-keyring.deb"
    curl -fsSL -o "$keyring_deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i "$keyring_deb"
    rm -f "$keyring_deb"

    apt-get update
    apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_TOOLKIT_APT_VERSION}"

    pin_cuda_env "/usr/local/cuda-${DESIRED_CUDA_VERSION}"
}

install_cuda_toolkit

if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found. This installs build tooling only; you still need a working" \
         "NVIDIA driver on this host to actually run GPU workloads."
fi

# ----------------------------------------------------------------------------
# 2. System packages and Node.js
# ----------------------------------------------------------------------------
log "Installing system packages and Node.js ${NODE_MAJOR}.x..."
curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -

apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    htop \
    libaio-dev \
    libgl1 \
    libglib2.0-0 \
    libglib2.0-dev \
    libhdf5-dev \
    libopenblas-dev \
    ninja-build \
    nodejs \
    pkg-config \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv" \
    rsync \
    tmux \
    vim

# ----------------------------------------------------------------------------
# 3. uv, and a Python virtual environment managed by it
# ----------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    log "uv already installed ($(uv --version)); skipping installer."
else
    log "Installing uv into ${BIN_DIR}..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${BIN_DIR}" sh
fi

log "Creating Python virtualenv at ${VIRTUAL_ENV} (via uv)..."
uv venv --python "python${PYTHON_VERSION}" "${VIRTUAL_ENV}"
ln -sf "${VIRTUAL_ENV}/bin/python" "${BIN_DIR}/python"
ln -sf "${VIRTUAL_ENV}/bin/pip" "${BIN_DIR}/pip"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
# The --no-build-isolation source builds below (Flash Attention, Detectron2)
# run against this venv's packages directly, so its build backends need to
# be present up front.
uv pip install --upgrade pip setuptools wheel packaging

# ----------------------------------------------------------------------------
# 4. PyTorch, aligned with the CUDA toolkit selected in step 1
# ----------------------------------------------------------------------------
log "Installing PyTorch ${TORCH_VERSION} (${PYTORCH_CUDA})..."
uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"

# ----------------------------------------------------------------------------
# 5. Data and science stack
# ----------------------------------------------------------------------------
log "Installing data/science stack..."
uv pip install \
    duckdb \
    "imageio[pyav]" \
    imagesize \
    pylance \
    matplotlib \
    numpy \
    opencv-python-headless \
    pandas \
    pillow \
    pyarrow \
    scikit-image \
    scikit-learn \
    scipy \
    seaborn

# ----------------------------------------------------------------------------
# 6. ML and DL ecosystem
# ----------------------------------------------------------------------------
log "Installing ML/DL ecosystem..."
uv pip install \
    accelerate \
    bitsandbytes \
    deepspeed \
    einops \
    fairscale \
    liger-kernel \
    onnx \
    onnxruntime \
    sentencepiece \
    tensorboard \
    timm \
    torchmetrics \
    transformers \
    "tritonclient[http]"

# ----------------------------------------------------------------------------
# 7. Dev, profiling, and Detectron2 runtime dependencies
# ----------------------------------------------------------------------------
log "Installing dev/profiling/Detectron2 runtime dependencies..."
uv pip install \
    cloudpickle \
    easydict \
    fvcore \
    hydra-core \
    iopath \
    jupyterlab \
    kornia \
    line_profiler \
    omegaconf \
    pycocotools \
    tabulate \
    termcolor \
    tqdm \
    yacs

# ----------------------------------------------------------------------------
# 8. Flash Attention, built from source (no prebuilt wheel matches every
#    torch/CUDA combination this script may end up selecting)
# ----------------------------------------------------------------------------
if [[ $SKIP_FLASH_ATTN -eq 1 ]]; then
    log "Skipping Flash Attention build (--skip-flash-attn)."
else
    # Fail fast: the source build must compile against the *same* CUDA
    # toolkit version the installed torch wheel was built for, or it fails
    # with a compile error (or, on OOM-killed workers, almost no diagnostics
    # at all).
    preflight_nvcc="$(detected_nvcc_version || true)"
    [[ "$preflight_nvcc" == "$DESIRED_CUDA_VERSION" ]] || die \
        "nvcc reports '${preflight_nvcc:-not found}', but Flash Attention must be built against" \
        "CUDA ${DESIRED_CUDA_VERSION} to match torch==${TORCH_VERSION}+${PYTORCH_CUDA}." \
        "Re-run without --skip-cuda, or fix CUDA_HOME/PATH to point at ${DESIRED_CUDA_VERSION} yourself, " \
        "or pass --skip-flash-attn to skip this step entirely."

    # Flash Attention compiles ~70 files against multiple GPU architectures
    # each, and nvcc/gcc drop large per-file scratch artifacts into $TMPDIR
    # along the way - easily enough to exhaust /tmp. EFFECTIVE_TMPDIR/TMPDIR
    # were already resolved (and, with --prefix, confined under it) above.

    # Best-effort cleanup of scratch dirs orphaned by earlier failed builds;
    # these can silently eat many GB across repeated re-runs.
    find "$EFFECTIVE_TMPDIR" -mindepth 1 -maxdepth 1 \( -name 'pip-install-*' -o -name 'pip-req-build-*' \) \
        -exec rm -rf {} + 2>/dev/null || true

    MIN_TMP_GB=25
    avail_tmp_gb=$(( $(df --output=avail -B1 "$EFFECTIVE_TMPDIR" | tail -1) / 1024 / 1024 / 1024 ))
    if (( avail_tmp_gb < MIN_TMP_GB )); then
        die "Only ${avail_tmp_gb}GiB free on ${EFFECTIVE_TMPDIR}, but the Flash Attention source build" \
            "needs on the order of ${MIN_TMP_GB}+GiB of scratch space (it compiles ~70 files across" \
            "multiple GPU architectures in parallel). Free up space, or point elsewhere with" \
            "--build-tmpdir /path/on/a/bigger/disk (or the BUILD_TMPDIR env var)."
    fi
    log "Using ${EFFECTIVE_TMPDIR} for build scratch space (${avail_tmp_gb}GiB free)."

    # Each parallel compile job can use several GB of RAM and scratch disk;
    # too high a MAX_JOBS commonly OOM-kills a compiler worker (bare
    # "finished with status 'error'", no diagnostics) or fills $TMPDIR
    # outright. Auto-cap the job count from available RAM/CPU/disk unless the
    # caller explicitly set MAX_JOBS.
    if [[ -n "$MAX_JOBS_EXPLICIT" ]]; then
        MAX_JOBS="$MAX_JOBS_EXPLICIT"
    else
        mem_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
        cpu_n=$(nproc)
        disk_jobs=$(( avail_tmp_gb / 2 ))
        MAX_JOBS=$(( mem_gb / 4 ))
        (( MAX_JOBS < 1 )) && MAX_JOBS=1
        (( MAX_JOBS > cpu_n )) && MAX_JOBS=$cpu_n
        (( disk_jobs < 1 )) && disk_jobs=1
        (( MAX_JOBS > disk_jobs )) && MAX_JOBS=$disk_jobs
        (( MAX_JOBS > 32 )) && MAX_JOBS=32
        log "Auto-selected MAX_JOBS=${MAX_JOBS} (available RAM: ${mem_gb}GiB, CPUs: ${cpu_n}," \
            "scratch disk: ${avail_tmp_gb}GiB). Override with MAX_JOBS=<n> if you want a different value."
    fi

    # uv/pip hide the real compiler/linker error behind a live-updating
    # status line, and only reformat it on failure - unreliably so when piped
    # to a log file. Force plain, immediately-streamed, uncolored output so
    # the real failure is never lost.
    export NO_COLOR=1
    export UV_NO_PROGRESS=1

    log "Building Flash Attention ${FLASH_ATTN_VERSION} from source (MAX_JOBS=${MAX_JOBS})..."
    if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
        log "Restricting build to TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        export TORCH_CUDA_ARCH_LIST
    fi
    MAX_JOBS="${MAX_JOBS}" uv pip install --no-build-isolation --verbose \
        "flash-attn==${FLASH_ATTN_VERSION}"
fi

# ----------------------------------------------------------------------------
# 9. Detectron2 from source
# ----------------------------------------------------------------------------
log "Installing Detectron2 (ref: ${DETECTRON2_REF}) from source..."
uv pip install --no-build-isolation --no-deps --verbose \
    "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_REF}"

# ----------------------------------------------------------------------------
# 10. Lightweight import checks
# ----------------------------------------------------------------------------
log "Running import checks..."
SKIP_FLASH_ATTN="$SKIP_FLASH_ATTN" python - <<'PY'
import os

import torch
import torchvision
import torchaudio
import detectron2

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)

if os.environ.get("SKIP_FLASH_ATTN") == "1":
    print("flash_attn: skipped (--skip-flash-attn)")
else:
    import flash_attn
    print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))

print("detectron2 import: ok")
PY

# ----------------------------------------------------------------------------
# 11. SSH server and rsync
# ----------------------------------------------------------------------------
if [[ $SKIP_SSH_INSTALL -eq 1 ]]; then
    log "Skipping OpenSSH server install (--skip-ssh)."
else
    log "Installing OpenSSH server and rsync..."
    apt-get install -y --no-install-recommends openssh-server rsync
    mkdir -p /var/run/sshd

    if [[ $ALLOW_ROOT_SSH -eq 1 ]]; then
        warn "Enabling root login + password authentication in sshd_config (--allow-root-ssh)." \
             "This matches the Dockerfile's container defaults but is a real security" \
             "consideration on a bare-metal host. Make sure this is intentional."
        sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
        sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
        systemctl restart ssh 2>/dev/null || service ssh restart || true
    else
        log "Leaving sshd_config untouched (use --allow-root-ssh to mirror the container's" \
            "PermitRootLogin/PasswordAuthentication defaults)."
    fi

    systemctl enable --now ssh 2>/dev/null || service ssh start || true
fi

# ----------------------------------------------------------------------------
# 12. Convenience launcher (mirrors the Dockerfile's CMD)
# ----------------------------------------------------------------------------
LAUNCHER="${BIN_DIR}/start-workbench"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Starts JupyterLab the same way the workbench_cu13_py312 container's CMD does.
# Set PUBLIC_KEY to have it appended to /root/.ssh/authorized_keys on start.
set -euo pipefail
if [[ -n "\${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    echo "\${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
service ssh start 2>/dev/null || systemctl start ssh 2>/dev/null || true
exec "${VIRTUAL_ENV}/bin/jupyter-lab" --no-browser --ip 0.0.0.0 --allow-root --port 8884
EOF
chmod +x "$LAUNCHER"

log "Done."
if [[ $SKIP_FLASH_ATTN -eq 1 ]]; then
    FLASH_ATTN_SUMMARY="skipped (--skip-flash-attn)"
else
    FLASH_ATTN_SUMMARY="${FLASH_ATTN_VERSION}"
fi
cat <<EOF

Installed environment summary:
  - Python venv:   ${VIRTUAL_ENV}  (uv-managed; also symlinked as ${BIN_DIR}/python, ${BIN_DIR}/pip)
  - CUDA/PyTorch:  CUDA ${DESIRED_CUDA_VERSION}, torch ${TORCH_VERSION} (${PYTORCH_CUDA})
  - Flash Attn:    ${FLASH_ATTN_SUMMARY}
  - Detectron2:    ${DETECTRON2_REF}
  - Build scratch: ${EFFECTIVE_TMPDIR}
  - Launcher:      ${LAUNCHER}   (starts sshd + jupyter-lab on port 8884)

Open a new shell (or 'source ${ENV_FILE}') to pick up the PATH/CUDA/cache env changes.
EOF
if [[ -n "$PREFIX" ]]; then
    cat <<EOF

--prefix ${PREFIX} was used: the venv, uv/pip/HF/torch/matplotlib caches,
Jupyter dirs, build scratch space, uv/python/pip binaries, and the launcher
all live under it, and ${ENV_FILE} exports everything needed to use them
(add 'source ${ENV_FILE}' to your shell rc to make it permanent).

NOT covered by --prefix (these always land in normal host system paths,
apt/dpkg cannot be redirected without a chroot/container):
  - The CUDA toolkit's own files (/usr/local/cuda-${DESIRED_CUDA_VERSION})
  - apt-installed packages (build-essential, nodejs, openssh-server, etc.)
  - sshd's config/host keys (/etc/ssh, /var/run/sshd) and authorized_keys
    written by the launcher (/root/.ssh)
EOF
fi
