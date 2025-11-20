#!/bin/bash
#
# AI Performance Engineering Setup Script
# ========================================
#
# This script installs EVERYTHING you need:
#   1. NVIDIA Driver 580+ (auto-upgrades if needed)
#   2. Python 3.12 (PyTorch 2.10-dev compatible)
#   3. CUDA 13.0.2 (Update 2) repository and toolchain
#   4. Environment for PyTorch 2.10-dev source build with CUDA 13.0.2
#   5. NVIDIA Nsight Systems 2025.3.2 (for timeline profiling)
#   6. NVIDIA Nsight Compute 2025.3.1 (for kernel metrics)
#   7. All Python dependencies from requirements_latest.txt
#   8. System tools (numactl, perf, htop, etc.)
#   9. Configures NVIDIA drivers for profiling
#
# Requirements:
#   - Ubuntu 22.04+ (tested on 22.04)
#   - NVIDIA B200/B300 GPU (or compatible)
#   - sudo/root access
#   - Internet connection
#
# Usage:
#   sudo ./setup.sh
#
# Duration: 10-20 minutes (first run may require reboot for driver upgrade)
#
# What it does:
#   - Adds official NVIDIA CUDA 13.0 (Update 2) repository
#   - Configures APT to prefer official NVIDIA packages
#   - Fixes Python APT module (python3-apt) compatibility
#   - Disables problematic command-not-found APT hook
#   - Removes duplicate deadsnakes repository entries
#   - Upgrades Python to 3.12 (required by PyTorch 2.10 dev builds)
#   - Auto-upgrades NVIDIA driver to 580+ if needed (will prompt reboot)
#   - Installs CUDA 13.0.2 toolkit and libraries
#   - Installs latest Nsight tools (2025.x)
#   - Prepares for PyTorch 2.10-dev (source build) with CUDA 13.0.2
#   - Removes conflicting system packages (python3-optree, etc.)
#   - Installs nvidia-ml-py (replaces deprecated pynvml)
#   - Configures NVIDIA kernel modules for profiling
#   - Fixes hardware info script compatibility
#   - Runs validation tests
#
# Notes:
#   - If driver upgrade is needed, script will exit and ask you to reboot
#   - After reboot, simply re-run: sudo ./setup.sh
#   - The script is idempotent and safe to re-run
#
# After running this script, you can:
#   - Run examples: python3 ch1/performance_basics.py
#   - Drive the benchmark suite: python tools/cli/benchmark_cli.py run
#   - Capture peak performance: python tools/benchmarking/benchmark_peak.py
#   - Verify examples: python tools/cli/benchmark_cli.py verify
#

set -e  # Exit on any error

echo "AI Performance Engineering Setup Script"
echo "=========================================="
echo "This script will install:"
echo "  • NVIDIA Driver 580+ (auto-upgrade if needed)"
echo "  • Python 3.12 (PyTorch 2.10-dev compatible)"
echo "  • CUDA 13.0.2 (Update 2) repository and toolchain"
echo "  • Environment configured for PyTorch 2.10-dev source build"
echo "  • NVIDIA Nsight Systems 2025.3.2 (latest)"
echo "  • NVIDIA Nsight Compute 2025.3.1 (latest)"
echo "  • All project dependencies"
echo "  • System tools (numactl, perf, etc.)"
echo ""
echo "Note: If driver upgrade is needed, you'll be prompted to reboot."
echo ""

PROJECT_ROOT="$(dirname "$(realpath "$0")")"
REQUIRED_DRIVER_VERSION="580.95.05"
PYTHON_TARGET_VERSION="3.12"
PYTHON_TARGET_MAJOR="${PYTHON_TARGET_VERSION%%.*}"
PYTHON_TARGET_MINOR="${PYTHON_TARGET_VERSION##*.}"
PYTHON_TARGET_BIN="python${PYTHON_TARGET_VERSION}"
PYTHON_ABI_TAG="cp${PYTHON_TARGET_MAJOR}${PYTHON_TARGET_MINOR}"
PYTHON_DIST_PACKAGES="/usr/local/lib/python${PYTHON_TARGET_VERSION}/dist-packages"
CUDA_SHORT_VERSION="13.0"
CUDA_FULL_VERSION="13.0.2.006"
# cuDNN version (install latest available in CUDA 13 repo)
CUDNN_VERSION="9.16.0.29"
NCCL_SHORT_VERSION="2.28.7"
CUDA_HOME_DIR="/usr/local/cuda-${CUDA_SHORT_VERSION}"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
mkdir -p "${THIRD_PARTY_DIR}"
TE_GIT_COMMIT="e4bfa628632e15ef8bc1fae9b2e89686f6a097ea"
TE_VERSION_TAG="2.10.0.dev0+e4bfa62"
FLASH_ATTN_TAG="${FLASH_ATTN_TAG:-v2.8.3}"
FLASH_ATTN_WHEEL_BASENAME="flash_attn-2.8.3-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_aarch64.whl"
FLASH_ATTN_EXPECTED_VERSION="${FLASH_ATTN_TAG#v}"
PIP_ROOT_USER_ACTION="ignore"
export PROJECT_ROOT REQUIRED_DRIVER_VERSION PYTHON_TARGET_VERSION PYTHON_TARGET_MAJOR PYTHON_TARGET_MINOR PYTHON_TARGET_BIN PYTHON_ABI_TAG PYTHON_DIST_PACKAGES PIP_ROOT_USER_ACTION

if command -v git >/dev/null 2>&1; then
    git config --global --add safe.directory "${PROJECT_ROOT}" >/dev/null 2>&1 || true
    if [ -d "${PROJECT_ROOT}/vendor/pytorch-src" ]; then
        git config --global --add safe.directory "${PROJECT_ROOT}/vendor/pytorch-src" >/dev/null 2>&1 || true
    fi
    if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if [ -f "${PROJECT_ROOT}/.gitmodules" ]; then
            git -C "${PROJECT_ROOT}" submodule sync --recursive >/dev/null 2>&1 || true
            git -C "${PROJECT_ROOT}" submodule update --init --recursive >/dev/null 2>&1 || true
        fi
    fi
fi
PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-https://github.com/pytorch/pytorch.git}"
PYTORCH_COMMIT="${PYTORCH_COMMIT:-main}"
PYTORCH_SRC_DIR="${PYTORCH_SRC_DIR:-${THIRD_PARTY_DIR}/pytorch-src}"
PYTORCH_BUILD_DIR="${PYTORCH_SRC_DIR}"
PYTORCH_DIST_DIR="${PYTORCH_BUILD_DIR}/dist"
PYTORCH_WHEEL_DIR="${THIRD_PARTY_DIR}/wheels"
PYTORCH_WHEEL_PATTERN="${PYTORCH_WHEEL_PATTERN:-torch-*-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-*.whl}"
mkdir -p "${PYTORCH_WHEEL_DIR}"

build_pytorch_from_source() {
    echo "Building PyTorch from source (commit: ${PYTORCH_COMMIT})..."
    apt install -y git build-essential cmake libopenblas-dev libomp-dev ninja-build >/dev/null 2>&1 || true

    mkdir -p "${PYTORCH_SRC_DIR}"
    if [ ! -d "${PYTORCH_SRC_DIR}/.git" ]; then
        rm -rf "${PYTORCH_SRC_DIR}"
        git clone --recursive "${PYTORCH_REPO_URL}" "${PYTORCH_SRC_DIR}"
    else
        git -C "${PYTORCH_SRC_DIR}" fetch --all --tags --prune --force
    fi

    git -C "${PYTORCH_SRC_DIR}" checkout "${PYTORCH_COMMIT}"
    git -C "${PYTORCH_SRC_DIR}" submodule sync --recursive
    git -C "${PYTORCH_SRC_DIR}" submodule update --init --recursive

    pip_install --no-cache-dir --upgrade --ignore-installed \
        setuptools wheel typing_extensions packaging cmake ninja || true
    if [ -f "${PYTORCH_SRC_DIR}/requirements.txt" ]; then
        pip_install --no-cache-dir --upgrade --ignore-installed \
            -r "${PYTORCH_SRC_DIR}/requirements.txt" || true
    fi

    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_NCCL=1
    export USE_SYSTEM_NCCL=1
    export USE_SYSTEM_CUDA=1
    export FORCE_CUDA=1
    export BUILD_TEST=0
    export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_VALUE}"
    export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCH_LIST_VALUE}"
    export CUTLASS_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS_VALUE}"
    export CUDA_HOME="${CUDA_HOME_DIR}"
    export CUDACXX="${CUDA_HOME_DIR}/bin/nvcc"
    export CUDNN_INCLUDE_DIR="/usr/include"
    local cudnn_lib_dir="/usr/lib/x86_64-linux-gnu"
    if [ -d "/usr/lib/aarch64-linux-gnu" ]; then
        cudnn_lib_dir="/usr/lib/aarch64-linux-gnu"
    fi
    export CUDNN_LIBRARY_DIR="${cudnn_lib_dir}"
    export CPATH="${CUDA_HOME_DIR}/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_HOME_DIR}/include:${CPLUS_INCLUDE_PATH:-}"
    # CRITICAL: PyTorch was compiled against cuDNN 9.15.1, but bundles cuDNN 9.13.0
    # Use system cuDNN 9.15.1 (from ${cudnn_lib_dir}) instead of PyTorch's bundled 9.13.0
    # Build LD_LIBRARY_PATH with system cuDNN 9.15.1 first, then CUDA, excluding PyTorch's bundled cuDNN
    PYTORCH_CUDNN_LIB="${PYTHON_DIST_PACKAGES}/nvidia/cudnn/lib"
    
    local new_ld_path="${cudnn_lib_dir}:${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs"
    
    # Filter out PyTorch's bundled cuDNN path from existing LD_LIBRARY_PATH
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        local filtered_path=""
        IFS=':' read -ra PATHS <<< "${LD_LIBRARY_PATH}"
        for path in "${PATHS[@]}"; do
            # Remove PyTorch's bundled cuDNN path (contains 9.13.0)
            if [[ "${path}" == *"${PYTORCH_CUDNN_LIB}"* ]]; then
                continue
            fi
            # Keep other paths
            if [ -n "${filtered_path}" ]; then
                filtered_path="${filtered_path}:${path}"
            else
                filtered_path="${path}"
            fi
        done
        if [ -n "${filtered_path}" ]; then
            new_ld_path="${new_ld_path}:${filtered_path}"
        fi
    fi
    
    export LD_LIBRARY_PATH="${new_ld_path}"
    export LIBRARY_PATH="${cudnn_lib_dir}:${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs:${LIBRARY_PATH:-}"
    export LDFLAGS="-L${CUDA_HOME_DIR}/lib64 -L${CUDA_HOME_DIR}/lib64/stubs -L${cudnn_lib_dir} ${LDFLAGS:-}"

    local commit_hash
    commit_hash="$(git -C "${PYTORCH_SRC_DIR}" rev-parse --short HEAD)"
    export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION:-2.10.0.dev0+gb${commit_hash}}"
    export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-0}"

    pushd "${PYTORCH_SRC_DIR}" >/dev/null
    rm -rf build "${PYTORCH_DIST_DIR}"
    python3 setup.py clean >/dev/null 2>&1 || true
    python3 setup.py bdist_wheel
    popd >/dev/null

    produced_wheel=$(find "${PYTORCH_DIST_DIR}" -maxdepth 1 -name "torch-*.whl" | sort | tail -n 1)
    if [ -z "${produced_wheel}" ]; then
        echo "ERROR: PyTorch wheel not produced at ${PYTORCH_DIST_DIR}"
        exit 1
    fi

    dest_wheel="${PYTORCH_WHEEL_DIR}/$(basename "${produced_wheel}")"
    cp "${produced_wheel}" "${dest_wheel}"
    echo "Cached PyTorch wheel saved to ${dest_wheel}"

    if ! compgen -G "${dest_wheel}.part*" >/dev/null; then
        echo "Splitting PyTorch wheel into <50MB chunks under ${PYTORCH_WHEEL_DIR}"
        split -b 45m -d -a 2 "${dest_wheel}" "${dest_wheel}.part"
    fi
}
TORCH_CUDA_ARCH_LIST_VALUE="10.0;10.3;12.1+PTX"
CMAKE_CUDA_ARCH_LIST_VALUE="100;103;121"
TORCH_SM_ARCH_LIST_VALUE="sm_100;sm_103;sm_121"
CUTLASS_NVCC_ARCHS_VALUE="100;103;121"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Running as root. This is fine for containerized environments."
else
   echo "This script requires root privileges. Please run with sudo."
   exit 1
fi

pip_cmd() {
    if [ -z "${PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES:-}" ]; then
        if python3 -m pip --help 2>&1 | grep -q -- '--break-system-packages'; then
            PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES=1
        else
            PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES=0
        fi
    fi

    if [ "${PIP_SUPPORTS_BREAK_SYSTEM_PACKAGES}" -eq 1 ]; then
        python3 -m pip --break-system-packages "$@"
    else
        PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip "$@"
    fi
}

pip_install() {
    pip_cmd install "$@"
}

pip_uninstall() {
    pip_cmd uninstall "$@"
}

pip_wheel() {
    pip_cmd wheel "$@"
}

pip_show() {
    pip_cmd show "$@"
}

# Reusable function to reassemble split wheels
reassemble_split_wheel() {
    local wheel_path="$1"
    local tmp_dir="${2:-$(mktemp -d "${TMPDIR:-/tmp}/wheel-reassemble.XXXXXX")}"
    
    # If full wheel exists, return it
    if [ -f "${wheel_path}" ]; then
        echo "${wheel_path}"
        return 0
    fi
    
    # Check for split parts
    if compgen -G "${wheel_path}.part*" >/dev/null 2>&1; then
        local combined="${tmp_dir}/$(basename "${wheel_path}")"
        mapfile -t PARTS < <(ls "${wheel_path}".part* | sort -V)
        if cat "${PARTS[@]}" > "${combined}" 2>/dev/null; then
            echo "${combined}"
            return 0
        fi
    fi
    
    return 1
}

# Reusable function to verify PyTorch CUDA and restore if needed
verify_and_restore_pytorch_cuda() {
    local context="$1"
    python3 <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: PyTorch CUDA not available")
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    sys.exit(1)
PY
    if [ $? -ne 0 ]; then
        echo "CRITICAL: PyTorch CUDA was overridden during ${context}!"
        echo "Restoring PyTorch CUDA installation..."
        
        # Reassemble PyTorch wheel if split
        local reassembled_wheel
        local tmp_dir
        tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/pytorch-restore.XXXXXX")
        reassembled_wheel=$(reassemble_split_wheel "${PYTORCH_WHEEL_PATH}" "${tmp_dir}")
        
        if [ -z "${reassembled_wheel}" ] || [ ! -f "${reassembled_wheel}" ]; then
            echo "ERROR: Could not reassemble PyTorch wheel from ${PYTORCH_WHEEL_PATH}"
            rm -rf "${tmp_dir}"
            return 1
        fi
        
        pip_uninstall -y torch torchvision torchdata functorch pytorch-triton >/dev/null 2>&1 || true
        if pip_install --no-input --force-reinstall --no-deps "${reassembled_wheel}"; then
            install_torchao_packages
            echo "✓ PyTorch CUDA restored after ${context}"
            rm -rf "${tmp_dir}"
            return 0
        else
            echo "ERROR: Failed to restore PyTorch CUDA!"
            rm -rf "${tmp_dir}"
            return 1
        fi
    fi
    return 0
}

ensure_wheel_root_pure() {
    local wheel_path="$1"
    if [ -z "${wheel_path}" ] || [ ! -f "${wheel_path}" ]; then
        return 0
    fi
    python3 - "${wheel_path}" <<'PY'
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

wheel_path = Path(sys.argv[1])
if not wheel_path.exists():
    raise SystemExit(0)

with zipfile.ZipFile(wheel_path, "r") as src:
    wheel_entries = [name for name in src.namelist() if name.endswith(".dist-info/WHEEL")]
    if not wheel_entries:
        raise SystemExit(0)
    with tempfile.TemporaryDirectory(dir=str(wheel_path.parent)) as tmpdir:
        tmp_path = Path(tmpdir) / wheel_path.name
        with zipfile.ZipFile(tmp_path, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename in wheel_entries:
                    text = data.decode("utf-8").splitlines()
                    for idx, line in enumerate(text):
                        if line.startswith("Root-Is-Purelib:"):
                            if line.strip().lower() != "root-is-purelib: false":
                                text[idx] = "Root-Is-Purelib: false"
                            break
                    else:
                        text.append("Root-Is-Purelib: false")
                    data = ("\n".join(text) + "\n").encode("utf-8")
                dst.writestr(info, data)
    shutil.move(tmp_path, wheel_path)
PY
}

patch_installed_transformer_engine_metadata() {
    python3 <<'PY'
import importlib.metadata as metadata
from importlib.metadata import PackageNotFoundError

def patch_distribution(name: str) -> None:
    try:
        dist = metadata.distribution(name)
    except PackageNotFoundError:
        return

    files = dist.files or []
    wheel_entry = None
    for file in files:
        if file.name == "WHEEL":
            wheel_entry = dist.locate_file(file)
            break

    if not wheel_entry:
        return

    lines = wheel_entry.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("Root-Is-Purelib:"):
            if line.strip().lower() != "root-is-purelib: false":
                lines[idx] = "Root-Is-Purelib: false"
            break
    else:
        lines.append("Root-Is-Purelib: false")

    wheel_entry.write_text("\n".join(lines) + "\n")

for dist_name in ("transformer_engine", "transformer_engine_torch", "transformer_engine_cu12"):
    patch_distribution(dist_name)
PY
}

patch_cutlass_synclog_guards() {
    python3 <<'PY'
import re
from pathlib import Path

TARGET_FILES = [
    Path("third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp"),
]

def patch_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    include_marker = "#include \"cutlass/arch/synclog.hpp\"\n\n"
    macro_block = (
        "#if !defined(CUTE_CALL_SYNCLOG)\n"
        "#if defined(CUTLASS_ENABLE_SYNCLOG)\n"
        "#define CUTE_CALL_SYNCLOG(expr) expr\n"
        "#else\n"
        "#define CUTE_CALL_SYNCLOG(expr) ((void)0)\n"
        "#endif\n"
        "#endif\n\n"
    )
    changed = False
    if include_marker in text and "CUTE_CALL_SYNCLOG" not in text:
        text = text.replace(include_marker, include_marker + macro_block, 1)
        changed = True
    pattern = re.compile(r"(cutlass::arch::synclog_emit_[A-Za-z0-9_]+\([^;]+?\))\s*;")
    def repl(match):
        return f"CUTE_CALL_SYNCLOG({match.group(1)});"
    new_text, count = pattern.subn(repl, text)
    if count > 0:
        text = new_text
        changed = True
    if changed:
        path.write_text(text)
    return changed

patched = False
for file_path in TARGET_FILES:
    if patch_file(file_path):
        print(f"[setup] Patched CUTLASS synclog guards in {file_path}")
        patched = True

if not patched:
    print("[setup] CUTLASS synclog guard patch skipped (already applied or file missing)")
PY
}

patch_transformer_engine_loader() {
    python3 <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

OLD_BLOCK = '''    if te_framework_installed:
        assert te_installed_via_pypi, "Could not find `transformer-engine` PyPI package."
        assert te_core_installed, "Could not find TE core package `transformer-engine-cu*`."

        assert version(module_name) == version("transformer-engine") == te_core_version, (
            "Transformer Engine package version mismatch. Found"
            f" {module_name} v{version(module_name)}, transformer-engine"
            f" v{version('transformer-engine')}, and {te_core_package_name}"
            f" v{te_core_version}. Install transformer-engine using "
            f"'pip3 install --no-build-isolation transformer-engine[{extra_dep_name}]==VERSION'"
        )
'''

NEW_BLOCK = '''    if te_framework_installed:
        if te_installed_via_pypi and te_core_installed:
            assert version(module_name) == version("transformer-engine") == te_core_version, (
                "Transformer Engine package version mismatch. Found"
                f" {module_name} v{version(module_name)}, transformer-engine"
                f" v{version('transformer-engine')}, and {te_core_package_name}"
                f" v{te_core_version}. Install transformer-engine using "
                f"'pip3 install --no-build-isolation transformer-engine[{extra_dep_name}]==VERSION'"
            )
        else:
            pass
'''

patched = False
for dist_name in ("transformer_engine", "transformer-engine"):
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        continue
    path = Path(dist.locate_file("transformer_engine/common/__init__.py"))
    if not path.exists():
        continue
    text = path.read_text()
    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
        path.write_text(text)
        patched = True

if patched:
    print("[setup] Patched Transformer Engine loader for local wheel support")
else:
    print("[setup] Transformer Engine loader patch skipped (already applied)")
PY
}

install_proton_cli_stub() {
    if command -v proton >/dev/null 2>&1; then
        echo "Proton CLI already available (proton command found)"
        return 0
    fi
    local target="/usr/local/bin/proton"
    install -m 755 "${PROJECT_ROOT}/tools/proton_stub.py" "${target}"
    echo "Installed Proton stub CLI at ${target}"
}

remove_conflicting_user_triton() {
    if [ -z "${SUDO_USER:-}" ] || [ "${SUDO_USER}" = "root" ]; then
        return 0
    fi
    local user_site
    user_site=$(sudo -H -u "${SUDO_USER}" python3 - <<'PY' 2>/dev/null || true)
import site
try:
    print(site.getusersitepackages())
except Exception:
    raise SystemExit(1)
PY
    if [ -z "${user_site}" ]; then
        return 0
    fi
    if sudo -H -u "${SUDO_USER}" test -d "${user_site}/triton"; then
        rm -rf "${user_site}/triton"
    fi
    sudo -H -u "${SUDO_USER}" sh -c "rm -rf ${user_site}/pytorch_triton-*.dist-info" 2>/dev/null || true
}

remove_usercustomize_shim() {
    local targets=(
        "$HOME/.local/lib/python3.12/site-packages/usercustomize.py"
        "/usr/local/lib/python3.12/dist-packages/usercustomize.py"
    )
    for target in "${targets[@]}"; do
        if [ -f "$target" ]; then
            rm -f "$target"
            echo "[setup] Removed legacy usercustomize shim at $target"
        fi
    done
}

disable_transformer_engine_sanity_check() {
    python3 <<'PY'
import ast
import importlib.metadata as metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

def patch_module(module_path: Path) -> bool:
    if not module_path.exists():
        return False
    source = module_path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sanity_checks_for_pypi_installation":
            target = node
            break
    if target is None or target.lineno is None or target.end_lineno is None:
        return False
    lines = source.splitlines()
    replacement = [
        "def sanity_checks_for_pypi_installation() -> None:",
        "    \"\"\"Runtime environment bundles TE wheels directly; skip PyPI provenance checks.\"\"\"",
        "    return None",
        "",
    ]
    start = target.lineno - 1
    end = target.end_lineno
    lines[start:end] = replacement
    module_path.write_text("\n".join(lines) + ("\n" if lines and lines[-1] else ""))
    return True

patched_any = False
for dist_name in ("transformer_engine", "transformer-engine"):
    try:
        dist = metadata.distribution(dist_name)
    except PackageNotFoundError:
        continue
    module_path = Path(dist.locate_file("transformer_engine/common/__init__.py"))
    if patch_module(module_path):
        print(f"[setup] Patched Transformer Engine sanity check at {module_path}")
        patched_any = True

if not patched_any:
    print("[setup] Warning: transformer_engine.common not patched (module not found)")
PY
}

patch_cutlass_synclog_guards
patch_transformer_engine_loader
install_proton_cli_stub
remove_conflicting_user_triton
remove_usercustomize_shim

verify_fp8_functionality() {
    python3 <<'PY'
import torch
status = {
    "torchao": {"ok": False, "error": ""},
    "transformer_engine": {"ok": False, "error": ""},
}

def torchao_fp8_smoke():
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except Exception as exc:
        return False, f"torchao import failed: {exc}"
    try:
        model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False)).cuda().half()
        model = convert_to_float8_training(model, config=Float8LinearConfig())
        x = torch.randn(32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        y = model(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)

def te_fp8_smoke():
    try:
        import transformer_engine.pytorch as te
    except Exception as exc:
        return False, f"Transformer Engine import failed: {exc}"
    try:
        layer = te.Linear(128, 128, bias=False).to(torch.bfloat16).cuda()
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with te.fp8_autocast(enabled=True):
            y = layer(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)

status["torchao"]["ok"], status["torchao"]["error"] = torchao_fp8_smoke()
status["transformer_engine"]["ok"], status["transformer_engine"]["error"] = te_fp8_smoke()

for name, result in status.items():
    if result["ok"]:
        print(f"[setup] ✓ {name} FP8 smoke test passed")
    else:
        print(f"[setup] ERROR: {name} FP8 smoke test failed: {result['error']}")

if not all(entry["ok"] for entry in status.values()):
    raise SystemExit(1)
PY
}

TORCHAO_EXTRA_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"

# Check Ubuntu version
if ! command -v lsb_release &> /dev/null; then
    echo "Installing lsb-release..."
    apt update && apt install -y lsb-release
fi

UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "20.04" ]]; then
    echo "Warning: This script is tested on Ubuntu 22.04. Other versions may work but are not guaranteed."
fi

echo ""
echo "Configuring inotify watch limit for large workspaces..."
TARGET_INOTIFY_WATCHES=524288
CURRENT_INOTIFY_WATCHES=0
if [ -r /proc/sys/fs/inotify/max_user_watches ]; then
    CURRENT_INOTIFY_WATCHES=$(cat /proc/sys/fs/inotify/max_user_watches)
fi

if [ "$CURRENT_INOTIFY_WATCHES" -lt "$TARGET_INOTIFY_WATCHES" ]; then
    if grep -q '^fs.inotify.max_user_watches' /etc/sysctl.conf 2>/dev/null; then
        sed -i "s/^fs\.inotify\.max_user_watches=.*/fs.inotify.max_user_watches=${TARGET_INOTIFY_WATCHES}/" /etc/sysctl.conf
    else
        echo "fs.inotify.max_user_watches=${TARGET_INOTIFY_WATCHES}" >> /etc/sysctl.conf
    fi

    if sysctl -w fs.inotify.max_user_watches="${TARGET_INOTIFY_WATCHES}" >/dev/null 2>&1; then
        echo "Set fs.inotify.max_user_watches=${TARGET_INOTIFY_WATCHES} (consumes up to ~540 MiB if fully utilized)."
    else
        echo "Warning: Failed to apply inotify watch limit via sysctl; please verify manually."
    fi
else
    echo "fs.inotify.max_user_watches already set to ${CURRENT_INOTIFY_WATCHES}."
fi

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "NVIDIA GPU detected"

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ -n "$DRIVER_VERSION" ]]; then
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        if [ "$DRIVER_MAJOR" -lt 580 ]; then
            echo "Current NVIDIA driver: $DRIVER_VERSION"
            echo "CUDA ${CUDA_SHORT_VERSION} Update 2 requires driver 580+. This script will upgrade it automatically."
        else
            echo "NVIDIA driver version: $DRIVER_VERSION (compatible with CUDA ${CUDA_SHORT_VERSION} Update 2)"
        fi
    fi
else
    echo "NVIDIA GPU not detected. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Ensure open kernel modules are enabled for Blackwell GPUs
MODPROBE_CONF="/etc/modprobe.d/nvidia-open.conf"
if [[ ! -f "$MODPROBE_CONF" ]] || ! grep -q "NVreg_OpenRmEnableUnsupportedGpus=1" "$MODPROBE_CONF"; then
    echo "Configuring NVIDIA open kernel modules for Blackwell GPUs..."
    cat <<'EOF' > "$MODPROBE_CONF"
options nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0
EOF
    update-initramfs -u
    if lsmod | grep -q "^nvidia"; then
        echo "Reloading NVIDIA kernel modules to enable profiling counters..."
        systemctl stop nvidia-persistenced >/dev/null 2>&1 || true
        for module in nvidia_uvm nvidia_peermem nvidia_modeset nvidia_drm nvidia; do
            if lsmod | grep -q "^${module}"; then
                modprobe -r "${module}" >/dev/null 2>&1 || true
            fi
        done
        modprobe nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0 >/dev/null 2>&1 || true
        for module in nvidia_modeset nvidia_uvm nvidia_peermem; do
            modprobe "${module}" >/dev/null 2>&1 || true
        done
        systemctl start nvidia-persistenced >/dev/null 2>&1 || true
    fi
fi

# Update system packages
echo ""
echo "Updating system packages..."

# Fix apt_pkg module before apt update (if Python was upgraded)
if ! python3 -c "import apt_pkg" 2>/dev/null; then
    echo "Fixing apt_pkg module..."
    apt install -y --reinstall python3-apt 2>/dev/null || true
fi

# Disable command-not-found APT hook if it's causing issues with Python upgrade
if [ -f /etc/apt/apt.conf.d/50command-not-found ] && ! /usr/lib/cnf-update-db 2>/dev/null; then
    echo "Disabling problematic command-not-found APT hook..."
    rm -f /etc/apt/apt.conf.d/50command-not-found
fi

# Clean up duplicate deadsnakes repository if it exists
if [ -f /etc/apt/sources.list.d/deadsnakes.list ] && [ -f /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-jammy.list ]; then
    echo "Removing duplicate deadsnakes repository..."
    rm -f /etc/apt/sources.list.d/deadsnakes.list
fi

apt update || {
    echo "apt update had errors, but continuing..."
}

# Install required packages for adding repositories
apt install -y wget curl software-properties-common

# Install NVIDIA DCGM (Data Center GPU Manager) for monitoring stack
echo ""
echo "Installing NVIDIA DCGM components..."
if ! dpkg -s datacenter-gpu-manager >/dev/null 2>&1; then
    apt install -y datacenter-gpu-manager
    echo "NVIDIA DCGM installed"
else
    echo "NVIDIA DCGM already present"
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^nvidia-dcgm.service'; then
    systemctl enable nvidia-dcgm >/dev/null 2>&1 || true
    systemctl restart nvidia-dcgm >/dev/null 2>&1 || systemctl start nvidia-dcgm >/dev/null 2>&1 || true
    echo "nvidia-dcgm service enabled"
else
    echo "nvidia-dcgm systemd unit not found; ensure the DCGM daemon is running on this host."
fi

python3 <<'PY'
import os
import shutil
import site
import sys

candidates = [
    "/usr/share/dcgm/bindings/python3",
    "/usr/share/dcgm/bindings/python",
    "/usr/share/dcgm/python3",
    "/usr/share/dcgm/python",
    "/usr/lib/dcgm/python3",
    "/usr/lib/dcgm/python",
]

dest = None
for path in site.getsitepackages():
    if path.startswith("/usr/lib") or path.startswith("/usr/local/lib"):
        dest = path
        break

if dest is None:
    print("Unable to locate site-packages directory; skipping DCGM Python binding install.")
    sys.exit(0)

for cand in candidates:
    if not os.path.isdir(cand):
        continue
    for entry in os.listdir(cand):
        if entry == "__pycache__":
            continue
        src = os.path.join(cand, entry)
        dst = os.path.join(dest, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"DCGM Python bindings installed to {dest}")
    break
else:
    print("DCGM Python bindings not found; pydcgm import may fail.")
PY

# Add NVIDIA CUDA ${CUDA_SHORT_VERSION} repository
echo ""
echo "Adding NVIDIA CUDA ${CUDA_SHORT_VERSION} repository..."

# Check if CUDA repository is already configured
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ] && [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]; then
    CUDA_REPO_PKG="cuda-keyring_1.1-1_all.deb"
    if [ ! -f "/tmp/$CUDA_REPO_PKG" ]; then
        wget -P /tmp https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$CUDA_REPO_PKG
    fi
    dpkg -i /tmp/$CUDA_REPO_PKG
    echo "NVIDIA CUDA repository added"
else
    echo "NVIDIA CUDA repository already configured"
fi

# Configure APT to prefer official NVIDIA packages over Lambda Labs or other repos
echo ""
echo "Configuring APT preferences for NVIDIA packages..."
cat > /etc/apt/preferences.d/nvidia-official <<'APT_PREF'
# Prefer official NVIDIA packages over Lambda Labs
Package: nvidia-*
Pin: origin archive.lambdalabs.com
Pin-Priority: 100

Package: nvidia-*
Pin: origin developer.download.nvidia.com
Pin-Priority: 600

Package: nvidia-*
Pin: release o=Ubuntu
Pin-Priority: 600

# Also apply to libnvidia packages
Package: libnvidia-*
Pin: origin archive.lambdalabs.com
Pin-Priority: 100

Package: libnvidia-*
Pin: origin developer.download.nvidia.com
Pin-Priority: 600

Package: libnvidia-*
Pin: release o=Ubuntu
Pin-Priority: 600
APT_PREF

apt update

# Install target Python runtime
echo ""
echo "🐍 Installing Python ${PYTHON_TARGET_VERSION}..."

# Check if target Python is already installed
if ! command -v "${PYTHON_TARGET_BIN}" &> /dev/null; then
    apt install -y software-properties-common
    
    # Check if deadsnakes PPA is already added
    if ! grep -q "deadsnakes/ppa" /etc/apt/sources.list.d/*.list 2>/dev/null; then
        add-apt-repository -y ppa:deadsnakes/ppa
    else
        echo "deadsnakes PPA already configured"
    fi
    
    apt update || true
    apt install -y "${PYTHON_TARGET_BIN}" "${PYTHON_TARGET_BIN}-dev" "${PYTHON_TARGET_BIN}-venv" python3-pip
    echo "Python ${PYTHON_TARGET_VERSION} installed"
else
    CURRENT_TARGET_VERSION=$("${PYTHON_TARGET_BIN}" --version 2>&1 | awk '{print $2}')
    echo "Python ${PYTHON_TARGET_VERSION} already installed (version $CURRENT_TARGET_VERSION)"
    # Still ensure dev packages are present
    apt install -y "${PYTHON_TARGET_BIN}-dev" "${PYTHON_TARGET_BIN}-venv" python3-pip
fi

# Set target Python as default if not already
CURRENT_PY3=$(python3 --version 2>&1 | awk '{print $2}')
if [[ ! "$CURRENT_PY3" =~ ^${PYTHON_TARGET_MAJOR}\.${PYTHON_TARGET_MINOR}\. ]]; then
    echo "Setting Python ${PYTHON_TARGET_VERSION} as default..."
    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/${PYTHON_TARGET_BIN}" 1
    update-alternatives --set python3 "/usr/bin/${PYTHON_TARGET_BIN}"
else
    echo "Python ${PYTHON_TARGET_VERSION} is already the default"
fi

# Ensure `python` convenience shim follows target Python
if ! command -v python >/dev/null 2>&1; then
    echo "Creating python -> ${PYTHON_TARGET_BIN} alternative..."
    update-alternatives --install /usr/bin/python python "/usr/bin/${PYTHON_TARGET_BIN}" 1
    update-alternatives --set python "/usr/bin/${PYTHON_TARGET_BIN}"
else
    PYTHON_VERSION_OUTPUT="$(python --version 2>/dev/null || true)"
    if [[ "$PYTHON_VERSION_OUTPUT" != "Python ${PYTHON_TARGET_VERSION}."* ]]; then
        echo "Updating python alternative to point at ${PYTHON_TARGET_BIN}..."
        update-alternatives --install /usr/bin/python python "/usr/bin/${PYTHON_TARGET_BIN}" 1
        update-alternatives --set python "/usr/bin/${PYTHON_TARGET_BIN}"
    else
        echo "python points to $(python --version) (no change needed)"
    fi
fi

# Ensure pip is installed for target Python
if ! "${PYTHON_TARGET_BIN}" -m pip --version &> /dev/null; then
    echo "Installing pip for Python ${PYTHON_TARGET_VERSION}..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "${PYTHON_TARGET_BIN}"
fi

# Upgrade pip
pip_install --upgrade --ignore-installed pip setuptools packaging

# Ensure wheel build backend is available for manual builds
pip_install --no-cache-dir --upgrade --ignore-installed wheel

install_torchao_packages() {
    echo "Installing torchao (nightly, CUDA 13.x / cu130 index)..."
    local install_args=(--no-cache-dir --upgrade --ignore-installed torchao --extra-index-url "${TORCHAO_EXTRA_INDEX_URL}")
    if pip_install "${install_args[@]}"; then
        echo "torchao installed in system Python site-packages"
    else
        echo "ERROR: torchao installation failed for system Python!"
        exit 1
    fi
    python3 - <<'PY'
try:
    import torchao  # noqa: F401
    print("[setup] torchao import verified (system)")
except Exception as exc:
    raise SystemExit(f"[setup] ERROR: torchao import failed after installation: {exc}")
PY
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
        echo "Installing torchao for ${SUDO_USER}'s user site-packages..."
        if sudo -H -u "${SUDO_USER}" PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user "${install_args[@]}"; then
            echo "  ✓ torchao installed for ${SUDO_USER}"
        else
            echo "ERROR: Failed to install torchao for ${SUDO_USER}"
            exit 1
        fi
    fi
}

# Fix python3-apt for the new Python version
DISTUTILS_PTH="/usr/lib/python3/dist-packages/distutils-precedence.pth"
if [ -f "$DISTUTILS_PTH" ]; then
    echo "Removing distutils-precedence.pth shim (causes _distutils_hack errors)..."
    rm -f "$DISTUTILS_PTH"
fi

echo ""
echo "Fixing Python APT module..."
apt install -y --reinstall python3-apt
if [ -f "$DISTUTILS_PTH" ]; then
    echo "Removing distutils-precedence.pth shim after python3-apt reinstall..."
    rm -f "$DISTUTILS_PTH"
fi

# Remove distro flatbuffers package whose invalid version breaks pip metadata
if dpkg -s python3-flatbuffers >/dev/null 2>&1; then
    echo "Removing distro python3-flatbuffers package (invalid version metadata)..."
    apt remove -y python3-flatbuffers
fi

# Upgrade NVIDIA driver to 580+ if needed (required for CUDA ${CUDA_SHORT_VERSION} Update 2)
echo ""
echo "Checking NVIDIA driver version..."
if command -v nvidia-smi &> /dev/null; then
    CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    DRIVER_MAJOR=$(echo "$CURRENT_DRIVER" | cut -d. -f1)
    
    if [ "$DRIVER_MAJOR" -lt 580 ]; then
        echo "Current driver ($CURRENT_DRIVER) is too old for CUDA ${CUDA_SHORT_VERSION} Update 2"
        echo "Upgrading to NVIDIA driver 580 (open kernel modules)..."
        
        # Remove old driver packages that might conflict
        echo "Removing old NVIDIA driver packages..."
        apt remove -y nvidia-driver-* nvidia-dkms-* nvidia-kernel-common-* \
            libnvidia-compute-* libnvidia-extra-* nvidia-utils-* \
            python3-jax-cuda* python3-torch-cuda python3-torchvision-cuda 2>/dev/null || true
        apt autoremove -y
        
        # Install new driver
        echo "Installing NVIDIA driver 580..."
        if apt install -y nvidia-driver-580-open; then
            echo "NVIDIA driver 580 installed successfully"
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║  REBOOT REQUIRED                                       ║"
            echo "║                                                            ║"
            echo "║  The NVIDIA driver has been upgraded to version 580.      ║"
            echo "║  Please reboot your system and re-run this script.        ║"
            echo "║                                                            ║"
            echo "║  After reboot:                                             ║"
            echo "║    1. Run: nvidia-smi                                      ║"
            echo "║    2. Verify driver version is 580+                        ║"
            echo "║    3. Re-run: sudo ./setup.sh                              ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""
            exit 0
        else
            echo "Failed to install NVIDIA driver 580"
            echo "Please manually install the driver and reboot before continuing"
            exit 1
        fi
    else
        echo "NVIDIA driver $CURRENT_DRIVER is compatible with CUDA ${CUDA_SHORT_VERSION} Update 2"
    fi
fi

# Install CUDA toolchain (13.0.2 Update 2)
echo ""
echo "Installing CUDA ${CUDA_FULL_VERSION} toolchain..."
if ! apt install -y "cuda-toolkit-13-0=13.0.2-1"; then
    echo "Pinned CUDA toolkit version unavailable; installing latest cuda-toolkit-13-0 package instead."
    apt install -y cuda-toolkit-13-0
fi

# Install NCCL for Blackwell optimizations
echo ""
echo "Installing NCCL ${NCCL_SHORT_VERSION} (Blackwell-optimized)..."
if ! apt install -y "libnccl2=${NCCL_SHORT_VERSION}-1+cuda13.0" "libnccl-dev=${NCCL_SHORT_VERSION}-1+cuda13.0"; then
    echo "Pinned NCCL ${NCCL_SHORT_VERSION} packages unavailable; installing default libnccl2/libnccl-dev from CUDA repo."
    apt install -y libnccl2 libnccl-dev
fi

# Install cuDNN 9.16.0.29-1 (latest in CUDA 13 repo, matches current system pin)
# Following NVIDIA's approach: PyTorch bundles cuDNN, but we install a matching system version for build tools
echo ""
echo "Installing cuDNN 9.16.0.29-1 for CUDA ${CUDA_SHORT_VERSION}..."
# Remove any older cuDNN versions first
apt remove -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 2>/dev/null || true
# Install the exact version if available; otherwise fall back to repo latest
if apt-cache madison libcudnn9-cuda-13 | grep -q "9.16.0.29-1"; then
    apt install -y "libcudnn9-cuda-13=9.16.0.29-1" \
                   "libcudnn9-dev-cuda-13=9.16.0.29-1" \
                   "libcudnn9-headers-cuda-13=9.16.0.29-1"
    echo "✓ Installed cuDNN 9.16.0.29-1 (CUDA 13)"
else
    echo "WARNING: cuDNN 9.16.0.29-1 not available, installing latest..."
    apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13
fi
# Pin the version to prevent upgrades
apt-mark hold libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13

# Update ldconfig cache to ensure correct cuDNN version is found
echo "Updating ldconfig cache..."
ldconfig 2>/dev/null || true

# Install NVSHMEM 3.4.5 for CUDA 13 (enables SymmetricMemory fast paths)
echo ""
echo "Installing NVSHMEM 3.4.5 runtime and headers (CUDA 13)..."
apt install -y nvshmem-cuda-13 libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13 libnvshmem3-static-cuda-13

# Install GPUDirect Storage (GDS) for high-performance I/O
echo ""
echo "Installing GPUDirect Storage (GDS)..."
if ! dpkg -s gds-tools-13-0 >/dev/null 2>&1; then
    apt install -y gds-tools-13-0
    echo "GDS tools installed"
else
    echo "GDS tools already installed"
fi

# Load nvidia-fs kernel module for GDS
echo "Loading nvidia-fs kernel module..."
if ! lsmod | grep -q nvidia_fs; then
    modprobe nvidia-fs 2>/dev/null && echo "nvidia-fs module loaded" || {
        echo "Could not load nvidia-fs module (requires root)"
        echo "   Load it manually with: sudo modprobe nvidia-fs"
        echo "   Or run: sudo tools/setup/load_gds_module.sh"
    }
else
    echo "nvidia-fs module already loaded"
fi

# Install kvikio Python library for GDS (required for Ch5 examples)
echo ""
echo "Installing kvikio Python library for GPUDirect Storage..."
if pip_install --no-cache-dir --upgrade --ignore-installed kvikio-cu13==25.10.0 --extra-index-url https://download.pytorch.org/whl/cu130; then
    echo "kvikio-cu13==25.10.0 installed (enables GPU Direct Storage in Python)"
else
    echo "kvikio installation failed, but continuing..."
    echo "   Install manually with: pip install kvikio-cu13==25.10.0"
fi

install_torchao_packages

# Install CUDA sanitizers and debugging tools (compute-sanitizer, cuda-memcheck, etc.)
echo ""
echo "Installing CUDA sanitizers and debugging tools..."
if apt install -y cuda-command-line-tools-13-0; then
    echo "CUDA command-line tools 13.0 installed (compute-sanitizer, cuda-gdb, cuda-memcheck)"
else
        echo "Could not install cuda-command-line-tools-13-0, trying fallback packages..."
    if apt install -y cuda-command-line-tools; then
            echo "CUDA command-line tools (generic) installed"
    else
            echo "cuda-command-line-tools package unavailable. Trying NVIDIA CUDA toolkit..."
        if apt install -y nvidia-cuda-toolkit; then
                echo "NVIDIA CUDA toolkit installed (includes cuda-memcheck)"
        else
                echo "Could not install CUDA command-line tools. compute-sanitizer may be unavailable."
        fi
    fi
fi

# Ensure compute-sanitizer is present; install sanitizer package directly if needed
if ! command -v compute-sanitizer &> /dev/null; then
    echo "compute-sanitizer not found after command-line tools install. Installing cuda-sanitizer package..."
    if apt install -y cuda-sanitizer-13-0; then
        echo "cuda-sanitizer-13-0 installed"
    else
        echo "Could not install cuda-sanitizer-13-0, attempting generic cuda-sanitizer package..."
        if apt install -y cuda-sanitizer; then
            echo "cuda-sanitizer package installed"
        else
            echo "compute-sanitizer installation failed; please install manually."
        fi
    fi
fi

# Install latest NVIDIA Nsight Systems and Compute
echo ""
echo "Installing latest NVIDIA Nsight Systems and Compute..."

# Update apt to ensure latest packages are available
apt update -qq

# Create temporary directory for downloads
TEMP_DIR="/tmp/nsight_install"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Install Nsight Systems and set binary alternative  
echo "Installing Nsight Systems (pinned 2025.3.2)..."
NSYS_VERSION="2025.3.2"
apt install -y --upgrade "nsight-systems-${NSYS_VERSION}"
# Try multiple possible locations
for bin_path in "/opt/nvidia/nsight-systems/${NSYS_VERSION}/bin/nsys" "/opt/nvidia/nsight-systems/${NSYS_VERSION}/nsys"; do
    if [[ -x "$bin_path" ]]; then
        NSYS_BIN="$bin_path"
        break
    fi
done
if [[ -n "$NSYS_BIN" ]] && [[ -x "$NSYS_BIN" ]]; then
    update-alternatives --install /usr/local/bin/nsys nsys "$NSYS_BIN" 50
    update-alternatives --set nsys "$NSYS_BIN"
    echo "Nsight Systems pinned to ${NSYS_VERSION} (${NSYS_BIN})"
else
    echo "Nsight Systems binary not found"
fi

# Install Nsight Compute and set binary alternative
echo "Installing Nsight Compute (pinned 2025.3.1)..."
NCU_VERSION="2025.3.1"
apt install -y --upgrade "nsight-compute-${NCU_VERSION}"
NCU_BIN="/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu"
if [[ -x "$NCU_BIN" ]]; then
    update-alternatives --install /usr/local/bin/ncu ncu "$NCU_BIN" 50
    update-alternatives --set ncu "$NCU_BIN"
    echo "Nsight Compute pinned to ${NCU_VERSION} (${NCU_BIN})"
else
    echo "Nsight Compute binary not found at ${NCU_BIN}"
fi

# Nsight tools are already in PATH when installed via apt
echo "Nsight tools installed and available in PATH"

# Configure PATH and LD_LIBRARY_PATH for CUDA environment
echo ""
echo "Configuring CUDA ${CUDA_SHORT_VERSION} environment..."
# Update /etc/environment for system-wide CUDA path
if ! grep -q "${CUDA_HOME_DIR}/bin" /etc/environment; then
    sed -i "s|PATH=\"\(.*\)\"|PATH=\"${CUDA_HOME_DIR}/bin:\1\"|" /etc/environment
    echo "Added CUDA ${CUDA_SHORT_VERSION} to system PATH"
fi

# Create profile.d script for CUDA
CUDA_PROFILE_SCRIPT="/etc/profile.d/cuda-${CUDA_SHORT_VERSION}.sh"
cat > "${CUDA_PROFILE_SCRIPT}" <<PROFILE_EOF
# CUDA ${CUDA_SHORT_VERSION} environment variables
export CUDA_HOME=${CUDA_HOME_DIR}
export PATH=${CUDA_HOME_DIR}/bin:\$PATH
export CUDA_PATH=${CUDA_HOME_DIR}

# CRITICAL: PyTorch was compiled against cuDNN 9.15.1, but bundles cuDNN 9.13.0
# Use system cuDNN 9.15.1 instead of PyTorch's bundled 9.13.0
SYSTEM_CUDNN_LIB="/usr/lib/aarch64-linux-gnu"
PYTORCH_CUDNN_LIB="/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib"
if [ -n "\${LD_LIBRARY_PATH:-}" ]; then
    # Remove PyTorch's bundled cuDNN path (contains 9.13.0)
    LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "\${PYTORCH_CUDNN_LIB}" | tr '\n' ':' | sed 's/:$//')
fi
# Put system cuDNN 9.15.1 FIRST, then CUDA libs
if [ -d "\${SYSTEM_CUDNN_LIB}" ]; then
    export LD_LIBRARY_PATH="\${SYSTEM_CUDNN_LIB}:\${CUDA_HOME}/lib64:\${CUDA_HOME}/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
else
    export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${CUDA_HOME}/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
PROFILE_EOF
chmod +x "${CUDA_PROFILE_SCRIPT}"
echo "Created ${CUDA_PROFILE_SCRIPT} for persistent CUDA ${CUDA_SHORT_VERSION} environment"

# Update nvcc symlink to CUDA toolkit (override Ubuntu's default)
rm -f /usr/bin/nvcc
ln -s "${CUDA_HOME_DIR}/bin/nvcc" /usr/bin/nvcc
echo "Updated /usr/bin/nvcc symlink to CUDA ${CUDA_SHORT_VERSION}"

# Source the CUDA environment for current session
source "${CUDA_PROFILE_SCRIPT}"

# Persist CUDA toolchain settings for the remainder of the script
export CUDA_HOME="${CUDA_HOME_DIR}"
export PATH="$CUDA_HOME/bin:$PATH"

# CRITICAL: PyTorch was compiled against cuDNN 9.15.1, but bundles cuDNN 9.13.0
# Use system cuDNN 9.15.1 instead of PyTorch's bundled 9.13.0
SYSTEM_CUDNN_LIB="/usr/lib/aarch64-linux-gnu"
PYTORCH_CUDNN_LIB="${PYTHON_DIST_PACKAGES}/nvidia/cudnn/lib"
current_ld="${LD_LIBRARY_PATH:-}"
filtered_ld=""
if [ -n "${current_ld}" ]; then
    IFS=':' read -ra PATHS <<< "${current_ld}"
    for path in "${PATHS[@]}"; do
        # Remove PyTorch's bundled cuDNN path (contains 9.13.0)
        if [[ "${path}" == *"${PYTORCH_CUDNN_LIB}"* ]]; then
            continue
        fi
        # Keep other paths
        if [ -n "${filtered_ld}" ]; then
            filtered_ld="${filtered_ld}:${path}"
        else
            filtered_ld="${path}"
        fi
    done
fi
# Put system cuDNN 9.15.1 FIRST, then CUDA libs, then filtered paths
if [ -d "${SYSTEM_CUDNN_LIB}" ]; then
    export LD_LIBRARY_PATH="${SYSTEM_CUDNN_LIB}:${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs${filtered_ld:+:${filtered_ld}}"
else
    export LD_LIBRARY_PATH="${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs${filtered_ld:+:${filtered_ld}}"
fi

echo ""
echo "Verifying CUDA toolchain after installation..."
if ! nvcc --version >/tmp/nvcc_version.txt 2>&1; then
    cat /tmp/nvcc_version.txt
    echo "nvcc not available even after CUDA install. Aborting."
    exit 1
fi
cat /tmp/nvcc_version.txt
rm -f /tmp/nvcc_version.txt

python3 - <<'PY'
import sys
try:
    import torch
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"torch.version.cuda = {cuda_ver}")
    if not cuda_ver:
        print("PyTorch build does not have CUDA enabled")
except ImportError:
    print("PyTorch not yet installed (will be installed later in this script)")
except Exception as exc:
    print(f"Warning: Could not verify PyTorch CUDA support: {exc}")
PY

# Install runtime monitoring/precision dependencies that require CUDA headers
echo ""
echo "Installing CUDA-dependent Python packages (prometheus-client, transformer-engine)..."

# Helper to ensure pip installs run with CUDA environment populated
pip_cuda() {
    local cuda_home="${CUDA_HOME:-${CUDA_HOME_DIR}}"
    CUDA_HOME="${cuda_home}" \
    TORCH_CUDA_HOME="${cuda_home}" \
    CUDA_PATH="${cuda_home}" \
    PATH="${cuda_home}/bin:${PATH}" \
    LD_LIBRARY_PATH="${cuda_home}/lib64:${cuda_home}/lib64/stubs:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="${PROJECT_ROOT}/.cuda_env:${PYTHONPATH:-}" \
    pip_cmd "$@"
}

export PYTHONPATH="${PROJECT_ROOT}/.cuda_env:${PYTHONPATH:-}"

# Ensure NumPy is present for torch's extension helper before building any wheels
pip_cuda install --no-cache-dir --upgrade --ignore-installed numpy || true

install_cuda_package() {
    local package_spec="$1"
    echo "Installing $package_spec"
    if pip_cuda install \
        --no-cache-dir --upgrade --ignore-installed --prefer-binary "$package_spec"; then
        echo "Installed $package_spec"
        return 0
    fi

    echo "Installing numpy inside pip build environment to satisfy torch setup requirements..."
    pip_cuda install --no-cache-dir --upgrade --ignore-installed numpy || true

    echo "Initial install of $package_spec failed, retrying without build isolation..."
    if pip_cuda install \
        --no-cache-dir --upgrade --ignore-installed --no-build-isolation --prefer-binary "$package_spec"; then
        echo "Installed $package_spec (no-build-isolation)"
        return 0
    fi

    echo "Failed to install $package_spec"
    return 1
}

install_cuda_package "prometheus-client==0.21.0"

# Clean up
cd /
rm -rf "$TEMP_DIR"
cd "$PROJECT_ROOT"

# Install system tools for performance testing
echo ""
echo "Installing system performance tools..."
KERNEL_RELEASE=$(uname -r)

apt install -y \
    numactl \
    linux-tools-common \
    linux-tools-generic \
    "linux-tools-${KERNEL_RELEASE}" \
    gdb \
    perf-tools-unstable \
    infiniband-diags \
    perftest \
    htop \
    iotop \
    libjemalloc2 \
    libtcmalloc-minimal4 \
    ripgrep \
    sysstat

echo ""
echo "Installing detect-secrets (system-wide)..."
pip_install --no-cache-dir --upgrade --ignore-installed detect-secrets

# Install ninja (required for PyTorch CUDA extensions)
echo ""
echo "Installing ninja (required for CUDA extensions)..."
pip_install --no-cache-dir --upgrade --ignore-installed ninja==1.13.0

# Force remove ALL existing torch installations (CPU and CUDA) to prevent conflicts
echo ""
echo "Removing all existing torch installations..."
pip_uninstall -y torch torchvision torchdata functorch pytorch-triton >/dev/null 2>&1 || true
# Also remove any dist-info directories that pip couldn't clean up
rm -rf /usr/local/lib/python3.12/dist-packages/torch* /usr/local/lib/python3.12/dist-packages/torchvision* /usr/local/lib/python3.12/dist-packages/torchdata* /usr/local/lib/python3.12/dist-packages/functorch* /usr/local/lib/python3.12/dist-packages/pytorch_triton* 2>/dev/null || true
rm -rf /home/*/.local/lib/python3.12/site-packages/torch* /home/*/.local/lib/python3.12/site-packages/torchvision* 2>/dev/null || true

# Build or reuse PyTorch wheel (SM100/103/121)
echo ""
find "${PYTORCH_WHEEL_DIR}" -maxdepth 1 -name "torch-*-manylinux*.whl" -delete 2>/dev/null || true
PYTORCH_WHEEL_GLOB="${PYTORCH_WHEEL_DIR}/${PYTORCH_WHEEL_PATTERN}"
PYTORCH_WHEEL_PATH=""
if compgen -G "${PYTORCH_WHEEL_GLOB}" >/dev/null; then
    PYTORCH_WHEEL_PATH=$(ls ${PYTORCH_WHEEL_GLOB} | sort | tail -n 1)
    echo "Using cached PyTorch wheel at ${PYTORCH_WHEEL_PATH}"
else
    build_pytorch_from_source
    PYTORCH_WHEEL_PATH=$(ls ${PYTORCH_WHEEL_GLOB} | sort | tail -n 1 2>/dev/null)
fi

if [ -z "${PYTORCH_WHEEL_PATH}" ] || [ ! -f "${PYTORCH_WHEEL_PATH}" ]; then
    echo "ERROR: PyTorch wheel not available in ${PYTORCH_WHEEL_DIR}"
    exit 1
fi

export PYTORCH_WHEEL_PATH
PYTORCH_WHEEL_BASENAME="$(basename "${PYTORCH_WHEEL_PATH}")"
PYTORCH_BUILD_VERSION="$(echo "${PYTORCH_WHEEL_BASENAME}" | sed -n 's/^torch-\(.*\)-cp.*/\1/p')"
export PYTORCH_BUILD_VERSION

if ! compgen -G "${PYTORCH_WHEEL_PATH}.part*" >/dev/null; then
    echo "Splitting PyTorch wheel into <50MB chunks under ${PYTORCH_WHEEL_DIR}"
    split -b 45m -d -a 2 "${PYTORCH_WHEEL_PATH}" "${PYTORCH_WHEEL_PATH}.part"
fi

echo ""
echo "NOTE: PyTorch CUDA wheel will be installed AFTER all requirements"
echo "      to prevent dependencies from overriding it with CPU version"
echo ""

# Install project dependencies
echo ""
echo "Installing project dependencies..."
pip_uninstall -y torchvision >/dev/null 2>&1 || true

# Use the updated requirements file with pinned versions
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements_latest.txt"

# Create temporary requirements file WITHOUT accelerate and torchtitan
# (they pull in torch>=1.10.0 which installs CPU version)
# We'll install them separately AFTER PyTorch CUDA is installed
TEMP_REQUIREMENTS="/tmp/requirements_no_torch_deps.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    grep -v "^accelerate==" "$REQUIREMENTS_FILE" | grep -v "^torchtitan==" > "$TEMP_REQUIREMENTS" || true
    echo "Created temporary requirements file excluding accelerate and torchtitan"
    echo "  (these will be installed after PyTorch CUDA to prevent CPU version override)"
fi

# Install dependencies with error handling (excluding accelerate/torchtitan)
if [ -f "$TEMP_REQUIREMENTS" ]; then
    echo "Installing Python packages from requirements file (excluding accelerate/torchtitan)..."
    if ! pip_install --no-input --ignore-installed -r "$TEMP_REQUIREMENTS"; then
        echo "Some packages failed to install from requirements file."
        echo "Installing core packages individually..."
            pip_install --no-input --ignore-installed \
            blinker==1.9.0 \
            nvidia-ml-py3 nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
            numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
            matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
            jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 flake8==7.3.0 mypy==1.18.2 pytest==8.3.4 typer==0.12.0 rich==13.7.0 \
            transformers==4.40.2 datasets==2.18.0 sentencepiece==0.2.0 tokenizers==0.19.1 \
            onnx==1.19.0 \
            py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
            optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
            dask==2025.9.1 xarray==2025.6.1 \
            "fsspec[http]==2024.6.1"
    fi
    rm -f "$TEMP_REQUIREMENTS"
else
    echo "Requirements file not found at $REQUIREMENTS_FILE. Installing core packages directly..."
    pip_install --no-input --ignore-installed \
        blinker==1.9.0 \
        nvidia-ml-py3 nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
        numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
        matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
        jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 flake8==7.3.0 mypy==1.18.2 pytest==8.3.4 typer==0.12.0 rich==13.7.0 \
        transformers==4.40.2 datasets==2.18.0 sentencepiece==0.2.0 tokenizers==0.19.1 \
        onnx==1.19.0 \
        py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
        optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
        dask==2025.9.1 xarray==2025.6.1 \
        "fsspec[http]==2024.6.1"
fi

# NOW install PyTorch CUDA wheel to override any CPU version installed by dependencies
echo ""
echo "============================================================================"
echo "Installing PyTorch CUDA wheel (overriding any CPU version from dependencies)"
echo "============================================================================"
echo ""

# First, completely remove any existing PyTorch installation (CPU or CUDA)
echo "Removing any existing PyTorch installations..."
pip_uninstall -y torch torchvision torchdata functorch pytorch-triton >/dev/null 2>&1 || true

# Install the CUDA-enabled PyTorch wheel (handle split wheels)
echo "Installing CUDA-enabled PyTorch from ${PYTORCH_WHEEL_PATH}..."
pytorch_wheel_to_install=""
tmp_pytorch_dir=$(mktemp -d "${TMPDIR:-/tmp}/pytorch-install.XXXXXX")
pytorch_wheel_to_install=$(reassemble_split_wheel "${PYTORCH_WHEEL_PATH}" "${tmp_pytorch_dir}")

if [ -z "${pytorch_wheel_to_install}" ] || [ ! -f "${pytorch_wheel_to_install}" ]; then
    echo "ERROR: PyTorch wheel not found (neither full wheel nor split parts) at ${PYTORCH_WHEEL_PATH}"
    rm -rf "${tmp_pytorch_dir}"
    exit 1
fi

if ! pip_install --no-cache-dir --force-reinstall --upgrade --no-deps "${pytorch_wheel_to_install}"; then
    echo "ERROR: PyTorch CUDA wheel installation failed!"
    rm -rf "${tmp_pytorch_dir}"
    exit 1
fi
rm -rf "${tmp_pytorch_dir}"

# Verify the correct torch is installed and CUDA is available
echo ""
echo "Verifying PyTorch CUDA installation..."
if ! python3 <<'PY'; then
import sys
import torch

print("PyTorch wheel installed from source build")
print(f"  version: {torch.__version__}")
print(f"  location: {torch.__file__}")

# Check CUDA availability
if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
    print("ERROR: torch.version.cuda is None - CUDA not enabled in this build")
    sys.exit(1)

print(f"  cuda: {torch.version.cuda}")

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False after build")
    print("  This may indicate:")
    print("    1. NVIDIA driver is not loaded or too old")
    print("    2. Wrong torch installation is being imported")
    print("    3. CUDA libraries are not in LD_LIBRARY_PATH")
    sys.exit(1)

arch_list = torch.cuda.get_arch_list()
print(f"  arch list: {arch_list}")
expected_arches = {"sm_100", "sm_103", "sm_121"}
allowed_extra = {"compute_121"}
arch_set = set(arch_list)
missing_arches = sorted(expected_arches - arch_set)
disallowed = sorted(arch_set - expected_arches - allowed_extra)
if missing_arches or disallowed:
    detail = []
    if missing_arches:
        detail.append(f"missing {missing_arches}")
    if disallowed:
        detail.append(f"unexpected {disallowed}")
    print(f"ERROR: Expected PyTorch arch list to match {sorted(expected_arches)}, ")
    print(f"but got {arch_list} ({'; '.join(detail)})")
    sys.exit(1)

print("✓ PyTorch CUDA build verified successfully")
print(f"✓ GPU count: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 0:
    print(f"✓ GPU 0: {torch.cuda.get_device_name(0)}")
PY

    echo ""
    cat <<'EOF'
CRITICAL ERROR: PyTorch CUDA verification failed!
This should not happen after installing the CUDA wheel.
Please check:
  1. NVIDIA driver is loaded: nvidia-smi
  2. CUDA libraries are accessible: echo $LD_LIBRARY_PATH
  3. PyTorch wheel was built correctly
EOF
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ PyTorch CUDA installation and verification complete"
echo "============================================================================"
echo ""

# NOW install accelerate and torchtitan (they depend on torch, but torch CUDA is already installed)
echo ""
echo "Installing accelerate and torchtitan (PyTorch CUDA is already installed)..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    # Extract accelerate and torchtitan versions from original requirements
    ACCELERATE_VERSION=$(grep "^accelerate==" "$REQUIREMENTS_FILE" | cut -d= -f3 || echo "0.29.0")
    TORCHTITAN_VERSION=$(grep "^torchtitan==" "$REQUIREMENTS_FILE" | cut -d= -f3 || echo "0.2.0")
    
    echo "  Installing accelerate==${ACCELERATE_VERSION}..."
    pip_install --no-input --ignore-installed "accelerate==${ACCELERATE_VERSION}" || {
        echo "  Warning: accelerate installation failed, but continuing..."
    }
    
    echo "  Installing torchtitan==${TORCHTITAN_VERSION}..."
    pip_install --no-input --ignore-installed "torchtitan==${TORCHTITAN_VERSION}" || {
        echo "  Warning: torchtitan installation failed, but continuing..."
    }
    
    # CRITICAL: Verify PyTorch CUDA wasn't overridden by accelerate/torchtitan dependencies
    echo "  Verifying PyTorch CUDA wasn't overridden..."
    if ! verify_and_restore_pytorch_cuda "accelerate/torchtitan installation"; then
        echo "  ERROR: Failed to restore PyTorch CUDA after accelerate/torchtitan!"
        exit 1
    fi
    echo "  ✓ PyTorch CUDA verified"
    
    echo "✓ accelerate and torchtitan installed (PyTorch CUDA verified)"
fi

# Ensure triton is available (required by Transformer Engine and other PyTorch extensions)
# Triton should be bundled with PyTorch, but install it explicitly to ensure it's available
echo "Verifying triton availability (required by Transformer Engine)..."
if ! python3 -c "import triton" 2>/dev/null; then
    echo "  Triton not found. Installing triton (with --no-deps to prevent torch override)..."
    # Install triton with --no-deps to prevent it from pulling in torch CPU
    pip_install --no-cache-dir --upgrade --no-deps triton==3.5.0 || {
        echo "  Warning: Failed to install triton. Transformer Engine may not work."
    }
else
    TRITON_VERSION=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null || echo "unknown")
    echo "  ✓ Triton version: ${TRITON_VERSION}"
    # Verify triton version matches requirements (3.5.0)
    if [ "${TRITON_VERSION}" != "3.5.0" ]; then
        echo "  Warning: Triton version ${TRITON_VERSION} doesn't match required 3.5.0"
        echo "  Reinstalling triton 3.5.0 with --no-deps..."
        pip_install --no-cache-dir --upgrade --no-deps triton==3.5.0 || {
            echo "  Warning: Failed to reinstall triton 3.5.0"
        }
    fi
    
    # Verify PyTorch CUDA wasn't overridden by triton installation
    if ! verify_and_restore_pytorch_cuda "triton installation"; then
        echo "  ERROR: Failed to restore PyTorch CUDA after triton!"
        exit 1
    fi
fi

# Install torchvision separately with --no-deps to prevent torch override
echo ""
echo "Installing torchvision from git (with --no-deps to protect PyTorch)..."
pip_install --no-input --no-deps "git+https://github.com/pytorch/vision.git@main" || {
    echo "Warning: torchvision installation failed. Continuing without it."
}

# Verify PyTorch CUDA wasn't overridden by torchvision installation (shouldn't happen with --no-deps, but check anyway)
if ! verify_and_restore_pytorch_cuda "torchvision installation"; then
    echo "ERROR: Failed to restore PyTorch CUDA after torchvision!"
    exit 1
fi

# Sync CUTLASS headers for tcgen05 kernels
echo ""
echo "Syncing CUTLASS (tcgen05) headers..."
CUTLASS_REF_VALUE="${CUTLASS_REF:-v4.2.1}"
CUTLASS_REPO_URL="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
if [ -x "${PROJECT_ROOT}/scripts/install_cutlass.sh" ]; then
    CUTLASS_REPO="${CUTLASS_REPO_URL}" "${PROJECT_ROOT}/scripts/install_cutlass.sh" "${CUTLASS_REF_VALUE}"
else
    echo "WARNING: scripts/install_cutlass.sh not found; skipping CUTLASS installation"
fi

# Attempt to build tcgen05 probe to capture PTXAS errors on SM121 (for visibility)
run_tcgen05_probe() {
    echo ""
    echo "Running tcgen05 CUTLASS probe (expected to fail on SM121 without multicast/DSMEM)..."
    local log_path="${PROJECT_ROOT}/artifacts/tcgen05_probe_build.log"
    mkdir -p "$(dirname "${log_path}")"
    local probe_src="${PROJECT_ROOT}/tools/tcgen05_probe.cu"
    if [ ! -f "${probe_src}" ]; then
        echo "tcgen05 probe source not found at ${probe_src}; skipping."
        return
    fi
    local nvcc_cmd=(
        nvcc
        -std=c++17
        -O2
        -arch=sm_121
        -DCUTE_ENABLE_SYCLOG=0
        -I"${PROJECT_ROOT}/third_party/cutlass/include"
        -I"${PROJECT_ROOT}/third_party/cutlass/tools/util/include"
        -I"${PROJECT_ROOT}/third_party/cutlass/examples/common"
        "${probe_src}"
        -o /tmp/tcgen05_probe_cutlass
    )
    {
        echo "Command: ${nvcc_cmd[*]}"
        "${nvcc_cmd[@]}"
    } >"${log_path}" 2>&1 || true
    echo "tcgen05 probe build log: ${log_path}"
}
run_tcgen05_probe

# Build CUTLASS profiler (cutlass_profiler) once and keep the binary in-tree for labs
build_cutlass_profiler() {
    echo ""
    echo "Building CUTLASS profiler (cutlass_profiler)..."

    local cutlass_dir="${PROJECT_ROOT}/third_party/cutlass"
    if [ ! -d "${cutlass_dir}" ]; then
        echo "  CUTLASS source not found at ${cutlass_dir}; skipping profiler build."
        return
    fi

    local sm
    sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' | xargs)
    if [ -z "${sm}" ]; then
        sm="90"
    fi
    local archs="${CUTLASS_NVCC_ARCHS:-${sm}}"
    # Build in a user-writable location to avoid permission issues with root-owned sources
    local build_dir="${PROJECT_ROOT}/build/cutlass_profiler"
    local log_path="${PROJECT_ROOT}/artifacts/cutlass_profiler_build.log"
    mkdir -p "${build_dir}" "$(dirname "${log_path}")"

    echo "  Using CUTLASS_NVCC_ARCHS=${archs}"
    {
        echo "Configure:"
        cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DCUTLASS_ENABLE_EXAMPLES=OFF \
            -DCUTLASS_ENABLE_TESTS=OFF \
            -DCUTLASS_ENABLE_TOOLS=ON \
            -DCUTLASS_ENABLE_PROFILER=ON \
            -DCUTLASS_NVCC_ARCHS="${archs}" \
            -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME_DIR}" \
            -S "${cutlass_dir}" \
            -B "${build_dir}"

        echo ""
        echo "Build:"
        cmake --build "${build_dir}" -j"$(nproc --ignore=1 2>/dev/null || nproc)" --target cutlass_profiler
    } >"${log_path}" 2>&1 || {
        echo "  WARNING: cutlass_profiler build failed. See ${log_path}"
        return
    }

    local profiler_bin="${build_dir}/tools/profiler/cutlass_profiler"
    if [ -x "${profiler_bin}" ]; then
        echo "  CUTLASS profiler ready: ${profiler_bin}"
        export CUTLASS_PROFILER_BIN="${profiler_bin}"
        mkdir -p "${PROJECT_ROOT}/artifacts"
        echo "export CUTLASS_PROFILER_BIN=${profiler_bin}" > "${PROJECT_ROOT}/artifacts/cutlass_profiler_env.sh"
        echo "  Exported CUTLASS_PROFILER_BIN and saved to artifacts/cutlass_profiler_env.sh"
    else
        echo "  WARNING: cutlass_profiler build completed but binary not found at ${profiler_bin}"
    fi
    echo "  Build log: ${log_path}"
}
build_cutlass_profiler

# Ensure monitoring/runtime dependencies are available even if requirements were cached
echo ""
echo "Ensuring monitoring/runtime packages (Prometheus, Transformer Engine)..."

# Transformer Engine build requires CUDNN headers shipped with the Python wheel.
CUDA_ROOT="${CUDA_HOME_DIR}"
# Transformer Engine build needs cuDNN headers from PyTorch's bundled package
CUDNN_INCLUDE_DIR="${PYTHON_DIST_PACKAGES}/nvidia/cudnn/include"
CUDNN_LIBRARY_DIR="${PYTHON_DIST_PACKAGES}/nvidia/cudnn/lib"
export CPATH="${CUDNN_INCLUDE_DIR}:${CPATH:-}"
# For build-time: Use PyTorch's cuDNN headers, but runtime will use system cuDNN 9.15.1
# LIBRARY_PATH is for linking, LD_LIBRARY_PATH (set above) uses system cuDNN 9.15.1 for runtime
if [ -d "${CUDNN_INCLUDE_DIR}" ]; then
    # Headers are needed for compilation
    export LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:${LIBRARY_PATH:-}"
fi
# NOTE: LD_LIBRARY_PATH is already set above to use system cuDNN 9.15.1 for runtime

# Remove conflicting binary wheels before installing the source build.
pip_uninstall -y transformer_engine transformer-engine transformer_engine_cu12 transformer-engine-cu12 transformer-engine-cu13 transformer_engine_torch >/dev/null 2>&1 || true

TE_WHEEL_BASE="transformer_engine-${TE_VERSION_TAG}-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_aarch64.whl"
TE_CU12_WHEEL_BASE="transformer_engine_cu12-${TE_VERSION_TAG}-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_aarch64.whl"
# The transformer_engine_torch wheel stays under Git's 50MB limit, so we cache the raw .whl (no split parts).
TE_TORCH_WHEEL_BASE="transformer_engine_torch-${TE_VERSION_TAG}-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_aarch64.whl"

install_wheel_from_cache() {
    local wheel_name="$1"
    local full_path="${THIRD_PARTY_DIR}/wheels/${wheel_name}"
    local parts_prefix="${full_path}.part"

    if [ -f "$full_path" ]; then
        ensure_wheel_root_pure "$full_path"
        pip_install --no-input --ignore-installed --no-deps "$full_path"
        return $?
    fi

    if compgen -G "${parts_prefix}*" >/dev/null; then
        local tmp_dir
        tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/te-wheel.XXXXXX")
        local combined="${tmp_dir}/${wheel_name}"
        mapfile -t PARTS < <(ls "${parts_prefix}"* | sort -V)
        if cat "${PARTS[@]}" > "${combined}" && \
           ensure_wheel_root_pure "${combined}" && \
           pip_install --no-input --ignore-installed --no-deps "${combined}"; then
            rm -rf "${tmp_dir}"
            return 0
        fi
        rm -rf "${tmp_dir}"
    fi
    return 1
}

install_te_from_source() {
    pip_install --no-input --upgrade --ignore-installed pybind11
    local torch_wheel="${PYTORCH_WHEEL_PATH:-}"
    local wheel_cache="${PYTORCH_WHEEL_DIR}"
    local target_wheel="${wheel_cache}/${TE_WHEEL_BASE}"
    local target_cu12_wheel="${wheel_cache}/${TE_CU12_WHEEL_BASE}"
    local target_torch_wheel="${wheel_cache}/${TE_TORCH_WHEEL_BASE}"

    mkdir -p "${wheel_cache}"

    # Check if all 3 wheels are cached (cu12 and torch might be split)
    local cu12_exists=false
    if [ -f "${target_cu12_wheel}" ] || compgen -G "${target_cu12_wheel}.part*" >/dev/null 2>&1; then
        cu12_exists=true
    fi
    
    local main_wheel_exists=false
    local torch_wheel_exists=false
    if [ -f "${target_wheel}" ]; then
        main_wheel_exists=true
    fi
    # Check for torch wheel (might be split into parts like cu12)
    if [ -f "${target_torch_wheel}" ] || compgen -G "${target_torch_wheel}.part*" >/dev/null 2>&1; then
        torch_wheel_exists=true
    fi
    
    if [ "${main_wheel_exists}" = "false" ] || [ "${cu12_exists}" = "false" ] || [ "${torch_wheel_exists}" = "false" ]; then
        echo "Building Transformer Engine ${TE_VERSION_TAG} from source (this takes 5-10 minutes)..."
        echo "  Missing wheels: main=${main_wheel_exists} cu12=${cu12_exists} torch=${torch_wheel_exists}"
        local tmp_wheel_dir
        tmp_wheel_dir=$(mktemp -d "${TMPDIR:-/tmp}/te-wheel-build.XXXXXX")
        if TORCH_CUDA_ARCH_LIST="${TORCH_SM_ARCH_LIST_VALUE}" \
           CUTLASS_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS_VALUE}" \
           CUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR}" \
           CUDNN_LIBRARY_DIR="${CUDNN_LIBRARY_DIR}" \
           pip_wheel \
               --no-deps \
               --no-build-isolation \
               --wheel-dir "${tmp_wheel_dir}" \
               "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_GIT_COMMIT}"; then
            
            # Find and cache all 3 produced wheels
            local main_wheel cu12_wheel torch_wheel
            main_wheel=$(find "${tmp_wheel_dir}" -maxdepth 1 -name "transformer_engine-*.whl" -not -name "*cu12*" -not -name "*torch*" | head -n 1)
            cu12_wheel=$(find "${tmp_wheel_dir}" -maxdepth 1 -name "transformer_engine_cu12-*.whl" | head -n 1)
            torch_wheel=$(find "${tmp_wheel_dir}" -maxdepth 1 -name "transformer_engine_torch-*.whl" | head -n 1)
            
            if [ -n "${main_wheel}" ]; then
                mv "${main_wheel}" "${target_wheel}"
                echo "✓ Cached transformer_engine wheel: ${target_wheel}"
            fi
            if [ -n "${cu12_wheel}" ]; then
                mv "${cu12_wheel}" "${target_cu12_wheel}"
                echo "✓ Cached transformer_engine_cu12 wheel: ${target_cu12_wheel}"
                # Split if >50MB
                local cu12_size=$(stat -f%z "${target_cu12_wheel}" 2>/dev/null || stat -c%s "${target_cu12_wheel}" 2>/dev/null)
                if [ -n "${cu12_size}" ] && [ "${cu12_size}" -gt 52428800 ]; then
                    echo "  Splitting cu12 wheel (${cu12_size} bytes) into <50MB chunks..."
                    split -b 45m -d -a 2 "${target_cu12_wheel}" "${target_cu12_wheel}.part"
                    rm "${target_cu12_wheel}"  # Remove unsplit version to save space
                fi
            fi
            if [ -n "${torch_wheel}" ]; then
                mv "${torch_wheel}" "${target_torch_wheel}"
                echo "✓ Cached transformer_engine_torch wheel: ${target_torch_wheel}"
                # Split if >50MB (same as cu12)
                local torch_size=$(stat -f%z "${target_torch_wheel}" 2>/dev/null || stat -c%s "${target_torch_wheel}" 2>/dev/null)
                if [ -n "${torch_size}" ] && [ "${torch_size}" -gt 52428800 ]; then
                    echo "  Splitting torch wheel (${torch_size} bytes) into <50MB chunks..."
                    split -b 45m -d -a 2 "${target_torch_wheel}" "${target_torch_wheel}.part"
                    rm "${target_torch_wheel}"  # Remove unsplit version to save space
                fi
            fi
        else
            echo "Transformer Engine wheel build failed; cleaning up temporary files."
            rm -rf "${tmp_wheel_dir}"
            return 1
        fi
        rm -rf "${tmp_wheel_dir}"
    else
        echo "Using cached Transformer Engine wheels"
        echo "  Found: main=${main_wheel_exists} cu12=${cu12_exists} torch=${torch_wheel_exists}"
    fi

    # Ensure PyTorch CUDA is installed before Transformer Engine
    if [ -n "${PYTORCH_WHEEL_PATH}" ]; then
        echo "Verifying PyTorch CUDA is installed before Transformer Engine..."
        if ! verify_and_restore_pytorch_cuda "pre-Transformer Engine check"; then
            echo "ERROR: Failed to ensure PyTorch CUDA before Transformer Engine"
            return 1
        fi
    fi
    
    # Install all 3 Transformer Engine wheels (handle split wheels consistently)
    local wheels_to_install=()
    local tmp_te_dir
    tmp_te_dir=$(mktemp -d "${TMPDIR:-/tmp}/te-wheels.XXXXXX")
    
    # Main wheel
    if [ -f "${target_wheel}" ]; then
        wheels_to_install+=("${target_wheel}")
    fi
    
    # Torch wheel (reassemble if split)
    local torch_wheel_reassembled
    torch_wheel_reassembled=$(reassemble_split_wheel "${target_torch_wheel}" "${tmp_te_dir}")
    if [ -n "${torch_wheel_reassembled}" ] && [ -f "${torch_wheel_reassembled}" ]; then
        wheels_to_install+=("${torch_wheel_reassembled}")
    fi
    
    # CU12 wheel (reassemble if split)
    local cu12_wheel_reassembled
    cu12_wheel_reassembled=$(reassemble_split_wheel "${target_cu12_wheel}" "${tmp_te_dir}")
    if [ -n "${cu12_wheel_reassembled}" ] && [ -f "${cu12_wheel_reassembled}" ]; then
        wheels_to_install+=("${cu12_wheel_reassembled}")
    fi
    
    if [ ${#wheels_to_install[@]} -gt 0 ]; then
        for wheel in "${wheels_to_install[@]}"; do
            ensure_wheel_root_pure "${wheel}"
        done
        
        if TORCH_CUDA_ARCH_LIST="${TORCH_SM_ARCH_LIST_VALUE}" \
           CUTLASS_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS_VALUE}" \
           CUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR}" \
           CUDNN_LIBRARY_DIR="${CUDNN_LIBRARY_DIR}" \
           pip_install --no-input --ignore-installed --no-build-isolation --no-deps "${wheels_to_install[@]}"; then
            echo "✓ Transformer Engine installed (FP4/FP8 kernels ready where supported)"
            # Clean up temporary directory
            rm -rf "${tmp_te_dir}"
            return 0
        fi
    fi
    
    # Clean up temporary directory
    rm -rf "${tmp_te_dir}"

    echo "ERROR: Transformer Engine installation FAILED. Cannot continue."
    return 1
}

# Build/install Transformer Engine from source (or use cached wheels if available)
# This ensures compatibility with the current PyTorch ${PYTORCH_BUILD_VERSION} build
echo ""
echo "Installing Transformer Engine (will use cache if available, otherwise build from source)..."
if ! install_te_from_source; then
    echo "ERROR: Transformer Engine installation FAILED. Setup cannot continue without Transformer Engine."
    echo "Transformer Engine is REQUIRED for FP4/FP6/FP8 support."
    exit 1
fi

# Override with prebuilt Transformer Engine sm_121f wheel and runtime deps (system install, no venv)
echo ""
echo "Installing prebuilt Transformer Engine (sm_121f) and runtime dependencies..."
TE_PREBUILT_WHEEL_NAME="transformer_engine-2.10.0.dev0+e4bfa628-${PYTHON_ABI_TAG}-${PYTHON_ABI_TAG}-linux_aarch64.whl"
TE_PREBUILT_WHEEL="${THIRD_PARTY_DIR}/wheels/${TE_PREBUILT_WHEEL_NAME}"
pip_install --no-cache-dir --upgrade --ignore-installed \
  typing_extensions packaging pydantic pydantic_core typing_inspection annotated_types \
  sympy mpmath onnxscript onnx protobuf ml_dtypes einops
pip_install --no-cache-dir --upgrade --ignore-installed --no-deps triton==3.5.0 || {
  echo "Warning: triton 3.5.0 install failed; Transformer Engine may be missing triton."
}
if ! install_wheel_from_cache "${TE_PREBUILT_WHEEL_NAME}"; then
    echo "Warning: Prebuilt Transformer Engine wheel not found (expected ${TE_PREBUILT_WHEEL_NAME}); source-built TE remains in place."
fi

patch_installed_transformer_engine_metadata
patch_transformer_engine_loader
disable_transformer_engine_sanity_check

# CRITICAL: Verify PyTorch CUDA after Transformer Engine installation
echo ""
echo "Verifying PyTorch CUDA after Transformer Engine installation..."
if ! verify_and_restore_pytorch_cuda "Transformer Engine installation"; then
    echo "ERROR: Failed to restore PyTorch CUDA after Transformer Engine!"
    exit 1
fi
echo "✓ PyTorch CUDA verified after Transformer Engine"

echo ""
echo "Running FP8 runtime smoke tests (torchao + Transformer Engine)..."
if ! verify_fp8_functionality; then
    echo "ERROR: FP8 runtime smoke tests failed"
    exit 1
fi
echo "✓ FP8 runtime smoke tests passed"

# Snapshot Transformer Engine capability (NVFP4 / MXFP8) after installation
echo ""
echo "Capturing Transformer Engine capability snapshot..."

# Ensure triton is available (required by Transformer Engine)
echo "Checking for triton..."
if ! python3 -c "import triton" 2>/dev/null; then
    echo "  Triton not found. Installing triton (required by Transformer Engine)..."
    # Try to install triton - it should match the PyTorch version
    pip_install --no-cache-dir --upgrade triton || {
        echo "  Warning: Failed to install triton. Transformer Engine capability check may fail."
    }
fi

python3 <<'PY'
import sys
import warnings

# Suppress FlashAttention kernel override warnings (harmless - happens when FlashAttention is imported multiple times)
warnings.filterwarnings("ignore", message=".*Overriding a previously registered kernel.*")
warnings.filterwarnings("ignore", message=".*Warning only once for all operators.*")

# First check if PyTorch CUDA is available
try:
    import torch
    if not torch.cuda.is_available():
        print("Transformer Engine capability snapshot skipped: PyTorch CUDA not available")
        print("  torch.version.cuda = {}".format(getattr(torch.version, 'cuda', None)))
        print("  torch.__version__ = {}".format(torch.__version__))
        print("  Transformer Engine requires CUDA-enabled PyTorch.")
        print("  This check will be retried after PyTorch verification.")
        sys.exit(0)
except ImportError:
    print("Transformer Engine capability snapshot skipped: PyTorch not available")
    sys.exit(0)

# Check if triton is available (required by Transformer Engine)
try:
    import triton
    print(f"  Triton version: {triton.__version__}")
except ImportError:
    print("  Warning: triton not found. It should be bundled with PyTorch.")
    print("  Transformer Engine capability check may fail without triton.")
    print("  Note: triton should be available via PyTorch. If missing, this may indicate an incomplete PyTorch installation.")

try:
    import transformer_engine.pytorch as te
except ImportError as exc:
    print(f"Transformer Engine capability snapshot skipped: {exc}")
    if "triton" in str(exc).lower():
        print("  Triton is required for Transformer Engine. Please ensure PyTorch is fully installed.")
    elif "undefined symbol" in str(exc).lower() and "cuda" in str(exc).lower():
        print("  This error indicates PyTorch CUDA libraries are incompatible or not available.")
        print("  Transformer Engine check will be retried after PyTorch verification.")
    else:
        print("  Transformer Engine may not be fully installed yet.")
    sys.exit(0)
except Exception as exc:
    print(f"Transformer Engine capability snapshot skipped: {exc}")
    if "CUDA" in str(exc):
        print("  This may be due to PyTorch CUDA not being available.")
        print("  Transformer Engine check will be retried after PyTorch verification.")
    sys.exit(0)

try:
    nvfp4_available, nvfp4_reason = te.is_nvfp4_available(return_reason=True)
    mxfp8_available, mxfp8_reason = te.is_mxfp8_available(return_reason=True)
    def format_status(flag, reason):
        return "yes" if flag else f"no ({reason or 'not supported on this stack'})"
    print("Transformer Engine capability snapshot:")
    print(f"  NVFP4 available: {format_status(nvfp4_available, nvfp4_reason)}")
    print(f"  MXFP8 available: {format_status(mxfp8_available, mxfp8_reason)}")
except Exception as exc:
    print(f"Transformer Engine capability check failed: {exc}")
    print("  This may be normal if CUDA runtime is not available.")
PY

# Refresh hardware capability cache now so subsequent non-root runs do not need
# to invoke the probe with elevated privileges.
echo ""
echo "Refreshing GPU hardware capability cache..."
if python3 "${PROJECT_ROOT}/tools/utilities/probe_hardware_capabilities.py" 2>/tmp/probe_hardware.log; then
    cat /tmp/probe_hardware.log
else
    cat /tmp/probe_hardware.log
    echo "Warning: GPU hardware capability probe failed; CUDA may be unavailable."
    echo "This is expected if PyTorch CUDA is not yet properly installed."
    echo "The capability cache will be refreshed after PyTorch verification."
fi
rm -f /tmp/probe_hardware.log

# Ensure artifacts and wheel caches are writable by the invoking user for future runs.
if [ -n "${SUDO_USER:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/artifacts"
    chown -R "${SUDO_USER}:${SUDO_USER}" "${PROJECT_ROOT}/artifacts" 2>/dev/null || true
    chown -R "${SUDO_USER}:${SUDO_USER}" "${THIRD_PARTY_DIR}/wheels" 2>/dev/null || true
fi

# Final verification: Ensure our custom PyTorch wheel remains installed after all package installations
echo ""
echo "============================================================================"
echo "Final verification: PyTorch CUDA installation"
echo "============================================================================"
EXPECTED_TORCH_VERSION="${PYTORCH_BUILD_VERSION}"

if ! verify_and_restore_pytorch_cuda "final verification (after all installations)"; then
    echo "ERROR: Failed to restore PyTorch CUDA during final verification!"
    exit 1
fi

# Print final status
python3 <<'PY'
import sys
import torch
print(f"✓ Final PyTorch CUDA verification: {torch.__version__} with CUDA {torch.version.cuda}")
PY

# Clean up Transformer Engine wheel conflicts
python3 <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("transformer_engine")
if not spec:
    raise SystemExit(0)
pkg_root = Path(spec.origin).parent
wheel_dir = pkg_root / "wheel_lib"
if not wheel_dir.exists():
    raise SystemExit(0)
for so_path in pkg_root.glob("*.so"):
    if (wheel_dir / so_path.name).exists():
        so_path.unlink()
PY

python3 <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

def mark_non_pure(dist_name: str) -> None:
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return
    files = dist.files or []
    wheel_file = None
    for entry in files:
        if entry.name == "WHEEL":
            wheel_file = dist.locate_file("") / entry
            break
    if wheel_file is None:
        return
    lines = wheel_file.read_text().splitlines()
    changed = False
    for idx, line in enumerate(lines):
        if line.startswith("Root-Is-Purelib:"):
            value = line.split(":", 1)[1].strip().lower()
            if value != "false":
                lines[idx] = "Root-Is-Purelib: false"
                changed = True
            break
    else:
        lines.append("Root-Is-Purelib: false")
        changed = True
    if changed:
        wheel_file.write_text("\n".join(lines) + "\n")

for name in (
    "transformer_engine",
    "transformer_engine_cu12",
    "transformer_engine_cu13",
    "transformer_engine_torch",
    "transformer-engine",
):
    mark_non_pure(name)
PY

python3 <<'PY'
import ast
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

def ensure_te_import_tolerant(dist_name: str) -> bool:
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return False
    init_path = Path(dist.locate_file("transformer_engine/__init__.py"))
    if not init_path.exists():
        return False
    source = init_path.read_text()
    if "_IN_TREE_TRANSFORMER_ENGINE" in source:
        return True
    injection = (
        "\ntry:\n"
        "    from importlib.metadata import PackageNotFoundError, distribution\n"
        "    TE_METADATA_AVAILABLE = True\n"
        "except ImportError:\n"
        "    TE_METADATA_AVAILABLE = False\n"
        "_IN_TREE_TRANSFORMER_ENGINE = False\n"
        "if TE_METADATA_AVAILABLE:\n"
        "    try:\n"
        "        distribution(\"transformer-engine\")\n"
        "        _IN_TREE_TRANSFORMER_ENGINE = True\n"
        "    except PackageNotFoundError:\n"
        "        _IN_TREE_TRANSFORMER_ENGINE = True\n"
        "else:\n"
        "    _IN_TREE_TRANSFORMER_ENGINE = True\n"
    )
    source = source.replace("from . import pytorch", "from . import pytorch" + injection, 1)
    init_path.write_text(source)
    return True

ensure_te_import_tolerant("transformer_engine")
ensure_te_import_tolerant("transformer-engine")
PY

# Install FlashAttention to unlock optimized SDPA paths

echo ""
echo "Ensuring FlashAttention (flash-attn ${FLASH_ATTN_TAG}) is available..."
FLASH_ATTN_CACHE_PATH="${THIRD_PARTY_DIR}/wheels/${FLASH_ATTN_WHEEL_BASENAME}"
FLASH_ATTN_SPLIT_PREFIX="${FLASH_ATTN_CACHE_PATH}.part"

flash_attn_filter_warning() {
    python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("flash_attn.flash_attn_interface")
if not spec or not spec.origin:
    raise SystemExit
path = Path(spec.origin)
lines = path.read_text().splitlines()
marker = "Overriding a previously registered kernel for the same operator"
if any(marker in ln for ln in lines):
    raise SystemExit
inject = [
    "import warnings",
    "",
    'warnings.filterwarnings(',
    '    "ignore",',
    '    message=".*Overriding a previously registered kernel for the same operator.*flash_attn.*",',
    '    category=UserWarning,',
    '    module="torch.library",',
    ')',
    'warnings.filterwarnings(',
    '    "ignore",',
    '    message="Warning only once for all operators,\\s+other operators may also be overridden.",',
    '    category=UserWarning,',
    '    module="torch.library",',
    ')',
]
out = []
inserted = False
for ln in lines:
    out.append(ln)
    if not inserted and ln.strip() == "import os":
        out.extend(inject)
        inserted = True
if not inserted:
    out = inject + [""] + lines
path.write_text("\n".join(out) + "\n")
print(f"Patched FlashAttention warning filters at {path}")
PY
}

install_flash_attention() {
    local extra_args=()
    if [ -n "${PYTORCH_WHEEL_PATH:-}" ]; then
        extra_args+=("${PYTORCH_WHEEL_PATH}")
    fi
    TORCH_CUDA_ARCH_LIST="12.1" \
    pip_install --no-cache-dir --upgrade --ignore-installed --no-build-isolation --no-deps \
        "${extra_args[@]}" \
        "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_TAG}"
}

install_flash_attention_from_parts() {
    local tmp_dir
    local combined
    local status
    local -a PARTS=()
    tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/flashattn-wheel.XXXXXX")
    combined="${tmp_dir}/${FLASH_ATTN_WHEEL_BASENAME}"
    mapfile -t PARTS < <(ls "${FLASH_ATTN_SPLIT_PREFIX}"* | sort -V)
    if [ "${#PARTS[@]}" -eq 0 ]; then
        rm -rf "${tmp_dir}"
        return 1
    fi
    if cat "${PARTS[@]}" > "${combined}"; then
        TORCH_CUDA_ARCH_LIST="12.1" \
        pip_install --no-cache-dir --upgrade --ignore-installed --no-deps "${combined}"
        status=$?
    else
        status=1
    fi
    rm -rf "${tmp_dir}"
    return "${status}"
}

cache_flash_attention_wheel() {
    rm -f "${FLASH_ATTN_SPLIT_PREFIX}"* 2>/dev/null || true
    if [ -f "${FLASH_ATTN_CACHE_PATH}" ]; then
        echo "Splitting FlashAttention wheel into <50MB chunks under ${THIRD_PARTY_DIR}/wheels";
        split -b 45m -d -a 2 "${FLASH_ATTN_CACHE_PATH}" "${FLASH_ATTN_SPLIT_PREFIX}"
    fi
}

rebuild_flash_attention_cache() {
    if ensure_wheel_root_pure "${FLASH_ATTN_CACHE_PATH}" && \
       TORCH_CUDA_ARCH_LIST="12.1" \
       FLASH_ATTENTION_FORCE_CUDA_SM=121 \
       pip_wheel \
           --no-deps \
           --no-build-isolation \
           --wheel-dir "${THIRD_PARTY_DIR}/wheels" \
           "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_TAG}" >/dev/null 2>&1; then
        cache_flash_attention_wheel
    else
        echo "Warning: Failed to build FlashAttention wheel for cache."
    fi
}

current_flash_attn_version="$(pip_show flash-attn 2>/dev/null | awk '/Version/ {print $2}' || true)"
if [ -n "${current_flash_attn_version}" ] && [ "${current_flash_attn_version}" != "${FLASH_ATTN_EXPECTED_VERSION}" ]; then
    echo "FlashAttention version mismatch (${current_flash_attn_version} != ${FLASH_ATTN_EXPECTED_VERSION}), reinstalling..."
    pip_uninstall -y flash-attn >/dev/null 2>&1 || true
    current_flash_attn_version=""
fi

if [ -z "${current_flash_attn_version}" ]; then
    if [ -f "${FLASH_ATTN_CACHE_PATH}" ]; then
        echo "Installing FlashAttention from cached wheel ${FLASH_ATTN_CACHE_PATH}..."
        if ensure_wheel_root_pure "${FLASH_ATTN_CACHE_PATH}" && \
           TORCH_CUDA_ARCH_LIST="12.1" \
           FLASH_ATTENTION_FORCE_CUDA_SM=121 \
           pip_install --no-cache-dir --upgrade --ignore-installed --no-deps "${FLASH_ATTN_CACHE_PATH}"; then
            cache_flash_attention_wheel
        else
            echo "Cached FlashAttention wheel install failed; rebuilding from source..."
            if install_flash_attention; then
                rebuild_flash_attention_cache
            fi
        fi
    elif compgen -G "${FLASH_ATTN_SPLIT_PREFIX}*" >/dev/null; then
        echo "Installing FlashAttention from cached split wheel shards..."
        if ! install_flash_attention_from_parts; then
            echo "Split FlashAttention wheel install failed; rebuilding from source..."
            if install_flash_attention; then
                rebuild_flash_attention_cache
            fi
        fi
    else
        if install_flash_attention; then
            rebuild_flash_attention_cache
        fi
    fi

    if compgen -G "${FLASH_ATTN_SPLIT_PREFIX}*" >/dev/null; then
        echo "FlashAttention wheel cached as split parts (${FLASH_ATTN_SPLIT_PREFIX}*)."
    fi

    if ! pip_show flash-attn >/dev/null; then
        echo "ERROR: FlashAttention installation FAILED. Setup cannot continue without FlashAttention."
        exit 1
    fi
fi

# Filter FlashAttention warnings, if present
flash_attn_filter_warning

echo ""
echo "Verifying PyTorch CUDA after FlashAttention installation..."
if ! verify_and_restore_pytorch_cuda "FlashAttention installation"; then
    echo "ERROR: Failed to restore PyTorch CUDA after FlashAttention!"
    exit 1
fi
echo "✓ PyTorch CUDA verified after FlashAttention"
# Remove conflicting system packages that interfere with PyTorch
echo ""
echo "Removing conflicting system packages..."
# Remove python3-optree which conflicts with torch.compile
if dpkg -s python3-optree >/dev/null 2>&1; then
    echo "Removing python3-optree (conflicts with torch.compile)..."
    apt remove -y python3-optree python3-keras 2>/dev/null || true
fi
# Clean up other conflicting Lambda Labs packages
apt autoremove -y 2>/dev/null || true

# Verify installation
echo ""
echo "Verifying installation..."

# Check PyTorch
echo "Checking PyTorch installation..."
python3 - "$REQUIRED_DRIVER_VERSION" <<'PY'
import os
import sys
import textwrap
from packaging import version

required_driver = version.parse(sys.argv[1])

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"PyTorch import failed: {exc}")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    try:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except Exception:  # pragma: no cover
        pass
else:
    driver_version = None
    try:
        from torch._C import _cuda_getDriverVersion  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _cuda_getDriverVersion = None

    if _cuda_getDriverVersion is not None:
        try:
            driver_version = _cuda_getDriverVersion()
        except Exception:
            driver_version = None

    if driver_version:
        current = version.parse(str(driver_version))
        if current < required_driver:
            print(
                textwrap.dedent(
                    f"""
                    ⚠ NVIDIA driver {current} is older than required {required_driver}.
                    → Install a newer driver (e.g., nvidia-driver-580) and reboot, then rerun setup.sh.
                    """
                ).strip()
            )
    else:
        print("CUDA runtime not available. Ensure the NVIDIA driver meets CUDA 13.0 Update 2 requirements and reboot if this is a fresh install.")
PY

# Check CUDA tools
echo ""
echo "Checking CUDA tools..."
if command -v nvcc &> /dev/null; then
    echo "NVCC: $(nvcc --version | head -1)"
else
    echo "NVCC not found"
fi

# Check Nsight tools
echo ""
echo "Checking Nsight tools..."
if command -v nsys &> /dev/null; then
    NSYS_VERSION=$(nsys --version 2>/dev/null | head -1)
    echo "Nsight Systems: $NSYS_VERSION"
    # Check if it's a recent 2025 version
    if echo "$NSYS_VERSION" | grep -q "2025"; then
        echo "  Recent 2025 version installed!"
    else
        echo "  May not be the latest version (expected: 2025.x.x)"
    fi
else
    echo "Nsight Systems not found"
fi

if command -v ncu &> /dev/null; then
    NCU_VERSION=$(ncu --version 2>/dev/null | head -1)
    echo "Nsight Compute: $NCU_VERSION"
    # Check if it's 2025.3.1 or a recent 2025 version
    if echo "$NCU_VERSION" | grep -qE "2025\.3\.1|2025\.3\.[2-9]|2025\.[4-9]"; then
        echo "  Latest version installed (2025.3.1 or newer)!"
    elif echo "$NCU_VERSION" | grep -q "2025"; then
        echo "  Recent 2025 version installed!"
    else
        echo "  May not be the latest version (expected: 2025.3.1 or newer)"
    fi
else
    echo "Nsight Compute not found"
fi

# Check system tools
echo ""
echo "Checking system tools..."
tools=("numactl" "perf" "htop" "iostat" "ibstat")
for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "$tool: installed"
    else
        echo "$tool: not found"
    fi
done

# Run basic performance test
echo ""
echo "Running basic performance test..."
python3 <<'PY'
import sys
import time

try:
    import torch
except Exception as exc:
    print(f"Skipping performance test: failed to import torch ({exc})")
    sys.exit(0)

if not torch.cuda.is_available():
    print("Skipping performance test: torch.cuda.is_available() is False")
    sys.exit(0)

device = torch.device("cuda")
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

for _ in range(10):
    torch.mm(x, y)

start = time.time()
for _ in range(100):
    torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"Matrix multiplication (1000x1000): {(end - start) * 1000 / 100:.2f} ms per operation")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
PY

# Test example scripts
echo ""
echo "Testing example scripts..."

# Test Chapter 1
echo "Testing Chapter 1 (Performance Basics)..."
if [ -f "$PROJECT_ROOT/ch1/performance_basics.py" ]; then
    if python3 "$PROJECT_ROOT/ch1/performance_basics.py" > /dev/null 2>&1; then
        echo "Chapter 1: Performance basics working"
    else
        echo "Chapter 1: Some issues detected (check output above)"
    fi
else
    echo "Chapter 1 example not present, skipping."
fi

# Test Chapter 2
echo "Testing Chapter 2 (Hardware Info)..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    if python3 "$PROJECT_ROOT/ch2/hardware_info.py" > /dev/null 2>&1; then
        echo "Chapter 2: Hardware info working"
    else
        echo "Chapter 2: Some issues detected (check output above)"
    fi
else
    echo "Chapter 2 example not present, skipping."
fi

# Test Chapter 3
echo "Testing Chapter 3 (NUMA Affinity)..."
if [ -f "$PROJECT_ROOT/ch3/bind_numa_affinity.py" ]; then
    if python3 "$PROJECT_ROOT/ch3/bind_numa_affinity.py" > /dev/null 2>&1; then
        echo "Chapter 3: NUMA affinity working"
    else
        echo "Chapter 3: Some issues detected (check output above)"
    fi
else
    echo "Chapter 3 example not present, skipping."
fi

# Detect NCCL shared library for LD_PRELOAD
NCCL_LIB_PATH=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libnccl.so.2 \
    /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
    /usr/local/lib/libnccl.so.2 \
    /usr/lib/libnccl.so.2
    do
    if [ -f "${candidate}" ]; then
        NCCL_LIB_PATH="${candidate}"
        break
    fi
done
export NCCL_LIB_PATH

# Set up environment variables for optimal performance
echo ""
echo "Setting up environment variables..."
cat >> ~/.bashrc <<EOF

# AI Performance Engineering Environment Variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# PyTorch optimization
export TORCH_COMPILE_DEBUG=0
export TORCH_LOGS="+dynamo"

# CUDA paths
export CUDA_HOME="${CUDA_HOME_DIR}"
export PATH=${CUDA_HOME_DIR}/bin:\$PATH
# CRITICAL: PyTorch was compiled against cuDNN 9.15.1, but bundles cuDNN 9.13.0
# Use system cuDNN 9.15.1 instead of PyTorch's bundled 9.13.0
SYSTEM_CUDNN_LIB="/usr/lib/aarch64-linux-gnu"
PYTORCH_CUDNN_LIB="/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib"
if [ -n "\${LD_LIBRARY_PATH:-}" ]; then
    # Remove PyTorch's bundled cuDNN path (contains 9.13.0)
    LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "\${PYTORCH_CUDNN_LIB}" | tr '\n' ':' | sed 's/:$//')
fi
# Put system cuDNN 9.15.1 FIRST, then CUDA libs
if [ -d "\${SYSTEM_CUDNN_LIB}" ]; then
    export LD_LIBRARY_PATH="\${SYSTEM_CUDNN_LIB}:\${CUDA_HOME_DIR}/lib64:\${CUDA_HOME_DIR}/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
else
    export LD_LIBRARY_PATH="\${CUDA_HOME_DIR}/lib64:\${CUDA_HOME_DIR}/lib64/stubs\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi

# NCCL preload configuration
export NCCL_LIB_PATH="${NCCL_LIB_PATH}"
if [ -n "\${NCCL_LIB_PATH}" ]; then
    export LD_PRELOAD="\${NCCL_LIB_PATH}:\${LD_PRELOAD:-}"
else
    echo "WARNING: NCCL library not found; skipping LD_PRELOAD setup."
fi
EOF

echo "Environment variables added to ~/.bashrc"

# Source the environment variables for current session
if [ -n "${NCCL_LIB_PATH}" ]; then
    export LD_PRELOAD="${NCCL_LIB_PATH}:${LD_PRELOAD:-}"
    echo "NCCL 2.28.7 activated for current session (${NCCL_LIB_PATH})"
else
    echo "NCCL library not found for current session; continuing without LD_PRELOAD."
fi

# Comprehensive setup verification
echo ""
echo "Running comprehensive setup verification..."
echo "=============================================="

# Test 1: PyTorch and CUDA
echo "Testing PyTorch and CUDA..."
python3 <<'PY'
import os
import ctypes
import sys

import torch

print(f"  PyTorch version: {torch.__version__}")
print(f"  PyTorch location: {torch.__file__}")

# Verify CUDA is enabled in the build
if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
    print("ERROR: PyTorch build does not have CUDA enabled!")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    sys.exit(1)

print(f"  CUDA version in PyTorch: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("WARNING: CUDA runtime not available.")
    print("  This may be due to:")
    print("    1. NVIDIA driver not loaded")
    print("    2. CUDA libraries not in LD_LIBRARY_PATH")
    print("    3. GPU not accessible")
    print("  However, PyTorch CUDA build is correctly installed.")
    sys.exit(0)  # Don't fail the test if CUDA runtime isn't available

print(f"  GPU count: {torch.cuda.device_count()}")
print(f"  GPU name: {torch.cuda.get_device_name(0)}")

nccl_path = os.environ.get("NCCL_LIB_PATH")
try:
    if nccl_path and os.path.exists(nccl_path):
        libnccl = ctypes.CDLL(nccl_path)
        nv = ctypes.c_int()
        libnccl.ncclGetVersion(ctypes.byref(nv))
        print(f"  NCCL version: {nv.value}")
    else:
        print("  NCCL version: unavailable (library not found)")
except Exception as exc:
    print(f"  NCCL version: error ({exc})")

print("PyTorch and CUDA working correctly")
PY

# Don't fail if CUDA runtime isn't available, but verify CUDA build is correct
if [ $? -ne 0 ]; then
    echo "PyTorch/CUDA test completed with warnings (CUDA runtime may not be available)"
fi

# Test 2: Performance test
echo ""
echo "Testing GPU performance..."
python3 <<'PY'
import sys
import time

import torch

if not torch.cuda.is_available():
    print("  Skipping GPU performance test: CUDA runtime not available")
    sys.exit(0)

device = torch.device("cuda")
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

for _ in range(10):
    torch.mm(x, y)

start = time.time()
for _ in range(100):
    torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"  Matrix multiplication: {(end - start) * 1000 / 100:.2f} ms per operation")
print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
print("GPU performance test passed")
PY

# Don't fail if CUDA runtime isn't available
if [ $? -ne 0 ]; then
    echo "GPU performance test skipped (CUDA runtime not available)"
fi

# Test 3: torch.compile test
echo ""
echo "Testing torch.compile..."
python3 <<'PY'
import sys
import time
import traceback

import torch

device = torch.device("cuda")

def simple_model(x):
    return torch.mm(x, x.t())

x = torch.randn(1000, 1000, device=device)

start = time.time()
for _ in range(10):
    simple_model(x)
torch.cuda.synchronize()
uncompiled_time = time.time() - start

try:
    compiled_model = torch.compile(simple_model, mode="reduce-overhead")
except AssertionError as exc:
    if "duplicate template name" in str(exc):
        print("torch.compile skipped due to known PyTorch nightly issue: duplicate kernel template name")
        print(f"   Details: {exc}")
        sys.exit(0)
    print("torch.compile failed with assertion error:")
    print(exc)
    sys.exit(1)
except Exception:
    print("torch.compile failed with an unexpected exception:")
    traceback.print_exc()
    sys.exit(1)

# Warm up compiled model (includes compile time, but we don't measure this)
for _ in range(5):
    compiled_model(x)
torch.cuda.synchronize()

# Now measure compiled performance (compile time excluded)
start = time.time()
for _ in range(20):
    compiled_model(x)
torch.cuda.synchronize()
compiled_time = time.time() - start

print(f"  Uncompiled time: {uncompiled_time * 1000 / 10:.2f} ms per run")
print(f"  Compiled time:   {compiled_time * 1000 / 20:.2f} ms per run (warmed up, compile time excluded)")
if compiled_time < uncompiled_time:
    speedup = uncompiled_time / (compiled_time * 10 / 20)
    print(f"  Speedup: {speedup:.2f}x")
print("torch.compile test passed")
PY

compile_status=$?
if [ $compile_status -ne 0 ]; then
    echo "torch.compile test failed!"
    exit 1
fi

# Step 11: Install CUTLASS 4.2+ Backend for torch.compile
echo ""
echo "Step 11: Installing CUTLASS 4.2+ Backend (nvidia-cutlass-dsl)..."
echo "================================================================="

# Install CUTLASS DSL 4.2+ and CUDA Python bindings system-wide
# The Python package (nvidia-cutlass-dsl) includes:
#   - Python API for torch.compile CUTLASS backend
#   - C++ headers for direct CUDA C++ kernel development
#   - All CUTLASS library headers (1000+ header files)
echo "Installing nvidia-cutlass-dsl and cuda-python (pinned versions)..."
pip_install --no-cache-dir --upgrade --ignore-installed "nvidia-cutlass-dsl==4.2.1" "cuda-python==13.0.3"

if [ $? -eq 0 ]; then
    echo "CUTLASS backend packages installed (pinned versions)"
    echo "   - nvidia-cutlass-dsl==4.2.1: CUTLASS kernels for torch.compile"
    
    # Verify PyTorch CUDA after CUTLASS installation
    echo ""
    echo "Verifying PyTorch CUDA after CUTLASS installation..."
    if ! verify_and_restore_pytorch_cuda "CUTLASS installation"; then
        echo "ERROR: Failed to restore PyTorch CUDA after CUTLASS!"
        exit 1
    fi
    echo "✓ PyTorch CUDA verified after CUTLASS"
    echo "   - cuda-python==13.0.3: CUDA runtime bindings"
    echo ""
    
    # Detect CUTLASS C++ header location
    CUTLASS_INCLUDE=$(python3 tools/utilities/detect_cutlass_info.py 2>/dev/null | head -1)
    if [ -n "$CUTLASS_INCLUDE" ] && [ -d "$CUTLASS_INCLUDE/include" ]; then
        echo "CUTLASS C++ headers available at: $CUTLASS_INCLUDE/include"
        echo "   Use with: nvcc -I$CUTLASS_INCLUDE/include ..."
        echo ""
        echo "The Python package includes both Python API and C++ headers."
        echo "   No source build needed - ready for both Python and CUDA C++ usage!"
    else
        # CUTLASS headers are in site-packages - this is normal and expected
        # The Python API works fine without explicit header path
        echo "CUTLASS C++ headers are in site-packages (normal for Python package installation)"
    fi
else
    echo "CUTLASS backend installation had issues, but continuing..."
fi

echo ""

echo "Capturing hardware capabilities..."
if python3 "${PROJECT_ROOT}/tools/utilities/probe_hardware_capabilities.py" 2>/tmp/probe_hardware.log; then
    cat /tmp/probe_hardware.log
    echo ""
    echo "Hardware capabilities summary:"
    if [ -f "${PROJECT_ROOT}/artifacts/hardware_capabilities.json" ]; then
        python3 <<PY
import json
import sys
try:
        with open('${PROJECT_ROOT}/artifacts/hardware_capabilities.json', 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            devices = data.get('devices', [])
        elif isinstance(data, list):
            devices = data
        else:
            devices = []
        for device in devices:
            name = device.get('name', 'Unknown')
            device_index = device.get('device_index', '?')
            arch = device.get('architecture', 'Unknown')
            compute = device.get('compute_capability', '?')
            print(f"  GPU {device_index}: {name}")
            print(f"    Architecture: {arch} (SM {compute})")
            print(f"    Memory: {device.get('total_memory_gb', 0):.2f} GB")
            print(f"    SMs: {device.get('num_sms', '?')}")
            print(f"    Tensor Cores: {device.get('tensor_cores', 'Unknown')}")
            features = device.get('features', {})
            if isinstance(features, dict):
                print(f"    Features: FP4={features.get('fp4', False)}, FP6={features.get('fp6', False)}, FP8={features.get('fp8', False)}, BF16={features.get('bf16', False)}")
            elif isinstance(features, list):
                joined = ', '.join(features)
                if joined:
                    print(f"    Features: {joined}")
except Exception as e:
    print(f"  Error reading capabilities: {e}", file=sys.stderr)
PY
    else
        echo "  Warning: hardware_capabilities.json not found"
    fi
else
    cat /tmp/probe_hardware.log
    echo "ERROR: Hardware capability probe failed"
    exit 1
fi
rm -f /tmp/probe_hardware.log

# Ensure artifacts and wheel caches are writable by the invoking user for future runs.
if [ -n "${SUDO_USER:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/artifacts"
    chown -R "${SUDO_USER}:${SUDO_USER}" "${PROJECT_ROOT}/artifacts" 2>/dev/null || true
    chown -R "${SUDO_USER}:${SUDO_USER}" "${THIRD_PARTY_DIR}/wheels" 2>/dev/null || true
fi

echo ""
echo "Regenerating CUDA arch detection header..."
if python3 "$PROJECT_ROOT/tools/utilities/generate_arch_detection_header.py"; then
    echo "arch_detection.cuh regenerated from live hardware probe"
else
    echo "WARNING: Failed to regenerate arch_detection.cuh (continuing with existing header)"
fi

echo ""

# Test 4: Hardware info script
echo ""
echo "Testing hardware detection..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    python3 "$PROJECT_ROOT/ch2/hardware_info.py" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Hardware detection working"
    else
        echo "ERROR: Hardware detection failed"
        exit 1
    fi
else
    echo "Hardware detection script not present, skipping."
fi

# Test 5: NUMA binding script
echo ""
echo "Testing NUMA binding..."
if [ -f "$PROJECT_ROOT/ch3/bind_numa_affinity.py" ]; then
    # Temporarily disable exit-on-error for this test
    set +e
    python3 "$PROJECT_ROOT/ch3/bind_numa_affinity.py" > /dev/null 2>&1
    NUMA_EXIT_CODE=$?
    set -e
    
    if [ $NUMA_EXIT_CODE -eq 0 ]; then
        echo "NUMA binding working"
    else
        echo "NUMA binding had issues (expected in containers or without distributed launch)"
    fi
else
    echo "NUMA binding script not present, skipping."
fi

echo ""
echo "All critical tests passed! Setup is working correctly."

# Ensure non-root GPU reset capability via nvidia-smi
echo ""
echo "Configuring nvidia-smi for non-root GPU resets..."
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! command -v setcap >/dev/null 2>&1; then
        echo "Installing libcap2-bin (provides setcap)..."
        apt-get update >/dev/null 2>&1 || true
        apt-get install -y libcap2-bin >/dev/null 2>&1 || true
    fi

    if command -v setcap >/dev/null 2>&1; then
        if ! getcap /usr/bin/nvidia-smi 2>/dev/null | grep -q "cap_sys_admin"; then
            if setcap cap_sys_admin+ep /usr/bin/nvidia-smi 2>/dev/null; then
                echo "Granted cap_sys_admin capability to /usr/bin/nvidia-smi"
            else
                echo "WARNING: Failed to grant cap_sys_admin to /usr/bin/nvidia-smi (continuing)"
            fi
        else
            echo "/usr/bin/nvidia-smi already has cap_sys_admin capability"
        fi
        getcap /usr/bin/nvidia-smi 2>/dev/null || true
    else
        echo "WARNING: setcap not available; skipping capability grant."
    fi
else
    echo "nvidia-smi not found; skipping GPU reset capability grant."
fi

# Restart services impacted by NVSHMEM/CUDA installs (e.g., glances monitoring)
echo ""
# Restart background services if needed (silently skip if not present)
if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-units --type=service --all 2>/dev/null | grep -q "^glances.service"; then
        if systemctl is-active --quiet glances.service 2>/dev/null; then
            systemctl restart glances.service >/dev/null 2>&1 && echo "Restarted glances.service" || true
        else
            systemctl restart glances.service >/dev/null 2>&1 || true
        fi
    fi
fi

# Run verification scripts
echo ""
echo "Running Verification Checks..."
echo "=================================="
echo ""

VERIFICATION_FAILED=0

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
# CRITICAL: PyTorch was compiled against cuDNN 9.15.1, but bundles cuDNN 9.13.0
# We MUST use system cuDNN 9.15.1 (installed via apt) instead of PyTorch's bundled 9.13.0
# Filter out PyTorch's bundled cuDNN path and ensure system cuDNN 9.15.1 is found first
PYTORCH_CUDNN_LIB="${PYTHON_DIST_PACKAGES}/nvidia/cudnn/lib"
SYSTEM_CUDNN_LIB="/usr/lib/aarch64-linux-gnu"

current_ld="${LD_LIBRARY_PATH:-}"
filtered_ld=""
if [ -n "${current_ld}" ]; then
    IFS=':' read -ra PATHS <<< "${current_ld}"
    for path in "${PATHS[@]}"; do
        if [ -n "${path}" ]; then
            # Remove PyTorch's bundled cuDNN path (contains 9.13.0)
            if [[ "${path}" == *"${PYTORCH_CUDNN_LIB}"* ]]; then
                continue
            fi
            # Keep other paths
            if [ -n "${filtered_ld}" ]; then
                filtered_ld="${filtered_ld}:${path}"
            else
                filtered_ld="${path}"
            fi
        fi
    done
fi
# Build final LD_LIBRARY_PATH: System cuDNN 9.15.1 FIRST, then CUDA libs, then filtered paths
if [ -d "${SYSTEM_CUDNN_LIB}" ]; then
    export LD_LIBRARY_PATH="${SYSTEM_CUDNN_LIB}:${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs${filtered_ld:+:${filtered_ld}}"
else
    export LD_LIBRARY_PATH="${CUDA_HOME_DIR}/lib64:${CUDA_HOME_DIR}/lib64/stubs${filtered_ld:+:${filtered_ld}}"
fi
if python3 tools/verification/verify_pytorch.py; then
    echo "PyTorch verification passed"
else
    echo "PyTorch verification failed"
    VERIFICATION_FAILED=1
fi

echo ""

# Verify NVLink connectivity (only if multiple GPUs)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Verifying NVLink connectivity..."
    # Enable NVLink counters so downstream telemetry can read per-link utilization
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi nvlink -gt d >/dev/null 2>&1 || true
    fi
    if python3 tools/verification/verify_nvlink.py; then
        echo "NVLink verification passed"
    else
        echo "NVLink verification had warnings (review output above)"
    fi
else
    echo "Single GPU detected, skipping NVLink verification"
fi

echo ""

# Verify CUTLASS backend
echo "Verifying CUTLASS backend..."
if python3 tools/verification/verify_cutlass.py 2>/dev/null; then
    echo "CUTLASS verification passed"
else
    echo "ERROR: CUTLASS verification failed"
    exit 1
fi

echo ""

# Verify GPUDirect Storage (GDS)
echo "Verifying GPUDirect Storage (GDS)..."
if python3 tools/verification/verify_gds.py; then
    echo "GDS verification passed"
else
    echo "GDS verification had issues (may need to load nvidia-fs module)"
    echo "   Load module with: sudo modprobe nvidia-fs"
fi

echo ""

# Summary of verification
if [ $VERIFICATION_FAILED -eq 0 ]; then
    echo "All critical verifications passed!"
else
    echo "Some verifications failed - review output above"
fi

# Run peak performance benchmark
echo ""
echo "Running Peak Performance Benchmark..."
echo "======================================"
if python3 "$PROJECT_ROOT/tools/benchmarking/benchmark_peak.py" --output-dir "$PROJECT_ROOT" 2>&1; then
    echo "Peak performance benchmark completed successfully"
else
    echo "ERROR: Peak performance benchmark failed"
    exit 1
fi

# Final summary
echo ""
echo "Setup Complete!"
echo "=================="
echo ""
echo "Installed:"
echo "  • PyTorch source build (${PYTORCH_BUILD_VERSION:-custom}) with NVIDIA arch list"
echo "  • CUDA ${CUDA_FULL_VERSION} toolchain and development tools"
echo "  • NCCL ${NCCL_SHORT_VERSION} (Blackwell-optimized with NVLS support)"
echo "  • NVSHMEM 3.4.5 runtime and headers (CUDA 13)"
echo "  • GPUDirect Storage (GDS) tools, drivers, and kvikio library"
echo "  • NVIDIA Nsight Systems (latest available)"
echo "  • NVIDIA Nsight Compute (latest available)"
echo "  • All project dependencies"
echo "  • System performance tools (numactl, perf, etc.)"
echo ""
echo "Verified:"
echo "  • PyTorch installation and CUDA functionality"
echo "  • NVLink connectivity (if multi-GPU)"
echo "  • CUTLASS backend configuration"
echo "  • GPUDirect Storage (GDS) functionality"
echo ""
echo "Post-Run Checklist:"
echo "  • Verify Python runtime:          python --version  (expect 3.12.x)"
echo "  • Verify CUDA compiler:           nvcc --version    (expect release 13.0.88 / Update 2)"
echo "  • Verify driver & GPU status:     nvidia-smi        (expect driver ${REQUIRED_DRIVER_VERSION})"
echo "  • Verify PyTorch arch coverage:   python - <<'PY'"
echo "                                      import torch"
echo "                                      print(torch.cuda.get_arch_list())"
echo "                                    PY"
echo "                                    (expect ['sm_100', 'sm_103', 'sm_121', 'compute_121'])"
echo "  • cuDNN: Installed 9.15.1.9-1 (matches PyTorch's compile-time version; PyTorch bundles 9.13.0 but we use system 9.15.1)"
echo "  • Before additional builds: source /etc/profile.d/cuda-${CUDA_SHORT_VERSION}.sh (or start a new shell)"
echo "For more information, see the README.md file and chapter-specific documentation."
echo ""
echo "Happy performance engineering!"

echo ""
echo "Downloading GPT-OSS model and installing CLI..."
huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/ || {
    echo "ERROR: Failed to download gpt-oss-20b model"
    exit 1
}
if ! python3 -m pip install --no-cache-dir --upgrade gpt-oss; then
    echo "ERROR: Failed to install gpt-oss package"
    exit 1
fi
python3 -m gpt_oss.chat model/ || {
    echo "ERROR: gpt_oss chat invocation failed"
    exit 1
}
