# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Based on https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py
This script outputs relevant system environment info
Run it with `python collect_env.py`.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import locale
import os
import re
import subprocess
import sys
from collections import namedtuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import doctr

    DOCTR_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    DOCTR_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TF_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

try:
    import torchvision

    TV_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TV_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    CV2_AVAILABLE = False

PY3 = sys.version_info >= (3, 0)


# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "doctr_version",
        "tf_version",
        "torch_version",
        "torchvision_version",
        "cv2_version",
        "os",
        "python_version",
        "is_cuda_available_tf",
        "is_cuda_available_torch",
        "cuda_runtime_version",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
    ],
)


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        enc = locale.getpreferredencoding()
        output = output.decode(enc)
        err = err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_nvidia_driver_version(run_lambda):
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]")
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def get_gpu_info(run_lambda):
    if get_platform() == "darwin":
        if TF_AVAILABLE and any(tf.config.list_physical_devices("GPU")):
            return tf.config.list_physical_devices("GPU")[0].name
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cudnn_version(run_lambda):
    """This will return a list of libcudnn.so; it's hard to tell which one is being used"""
    if get_platform() == "win32":
        cudnn_cmd = 'where /R "%CUDA_PATH%\\bin" cudnn*.dll'
    elif get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#install
        # https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installmac
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        lib = os.environ.get("CUDNN_LIBRARY")
        if lib is not None and os.path.isfile(lib):
            return os.path.realpath(lib)
        return None
    files = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files.add(fn)
    if not files:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = sorted(files)
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return "Probably one of the following:\n{}".format(result)


def get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        smi = '"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\%s"' % smi
    return smi


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda):
    return run_and_read_all(run_lambda, "wmic os get Caption | findstr /v Caption")


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "lsb_release -a", r"Description:\t(.*)")


def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"')


def get_os(run_lambda):
    platform = get_platform()

    if platform == "win32" or platform == "cygwin":
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "Mac OSX {}".format(version)

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return desc

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return desc

        return platform

    # Unknown platform
    return platform


def get_env_info():
    run_lambda = run

    doctr_str = doctr.__version__ if DOCTR_AVAILABLE else "N/A"

    if TF_AVAILABLE:
        tf_str = tf.__version__
        tf_cuda_available_str = any(tf.config.list_physical_devices("GPU"))
    else:
        tf_str = tf_cuda_available_str = "N/A"

    if TORCH_AVAILABLE:
        torch_str = torch.__version__
        torch_cuda_available_str = torch.cuda.is_available()
    else:
        torch_str = torch_cuda_available_str = "N/A"

    tv_str = torchvision.__version__ if TV_AVAILABLE else "N/A"

    cv2_str = cv2.__version__ if CV2_AVAILABLE else "N/A"

    return SystemEnv(
        doctr_version=doctr_str,
        tf_version=tf_str,
        torch_version=torch_str,
        torchvision_version=tv_str,
        cv2_version=cv2_str,
        python_version=".".join(map(str, sys.version_info[:3])),
        is_cuda_available_tf=tf_cuda_available_str,
        is_cuda_available_torch=torch_cuda_available_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        os=get_os(run_lambda),
    )


env_info_fmt = """
DocTR version: {doctr_version}
TensorFlow version: {tf_version}
PyTorch version: {torch_version} (torchvision {torchvision_version})
OpenCV version: {cv2_version}
OS: {os}
Python version: {python_version}
Is CUDA available (TensorFlow): {is_cuda_available_tf}
Is CUDA available (PyTorch): {is_cuda_available_torch}
CUDA runtime version: {cuda_runtime_version}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(envinfo.nvidia_gpu_models)

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    all_cuda_fields = dynamic_cuda_fields + ["cudnn_version"]
    all_dynamic_cuda_fields_missing = all(mutable_dict[field] is None for field in dynamic_cuda_fields)
    if TF_AVAILABLE and not any(tf.config.list_physical_devices("GPU")) and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = "No CUDA"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    """Collects environment information for debugging purposes
    Returns:
        str: environment information
    """
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...\n")
    output = get_pretty_env_info()
    print(output)


if __name__ == "__main__":
    main()
