# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake must be installed to build this package") from exc

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={cfg}")

        use_mingw = False
        mingw_bin = None

        if sys.platform == "win32":
            generator = os.environ.get("CMAKE_GENERATOR", "")
            if generator:
                cmake_args = ["-G", generator] + cmake_args
                if "mingw" in generator.lower():
                    use_mingw = True
                else:
                    cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}")
            else:
                try:
                    subprocess.check_output(["g++", "--version"], stderr=subprocess.STDOUT)
                    use_mingw = True
                    cmake_args = ["-G", "MinGW Makefiles"] + cmake_args
                    build_args = []
                except (OSError, subprocess.CalledProcessError):
                    cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}")

            if use_mingw:
                gxx_path = shutil.which("g++")
                if gxx_path:
                    mingw_bin = Path(gxx_path).parent
        else:
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        if use_mingw and mingw_bin is not None:
            runtime_libs = [
                "libstdc++-6.dll",
                "libgcc_s_seh-1.dll",
                "libwinpthread-1.dll",
            ]
            extdir_path = Path(extdir)
            extdir_path.mkdir(parents=True, exist_ok=True)
            for lib_name in runtime_libs:
                src_path = mingw_bin / lib_name
                if src_path.exists():
                    shutil.copy2(src_path, extdir_path / lib_name)
                else:
                    self.announce(
                        f"Warning: Expected MinGW runtime DLL '{lib_name}' not found next to g++ (looked in {mingw_bin}). "
                        "The built extension may fail to import if the DLL is not on PATH.",
                        level=3,
                    )


kimodo_packages = find_packages(include=["kimodo", "kimodo.*"])

# When set (e.g. in Docker), do not bundle motion_correction here; it is installed
# separately (e.g. from docker_requirements.txt as ./MotionCorrection) non-editable.
skip_motion_correction = os.environ.get("SKIP_MOTION_CORRECTION_IN_SETUP", "").strip().lower() in ("1", "true", "yes")

def _simde_header_available() -> bool:
    candidates = []
    for env_name in ("SIMDE_INCLUDE_DIR", "CONDA_PREFIX"):
        value = os.environ.get(env_name, "").strip()
        if value:
            candidates.append(Path(value))
            candidates.append(Path(value) / "include")
    cmake_prefix = os.environ.get("CMAKE_PREFIX_PATH", "").replace(";", os.pathsep)
    for entry in cmake_prefix.split(os.pathsep):
        if entry.strip():
            candidates.append(Path(entry.strip()))
            candidates.append(Path(entry.strip()) / "include")
    candidates.extend(
        [
            Path("/opt/homebrew/include"),
            Path("/usr/local/include"),
            Path("/opt/local/include"),
        ]
    )
    return any((path / "simde/x86/sse.h").exists() for path in candidates)


# The motion_correction native extension is written against x86 intrinsics. On
# arm64/aarch64 we can build it through SIMDe; if SIMDe is unavailable, keep the
# package install usable by skipping the optional native post-processing module.
_machine = platform.machine().lower()
_is_x86 = _machine in ("x86_64", "amd64", "i386", "i486", "i586", "i686")
_force_motion_correction = os.environ.get("FORCE_MOTION_CORRECTION_BUILD", "").strip().lower() in ("1", "true", "yes")
if not skip_motion_correction and not _is_x86 and not (_force_motion_correction or _simde_header_available()):
    print(
        f"Skipping motion_correction native extension: architecture '{platform.machine()}' is not x86 "
        "and SIMDe headers were not found. Install simde or set FORCE_MOTION_CORRECTION_BUILD=1 to override."
    )
    skip_motion_correction = True
elif not skip_motion_correction and not _is_x86:
    print(f"Building motion_correction for '{platform.machine()}' through SIMDe.")

if skip_motion_correction:
    packages = kimodo_packages
    package_dir = {}
    ext_modules = []
    cmdclass = {}
else:
    packages = kimodo_packages + ["motion_correction"]
    package_dir = {"motion_correction": "MotionCorrection/python/motion_correction"}
    ext_modules = [CMakeExtension("motion_correction._motion_correction", "MotionCorrection")]
    cmdclass = {"build_ext": CMakeBuild}

setup(
    packages=packages,
    package_dir=package_dir,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
