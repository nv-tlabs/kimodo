#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Setup script for correct_motion standalone package."""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build this package")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # import pdb; pdb.set_trace()  # Debug build process

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if sys.platform == "darwin":
            deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET") or sysconfig.get_config_var(
                "MACOSX_DEPLOYMENT_TARGET"
            )
            if deployment_target:
                cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}")
            archflags = os.environ.get("ARCHFLAGS", "").split()
            architectures = [archflags[i + 1] for i, token in enumerate(archflags[:-1]) if token == "-arch"]
            if len(set(architectures)) > 1:
                cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(dict.fromkeys(architectures))}")

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        use_mingw = False
        mingw_bin = None

        if sys.platform == "win32":
            generator = os.environ.get("CMAKE_GENERATOR", "")
            if generator:
                cmake_args = ["-G", generator] + cmake_args
                if "mingw" in generator.lower():
                    use_mingw = True
                else:
                    cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            else:
                # Try MinGW Makefiles as default on Windows
                try:
                    subprocess.check_output(["g++", "--version"], stderr=subprocess.STDOUT)
                    use_mingw = True
                    cmake_args = ["-G", "MinGW Makefiles"] + cmake_args
                    build_args = []  # MinGW Makefiles do not accept --config
                except (OSError, subprocess.CalledProcessError):
                    # If g++ is not found, let CMake use its default (Visual Studio)
                    cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]

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


setup(
    name="motion_correction",
    version="1.0.0",
    author="NVIDIA",
    description="Standalone Kimodo motion correction extension",
    long_description="",
    packages=["motion_correction"],
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("motion_correction._motion_correction")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10,<3.13",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23,<2",
    ],
)
