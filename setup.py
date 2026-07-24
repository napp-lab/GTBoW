import os
import re
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)
                    ) if self.debug is None else self.debug
        cfg = "Release" if debug else "Release"
        # print("THIS IS THE CONFIGURATION")
        # print(cfg)
        # import pdb; pdb.set_trace()
        cfg = "Release"
        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        # Inject Conda/Pixi prefix for headers and libs
        conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
        if conda_prefix:
            include_dir = os.path.join(conda_prefix, "include")
            lib_dir = os.path.join(conda_prefix, "lib")
            
            # Collect all include dirs needed for the build
            extra_includes = [f"-I{include_dir}"]
            
            # Add numpy include directory.
            # pip build isolation hides numpy from the build env, so we
            # cannot simply `import numpy`.  Instead, locate the headers
            # directly inside the conda/pixi prefix.
            numpy_inc = None
            # Method 1: ask the *real* environment Python (not the isolated one)
            real_python = os.path.join(conda_prefix, "bin", "python")
            if not os.path.exists(real_python):
                real_python = os.path.join(conda_prefix, "bin", "python3")
            if os.path.exists(real_python):
                try:
                    numpy_inc = subprocess.check_output(
                        [real_python, "-c", "import numpy; print(numpy.get_include())"],
                        text=True,
                    ).strip()
                except subprocess.CalledProcessError:
                    pass
            # Method 2: glob for it in site-packages
            if not numpy_inc:
                import glob
                candidates = glob.glob(
                    os.path.join(conda_prefix, "lib", "python*", "site-packages", "numpy", "_core", "include")
                ) + glob.glob(
                    os.path.join(conda_prefix, "lib", "python*", "site-packages", "numpy", "core", "include")
                )
                for c in candidates:
                    if os.path.isdir(c):
                        numpy_inc = c
                        break
            if numpy_inc:
                extra_includes.append(f"-I{numpy_inc}")
            
            cxx_flags = " ".join(extra_includes)
            cmake_args += [
                f"-DCMAKE_PREFIX_PATH={conda_prefix}",
                f"-DCMAKE_CXX_FLAGS={cxx_flags}",
                f"-DCMAKE_SHARED_LINKER_FLAGS=-L{lib_dir}",
            ]
            
            lz4_path = os.path.join(lib_dir, "liblz4.so")
            if not os.path.exists(lz4_path):
                lz4_path = os.path.join(lib_dir, "liblz4.dylib") # macOS
            if os.path.exists(lz4_path):
                cmake_args += [f"-DLZ4_LIBRARIES={lz4_path}"]

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(
                x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="pyGTBoW",
    version="1.0.0",
    author="Aaron",
    author_email="ajw344@cornell.edu",
    description="A BoW library for ground texture image retrieval",
    long_description="",
    ext_modules=[CMakeExtension("pyGTBoW")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    license_files=[
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "thirdparty/PYDBOW3_LICENSE.txt",
        "thirdparty/DBow3/LICENSE.txt",
        "thirdparty/DBow3/DBOW2_LICENSE.txt",
        "thirdparty/pybind11/LICENSE",
        "thirdparty/pybind11_opencv_numpy/LICENSE",
        "thirdparty/pybind11_opencv_numpy/NDARRAY_CONVERSION_LICENSE.txt",
        "thirdparty/pybind11_opencv_numpy/OPENCV_LICENSE.txt",
    ],
    extras_require={"test": ["pytest>=6.0"]},
    # The public release is currently tested on Python 3.14 only.
    python_requires=">=3.14,<3.15",
)
