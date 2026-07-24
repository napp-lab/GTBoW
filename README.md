# Improved Bag-of-Words for Ground Texture Localization

GTBoW is the official implementation of:

> **Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization**
>
> Aaron Wilhelm and Nils Napp, IEEE International Conference on Robotics and Automation (ICRA), 2025
>
> [Read the paper](https://www.csl.cornell.edu/~nnapp/papers/wilhelm_icra25.pdf)

GTBoW extends bag-of-words (BoW) image retrieval with geometric constraints
designed for ground texture localization. It provides a C++ implementation and
Python bindings through the `pyGTBoW` module.

## Overview

Conventional BoW methods can perform poorly on ground texture images, where
visual features are often less distinctive than those in general scenes. GTBoW
improves retrieval using:

- **Size binning:** Features assigned to the same visual word are distinguished
  by keypoint size.
- **Orientation-consistency verification:** Feature-orientation differences are
  incorporated into inverted-index scoring to reject inconsistent matches
  during retrieval.
- **Approximate k-means (AKM) vocabularies:** AKM provides higher-quality visual
  word assignments than the hierarchical k-means vocabularies commonly used in
  SLAM systems.
- **Soft assignment:** Each descriptor can be assigned to multiple visual words
  with confidence-based weights, reducing sensitivity to noise and
  quantization.

GTBoW can replace a generic BoW retrieval component in an existing ground
texture localization pipeline.

## Installation

### Pixi (recommended)

The included [Pixi](https://pixi.sh) environment provides the C++ and Python
dependencies and is currently configured for 64-bit Linux. After installing
Pixi, run the following from the repository root:

```bash
pixi run install
```

This creates an isolated environment and installs `pyGTBoW` in editable mode.
To verify the installation:

```bash
pixi run test
pixi run python -c "import pyGTBoW; print(pyGTBoW.__version__())"
```

### Existing environment

GTBoW is built from source. Before installing it, ensure that your environment
provides:

- Python 3.14 and NumPy
- a C++14-compatible compiler
- CMake
- OpenCV
- FLANN
- LZ4
- pybind11 and `pybind11_opencv_numpy` (vendored in `thirdparty/`)
- Ninja (optional, but recommended)

Then install the package with pip:

```bash
python -m pip install --no-build-isolation .
```

For an editable development installation, use:

```bash
python -m pip install --no-build-isolation -e .
```

Build isolation is disabled because the CMake build needs the NumPy headers
from the active environment.

Release builds disable Tracy profiling and CPU-specific `-march=native`
optimizations by default. Developers can opt in with
`CMAKE_ARGS="-DENABLE_TRACY=ON -DENABLE_NATIVE_ARCH=ON"` when installing.
Tracy-enabled builds fetch the pinned upstream Tracy version over the network.
LZ4 is required by FLANN's index persistence, independently of vocabulary
serialization.

### Supported environment

GTBoW 1.0.0 is tested and supported on 64-bit Linux with Python 3.14. Other
Python versions and platforms may work, but are not part of the current
release support policy.

## Usage

The example in [`examples/simple_example.py`](examples/simple_example.py)
illustrates how to train a size-binned vocabulary from the included ground
texture images, build a database, and perform a query. These test images are
adapted from the
[HD Ground database](https://github.com/JanFabianSchmid/HD_Ground) and are
licensed separately under CC BY-SA 4.0; see the
[data attribution](data/README.md) for details.

```bash
pixi run python examples/simple_example.py
```

A typical workflow is:

1. Extract binary descriptors, keypoint sizes, and keypoint poses from a set of
   training images.
2. Configure and train a `SizeBinnedVocabulary`.
3. Transform each image into assignments, weights, and a BoW vector.
4. Add those values to a `GTDatabase`.
5. Transform a query image and call `GTDatabase.queryBowVec`.

The pose array passed to `GTDatabase` must have one row per descriptor and three
columns: `(x, y, orientation)`. Orientations are expressed in radians.

Set `num_orientation_bins` to a value greater than one when querying to enable
orientation-consistency verification.

### Persistence

Vocabularies are saved to a directory:

```python
vocabulary.save("vocabulary")
vocabulary = SizeBinnedVocabulary.load("vocabulary", config)
```

Standard DBoW3 vocabularies use uncompressed binary files (or OpenCV YAML
when the filename contains `.yml`). Call `vocabulary.save(path)` without a
compression argument. Binary files retain the upstream DBoW3 signature and
compression flag, always saved as false, and interoperate with uncompressed
upstream DBoW3 vocabularies. Compressed DBoW3 files are rejected because GTBoW
does not distribute QuickLZ.

Geometric databases are saved to a single file and can be loaded into a new
instance:

```python
database.save("database.gtbow")

loaded_database = GTDatabase()
loaded_database.load("database.gtbow")
```

## Testing

With Pixi:

```bash
pixi run test
```

In an environment where `pyGTBoW` and pytest are already installed:

```bash
python -m pytest tests/
```

## Related projects

GTBoW builds on or incorporates code from the following open-source projects:

- [DBoW3](https://github.com/rmsalinas/DBow3), which provides the core BoW
  implementation extended by this project
- [pyDBoW3](https://github.com/Sologala/pyDBow3), which served as the basis for
  the Python bindings
- [pybind11](https://github.com/pybind/pybind11), used to create the Python
  extension
- [pybind11_opencv_numpy](https://github.com/edmBernard/pybind11_opencv_numpy),
  used to convert between OpenCV `Mat` and NumPy `ndarray` objects

For related research, see the
[Napp Lab website](https://www.csl.cornell.edu/~nnapp/) and the
[Ground Texture Localization project page](https://www.csl.cornell.edu/~nnapp/projects/ground_texture_slam/).

## Citation

If you use GTBoW in your research, please cite:

```bibtex
@inproceedings{wilhelm2025improved,
  title={Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization},
  author={Wilhelm, Aaron and Napp, Nils},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8020--8026},
  year={2025},
  organization={IEEE}
}
```

## License

The original GTBoW code is released under the [MIT License](LICENSE). Bundled
and derived third-party code remains subject to its respective license terms;
see [Third-Party Notices](THIRD_PARTY_NOTICES.md). The example images under
`data/dataset/` are licensed under CC BY-SA 4.0, as described in
[`data/README.md`](data/README.md).
