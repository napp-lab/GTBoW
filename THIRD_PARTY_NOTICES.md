# Third-Party Notices

GTBoW includes or derives from third-party software. The MIT license in the
repository root applies to the original GTBoW code; third-party components
remain subject to their own license terms.

## HD Ground

The example and test images under `data/dataset/` are adaptations of images
from the
[HD Ground database](https://github.com/JanFabianSchmid/HD_Ground), licensed
under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

See [`data/README.md`](data/README.md) for attribution, source filenames,
modifications, and the requested academic citation.

## DBoW3

The modified DBoW3 source under `thirdparty/DBow3/` is derived from
[DBoW3](https://github.com/rmsalinas/DBow3), which in turn is derived from
DBoW2. The adaptations in `src/GTDatabase.cpp`, `src/GTDatabase.h`, and
`src/GTQueryResults.h` also retain DBoW3/DBoW2 terms for their derived portions;
original GTBoW additions remain MIT. Source-level attribution has been restored
in these files.

Copyright (c) 2016 Rafael Muñoz-Salinas.

See [`thirdparty/DBow3/LICENSE.txt`](thirdparty/DBow3/LICENSE.txt) for its
redistribution conditions. The file-specific two-condition license in
`thirdparty/DBow3/src/exports.h` (copyright 2014 Rafael Muñoz Salinas) is also retained verbatim.

### DBoW2-derived portions

DBoW3 derives from [DBoW2](https://github.com/dorian3d/DBoW2). Portions of the
vendored sources retain Dorian Gálvez-López authorship.

Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com

The unmodified upstream license, including redistribution conditions,
disclaimer, and academic citation, is in
[`thirdparty/DBow3/DBOW2_LICENSE.txt`](thirdparty/DBow3/DBOW2_LICENSE.txt).
Both DBoW2 and DBoW3 require notification of the original author upon source
or binary redistribution. The repository owner must notify Dorian
Gálvez-López and Rafael Muñoz-Salinas before publication and keep a record.
These components are not relicensed under GTBoW's MIT license.

The upstream DBoW3 distribution includes QuickLZ-based binary vocabulary
compression. GTBoW's public distribution removes the QuickLZ source and all
compression and decompression calls. It supports uncompressed vocabulary files
only.

## pyDBoW3

The Python binding structure was based on
[pyDBoW3](https://github.com/Sologala/pyDBow3).

The upstream license names the Pybind Development Team (copyright 2016).
It has BSD-style redistribution conditions and an additional license grant
for enhancements made available publicly or to the original author. That
additional clause is retained; this is not described as an unmodified BSD
3-Clause license.

See [`thirdparty/PYDBOW3_LICENSE.txt`](thirdparty/PYDBOW3_LICENSE.txt).

## pybind11

pybind11 is distributed under a BSD 3-Clause license.
Copyright (c) 2016 Wenzel Jakob; upstream contributor notices are retained.

See [`thirdparty/pybind11/LICENSE`](thirdparty/pybind11/LICENSE).

## pybind11_opencv_numpy

pybind11_opencv_numpy by Erwan BERNARD is distributed under Apache License 2.0
(copyright 2016–2021), with file-specific provenance retained:

- `ndarray_converter.cpp` identifies MIT-licensed inspiration from
  [opencv-ndarray-conversion](https://github.com/yati-sagade/opencv-ndarray-conversion)
  by Yati Sagade. Its unmodified upstream license is included as
  [`NDARRAY_CONVERSION_LICENSE.txt`](thirdparty/pybind11_opencv_numpy/NDARRAY_CONVERSION_LICENSE.txt).
- That file also states that conversion functions were taken/adapted from
  OpenCV 4.5.2 `modules/python/src2/cv2.cpp`. The corresponding upstream Apache
  2.0 license is included as
  [`OPENCV_LICENSE.txt`](thirdparty/pybind11_opencv_numpy/OPENCV_LICENSE.txt).
- Comparison with upstream confirms that GTBoW's converter implementation is
  unchanged; its local modifications are attribution/license-location comments.
  `NumpyAllocator`, `toMat`, and `toNDArray` correspond to adaptations of OpenCV
  4.5.2's allocator and Mat conversion functions. Thread/GIL helpers also appear
  in Yati Sagade's source, so similarities do not establish exclusive authorship.
- The upstream MIT file label and Apache 2.0 project license do not explicitly
  delimit their respective coverage. All identified license texts and original
  notices are preserved. Exact contribution boundaries remain an upstream
  attribution question; no third-party source has been relicensed.

See
[`thirdparty/pybind11_opencv_numpy/LICENSE`](thirdparty/pybind11_opencv_numpy/LICENSE).
