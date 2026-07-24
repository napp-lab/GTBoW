# Example Image Attribution

The images in `data/dataset/` are adapted from the **HD Ground database** by
J. F. Schmid, S. F. Simon, R. Radhakrishnan, S. Frintrop, and R. Mester.

- Project: <https://github.com/JanFabianSchmid/HD_Ground>
- Source subset: `footpath_test_sq/test_path1/seq0027`
- Copyright notices: © 2021 in the downloaded `footpath_test_sq` subset and
  © 2022 in the HD Ground project README
- License:
  [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)

## Modifications

GTBoW selected 20 overlapping grayscale frames from the sequence, renamed
them, center-cropped each original from 1600×1200 to 1200×1200 pixels, resized
the crop to 512×512 pixels using area interpolation, and removed PNG metadata.
These adapted images are distributed under CC BY-SA 4.0.

| GTBoW file | Sequence time | Original HD Ground filename |
| --- | ---: | --- |
| `img00.png` | 0.00 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000160.png` |
| `img01.png` | 0.16 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000164.png` |
| `img02.png` | 0.32 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000168.png` |
| `img03.png` | 0.48 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000172.png` |
| `img04.png` | 0.64 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000176.png` |
| `img05.png` | 0.80 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000180.png` |
| `img06.png` | 0.96 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000184.png` |
| `img07.png` | 1.12 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000188.png` |
| `img08.png` | 1.28 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000192.png` |
| `img09.png` | 1.44 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000196.png` |
| `img10.png` | 1.60 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000200.png` |
| `img11.png` | 1.76 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000204.png` |
| `img12.png` | 1.92 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000208.png` |
| `img13.png` | 2.08 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000212.png` |
| `img14.png` | 2.24 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000216.png` |
| `img15.png` | 2.40 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000220.png` |
| `img16.png` | 2.56 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000224.png` |
| `img17.png` | 2.72 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000228.png` |
| `img18.png` | 2.88 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000232.png` |
| `img19.png` | 3.04 s | `HDG1_t001_train_2021-01-20_s0027_c01_i0000236.png` |

If you use these images in academic work, please also cite the HD Ground
publication:

```bibtex
@ARTICLE{HD_Ground_Schmid,
  author={J. F. Schmid and S. F. Simon and R. Radhakrishnan and S. Frintrop and R. Mester},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  title={{HD Ground} - A Database for Ground Texture Based Localization},
  year={2022},
  month={May},
  pages={7628--7634}
}
```
