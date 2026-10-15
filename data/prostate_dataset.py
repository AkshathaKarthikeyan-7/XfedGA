"""
Dataset loader for prostate MRI segmentation clients (PROMISE12, Decathlon
Task05_Prostate).

Unlike the breast datasets (2D images, ready to use as-is), these are 3D
volumes that need: (1) resampling to a common in-plane physical spacing,
since raw voxel spacing varies case-to-case (PROMISE12) and dataset-to-
dataset (PROMISE12 vs Decathlon); (2) 2D slice extraction along the z-axis;
(3) for Decathlon specifically, extracting only the T2 modality channel
and merging its two-region label (1=PZ, 2=TZ) into a single binary mask.

Output interface matches data/breast_dataset.py exactly:
    image: (1, H, W) float tensor, normalized to [0, 1]
    mask:  (H, W) long tensor, binary {0, 1}
"""
import os
import glob
import numpy as np
import torch
import cv2
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import zoom as ndi_zoom
from torch.utils.data import Dataset

from data.cache import DEFAULT_CACHE_ROOT, build_or_load, file_fingerprint, source_hash

TARGET_SPACING_MM = 0.5


def _resample_to_spacing_sitk(sitk_image, is_mask=False):
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    new_spacing = (TARGET_SPACING_MM, TARGET_SPACING_MM, original_spacing[2])
    new_size = [
        int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
        original_size[2],
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resampler.Execute(sitk_image)


def _resample_to_spacing_nib(volume, zooms_xy):
    zoom_factor_x = zooms_xy[0] / TARGET_SPACING_MM
    zoom_factor_y = zooms_xy[1] / TARGET_SPACING_MM
    return ndi_zoom(volume, (zoom_factor_x, zoom_factor_y, 1.0), order=1)


def _resample_to_spacing_nib_nearest(volume, zooms_xy):
    zoom_factor_x = zooms_xy[0] / TARGET_SPACING_MM
    zoom_factor_y = zooms_xy[1] / TARGET_SPACING_MM
    return ndi_zoom(volume, (zoom_factor_x, zoom_factor_y, 1.0), order=0)


def _normalize_and_resize(img_slice, mask_slice, size=(384, 384)):
    """1st/99th percentile clipping BEFORE min-max scaling -- raw MRI
    intensity has a small fraction of extreme-bright outlier voxels (edge/
    interpolation artifacts, bright fat signal) that, under plain min-max,
    compress the real diagnostic tissue contrast into a narrow, harshly
    stretched band. Percentile clipping keeps that contrast usable."""
    img_slice = img_slice.astype(np.float32)
    lo, hi = np.percentile(img_slice, 1), np.percentile(img_slice, 99)
    img_slice = np.clip(img_slice, lo, hi)

    img_slice = cv2.resize(img_slice, size, interpolation=cv2.INTER_LINEAR)
    mask_slice = cv2.resize(mask_slice.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)

    denom = img_slice.max() - img_slice.min()
    img_slice = (img_slice - img_slice.min()) / denom if denom > 0 else img_slice
    mask_slice = (mask_slice > 0).astype(np.float32)

    return img_slice, mask_slice


def _index_promise12(root):
    case_paths = sorted(glob.glob(os.path.join(root, "Case[0-9][0-9].mhd")))
    return [(p, "promise12") for p in case_paths]


def _index_decathlon(root):
    images_dir = os.path.join(root, "imagesTr")
    image_paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, "*.nii.gz"))
        if not os.path.basename(p).startswith("._")
    )
    return [(p, "decathlon") for p in image_paths]


def _extract_slices_promise12(volume_mhd_path, min_fg_fraction=0.0):
    seg_path = volume_mhd_path.replace(".mhd", "_segmentation.mhd")
    img_sitk = sitk.ReadImage(volume_mhd_path)
    mask_sitk = sitk.ReadImage(seg_path)

    img_sitk = _resample_to_spacing_sitk(img_sitk, is_mask=False)
    mask_sitk = _resample_to_spacing_sitk(mask_sitk, is_mask=True)

    img_vol = sitk.GetArrayFromImage(img_sitk)
    mask_vol = sitk.GetArrayFromImage(mask_sitk)

    slices = []
    for z in range(img_vol.shape[0]):
        img_slice, mask_slice = img_vol[z], mask_vol[z]
        fg_frac = (mask_slice > 0).sum() / mask_slice.size
        if fg_frac > min_fg_fraction:
            slices.append((img_slice, mask_slice))
    return slices


def _extract_slices_decathlon(volume_nii_path, min_fg_fraction=0.0):
    label_path = volume_nii_path.replace("imagesTr", "labelsTr")

    img_nii = nib.load(volume_nii_path)
    label_nii = nib.load(label_path)

    img_vol_4d = img_nii.get_fdata()
    t2_vol = img_vol_4d[..., 0]
    label_vol = label_nii.get_fdata()

    zooms = img_nii.header.get_zooms()[:2]
    t2_vol = _resample_to_spacing_nib(t2_vol, zooms)
    label_vol = _resample_to_spacing_nib_nearest(label_vol, zooms)

    slices = []
    for z in range(t2_vol.shape[2]):
        img_slice = t2_vol[:, :, z]
        mask_slice = (label_vol[:, :, z] > 0).astype(np.float32)
        fg_frac = mask_slice.sum() / mask_slice.size
        if fg_frac > min_fg_fraction:
            slices.append((img_slice, mask_slice))
    return slices


def split_promise12_clients(root, seed=42):
    all_cases = _index_promise12(root)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_cases))
    half = len(all_cases) // 2
    client1 = [all_cases[i] for i in idx[:half]]
    client2 = [all_cases[i] for i in idx[half:]]
    return client1, client2


INDEXERS = {"PROMISE12": _index_promise12, "Decathlon": _index_decathlon}
EXTRACTORS = {"promise12": _extract_slices_promise12, "decathlon": _extract_slices_decathlon}


def _case_files(case_path, dataset_tag):
    if dataset_tag == "promise12":
        seg = case_path.replace(".mhd", "_segmentation.mhd")
        return [case_path, case_path[:-4] + ".raw", seg, seg[:-4] + ".raw"]
    return [case_path, case_path.replace("imagesTr", "labelsTr")]


def prostate_cache(client_name, root, size=(384, 384), min_fg_fraction=0.0, cache_root=DEFAULT_CACHE_ROOT):
    """Cache of fully preprocessed float32 slices (+ binary masks) for EVERY case in `root`.
    Item id = case path; each case owns the rows of its kept slices, in slice order."""
    cases = INDEXERS[client_name](root)
    spec = {"dataset": client_name, "size": list(size), "min_fg_fraction": min_fg_fraction,
            "target_spacing_mm": TARGET_SPACING_MM, "version": 1,
            "code": source_hash(_resample_to_spacing_sitk, _resample_to_spacing_nib,
                                _resample_to_spacing_nib_nearest, _normalize_and_resize,
                                _extract_slices_promise12, _extract_slices_decathlon),
            "files": file_fingerprint([f for c, tag in cases for f in _case_files(c, tag)])}

    def preprocess(case):
        case_path, tag = case
        return [_normalize_and_resize(img, msk, tuple(size))
                for img, msk in EXTRACTORS[tag](case_path, min_fg_fraction)]

    return build_or_load(client_name, spec, [(c, (c, tag)) for c, tag in cases], preprocess,
                         cache_root=cache_root, image_dtype=np.float32)


class ProstateClientDataset(Dataset):
    """2-D slices of the given cases. With `cache_root` set (default), slices come from the
    on-disk cache of fully preprocessed slices (see prostate_cache), built on first use;
    with cache_root=None, volumes are loaded and preprocessed in memory as before."""
    INDEXERS = INDEXERS
    EXTRACTORS = EXTRACTORS

    def __init__(self, client_name, root, size=(384, 384), min_fg_fraction=0.0, cases=None,
                 cache_root=DEFAULT_CACHE_ROOT):
        self.size = size
        case_list = cases if cases is not None else self.INDEXERS[client_name](root)
        if len(case_list) == 0:
            raise RuntimeError(f"No cases found for {client_name} at {root}.")

        self.cache = prostate_cache(client_name, root, size, min_fg_fraction, cache_root) if cache_root else None
        self.slices = []
        if self.cache is not None:
            for case_path, _ in case_list:
                self.slices.extend(self.cache.rows(case_path))
        else:
            for case_path, dataset_tag in case_list:
                extractor = self.EXTRACTORS[dataset_tag]
                self.slices.extend(extractor(case_path, min_fg_fraction))

        if len(self.slices) == 0:
            raise RuntimeError(
                f"No non-empty slices extracted for {client_name} at {root}. "
                f"Check min_fg_fraction or resampling parameters."
            )
        print(f"  ProstateClientDataset[{client_name}]: {len(self.slices)} slices "
              f"from {len(case_list)} cases")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        if self.cache is not None:
            row = self.slices[idx]
            img, mask = self.cache.image(row), self.cache.mask(row)
        else:
            img_slice, mask_slice = self.slices[idx]
            img, mask = _normalize_and_resize(img_slice, mask_slice, self.size)
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(np.asarray(mask)).long()
        return img_t, mask_t
