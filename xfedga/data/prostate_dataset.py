"""
Dataset loader for prostate MRI segmentation clients (PROMISE12, Decathlon
Task05_Prostate).

Unlike the breast datasets (2D images, ready to use as-is), these are 3D
volumes that need: (1) resampling to a common in-plane physical spacing,
since raw voxel spacing varies case-to-case (PROMISE12) and dataset-to-
dataset (PROMISE12 vs Decathlon) -- without this, a "prostate-sized" region
would occupy a different number of pixels in different cases/datasets,
which would corrupt training; (2) 2D slice extraction along the z-axis;
(3) for Decathlon specifically, extracting only the T2 modality channel
(channel 0 of the 4D volume) and merging its two-region label (1=PZ,
2=TZ) into a single binary prostate mask, to match PROMISE12's binary
labeling and the breast pipeline's binary segmentation setup.

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

# Common in-plane physical spacing (mm) every slice is resampled to before
# resizing to the network's input size. 0.5mm is a reasonable middle ground
# between PROMISE12's finest (~0.27mm) and coarsest (~0.75mm) in-plane
# resolution -- adjust here if you find it needs tuning after inspecting
# a few resampled slices visually.
TARGET_SPACING_MM = 0.5


def _resample_to_spacing_sitk(sitk_image, is_mask=False):
    """Resample a SimpleITK image (PROMISE12 .mhd/.raw) to TARGET_SPACING_MM
    in-plane, keeping the through-plane (z) spacing untouched since we only
    need consistent in-plane anatomy per 2D slice, not through-plane."""
    original_spacing = sitk_image.GetSpacing()  # (x, y, z)
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
    """Resample a numpy volume (H, W, slices) from Decathlon/NIfTI, whose
    in-plane zoom (mm/voxel) is given by zooms_xy, to TARGET_SPACING_MM."""
    zoom_factor_x = zooms_xy[0] / TARGET_SPACING_MM
    zoom_factor_y = zooms_xy[1] / TARGET_SPACING_MM
    return ndi_zoom(volume, (zoom_factor_x, zoom_factor_y, 1.0), order=1)

def _resample_to_spacing_nib_nearest(volume, zooms_xy):
    """Same as _resample_to_spacing_nib but nearest-neighbor (order=0),
    for label/mask volumes -- prevents inventing fractional label values
    (e.g. 1.4) at region boundaries during resampling."""
    zoom_factor_x = zooms_xy[0] / TARGET_SPACING_MM
    zoom_factor_y = zooms_xy[1] / TARGET_SPACING_MM
    return ndi_zoom(volume, (zoom_factor_x, zoom_factor_y, 1.0), order=0)


def _normalize_and_resize(img_slice, mask_slice, size=(384, 384)):
    """Mirrors breast_dataset.py's _load_and_preprocess normalization exactly,
    but operating on already-loaded numpy slices instead of file paths."""
    img_slice = cv2.resize(img_slice.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    mask_slice = cv2.resize(mask_slice.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)

    denom = img_slice.max() - img_slice.min()
    img_slice = (img_slice - img_slice.min()) / denom if denom > 0 else img_slice
    mask_slice = (mask_slice > 0).astype(np.float32)  # binarize (merges PZ+TZ for Decathlon)

    return img_slice, mask_slice


def _index_promise12(root):
    """
    PROMISE12 layout (confirmed):
        root/Case00.mhd + Case00.raw           (volume)
        root/Case00_segmentation.mhd + .raw    (binary mask)
        ... Case01 .. Case49
    Returns a list of (case_id, "promise12") tuples; actual slice extraction
    happens lazily in PICKLE-friendly _extract_slices() to avoid loading all
    50 volumes into memory upfront.
    """
    case_paths = sorted(glob.glob(os.path.join(root, "Case[0-9][0-9].mhd")))
    return [(p, "promise12") for p in case_paths]


def _index_decathlon(root):
    """
    Decathlon Task05_Prostate layout (confirmed):
        root/imagesTr/prostate_XX.nii.gz   (4D: H, W, slices, 2 -- ch0=T2, ch1=ADC)
        root/labelsTr/prostate_XX.nii.gz   (3D: H, W, slices -- values {0,1,2})
    Skips macOS resource-fork junk files (._prostate_XX.nii.gz).
    """
    images_dir = os.path.join(root, "imagesTr")
    image_paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, "*.nii.gz"))
        if not os.path.basename(p).startswith("._")
    )
    return [(p, "decathlon") for p in image_paths]


def _extract_slices_promise12(volume_mhd_path, min_fg_fraction=0.0):
    """Loads one PROMISE12 case, resamples to common in-plane spacing,
    extracts 2D (image, mask) slice pairs, filters empty-mask slices."""
    seg_path = volume_mhd_path.replace(".mhd", "_segmentation.mhd")
    img_sitk = sitk.ReadImage(volume_mhd_path)
    mask_sitk = sitk.ReadImage(seg_path)

    img_sitk = _resample_to_spacing_sitk(img_sitk, is_mask=False)
    mask_sitk = _resample_to_spacing_sitk(mask_sitk, is_mask=True)

    img_vol = sitk.GetArrayFromImage(img_sitk)   # (z, y, x)
    mask_vol = sitk.GetArrayFromImage(mask_sitk)

    slices = []
    for z in range(img_vol.shape[0]):
        img_slice, mask_slice = img_vol[z], mask_vol[z]
        fg_frac = (mask_slice > 0).sum() / mask_slice.size
        if fg_frac > min_fg_fraction:
            slices.append((img_slice, mask_slice))
    return slices


def _extract_slices_decathlon(volume_nii_path, min_fg_fraction=0.0):
    """Loads one Decathlon case, extracts T2 channel only, resamples,
    merges PZ+TZ labels into binary, extracts 2D slice pairs."""
    label_path = volume_nii_path.replace("imagesTr", "labelsTr")

    img_nii = nib.load(volume_nii_path)
    label_nii = nib.load(label_path)

    img_vol_4d = img_nii.get_fdata()             # (H, W, slices, 2)
    t2_vol = img_vol_4d[..., 0]                   # T2 channel only, per project plan
    label_vol = label_nii.get_fdata()             # (H, W, slices), values {0,1,2}

    zooms = img_nii.header.get_zooms()[:2]        # (x, y) mm per voxel
    t2_vol = _resample_to_spacing_nib(t2_vol, zooms)
    label_vol = _resample_to_spacing_nib_nearest(label_vol, zooms)  # nearest would be better; see note below

    slices = []
    for z in range(t2_vol.shape[2]):
        img_slice = t2_vol[:, :, z]
        mask_slice = (label_vol[:, :, z] > 0).astype(np.float32)  # merge PZ(1)+TZ(2) -> 1
        fg_frac = mask_slice.sum() / mask_slice.size
        if fg_frac > min_fg_fraction:
            slices.append((img_slice, mask_slice))
    return slices

def split_promise12_clients(root, seed=42):
    """
    Random 50/50 case-level split of PROMISE12 into two synthetic federated
    clients (no site/institution metadata was shipped with this download,
    so a real-institution split, as used for the breast task, isn't
    possible here -- this is a synthetic non-IID partition by case index,
    weaker non-IID grounding than the breast task's 2-real-dataset split).
    Returns (client1_cases, client2_cases), each a list of
    (case_path, "promise12") tuples matching _index_promise12's format.
    """
    all_cases = _index_promise12(root)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_cases))
    half = len(all_cases) // 2
    client1 = [all_cases[i] for i in idx[:half]]
    client2 = [all_cases[i] for i in idx[half:]]
    return client1, client2

class ProstateClientDataset(Dataset):
    """
    Same interface as BreastClientDataset: __getitem__ returns
    (1, H, W) float tensor in [0,1], (H, W) long binary mask tensor.

    `client_name` selects which dataset to index: "PROMISE12" or "Decathlon".
    All slices are extracted and cached in memory at construction time
    (50 PROMISE12 cases / 32 Decathlon cases at ~15-47 slices each is a few
    thousand 2D slices total -- manageable in RAM; revisit with a lazy
    per-case cache if this becomes a memory problem on your machine).
    """
    INDEXERS = {"PROMISE12": _index_promise12, "Decathlon": _index_decathlon}
    EXTRACTORS = {"promise12": _extract_slices_promise12, "decathlon": _extract_slices_decathlon}

    def __init__(self, client_name, root, size=(384, 384), min_fg_fraction=0.0, cases=None):
        self.size = size
        case_list = cases if cases is not None else self.INDEXERS[client_name](root)
        if len(case_list) == 0:
            raise RuntimeError(f"No cases found for {client_name} at {root}.")

        self.slices = []
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
        img_slice, mask_slice = self.slices[idx]
        img, mask = _normalize_and_resize(img_slice, mask_slice, self.size)
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask_t = torch.from_numpy(mask).long()       # (H, W)
        return img_t, mask_t