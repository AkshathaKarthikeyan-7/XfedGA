import os
import glob

root = 'raw data/BUS-BRA'
base = os.path.join(root, 'BUSBRA')
print('base isdir:', os.path.isdir(base), '->', base)

images_dir = os.path.join(base, 'Images')
masks_dir = os.path.join(base, 'Masks')
print('images_dir isdir:', os.path.isdir(images_dir), '->', images_dir)
print('masks_dir isdir:', os.path.isdir(masks_dir), '->', masks_dir)

imgs = glob.glob(os.path.join(images_dir, '*.png'))
print('num images found:', len(imgs))
if imgs:
    sample = imgs[0]
    print('sample image:', sample)
    fname = os.path.basename(sample)
    mask_fname = fname.replace('bus_', 'mask_', 1)
    mask_path = os.path.join(masks_dir, mask_fname)
    print('expected mask filename:', mask_fname)
    print('expected mask path:', mask_path)
    print('mask exists:', os.path.exists(mask_path))
