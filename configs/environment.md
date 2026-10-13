# Environment setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install monai scikit-learn opencv-python pyyaml pandas matplotlib seaborn
pip install grad-cam   # pytorch-grad-cam, for Grad-CAM/Grad-CAM++
```

GPU: any CUDA-capable GPU with >=8GB VRAM is enough at 384x384 resolution with
batch size 8-16 for the Residual U-Net. Reduce batch size if you hit OOM.

If you don't have a local GPU, Kaggle Notebooks or Google Colab (T4/A100) both
work fine for this scale of experiment — useful since your datasets are already
on Kaggle.
