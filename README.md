# GIFDL: Generated Image Fluctuation Distortion Learning for Enhancing Steganographic Security

Overall workflow:

1. Train the generative model (Train)
2. Generate embedding costs and perform embedding (Test)

---

## 1. Training Stage (Train)

Training script: `allo_train.sh`

Core command:

```bash
python -u train_gifdl.py --outf "allo_1"
```

After training, a model checkpoint will be generated in the output directory, e.g.:

```
allo_1/netG_epoch__72.pth
```

This file will be used in the subsequent testing stage.

---

## 2. Testing Stage (Test)

Testing script: `test.sh`

The testing stage consists of two steps:

1. Generate embedding costs using the trained `netG`
2. Perform embedding using a MATLAB script

### 2.1 Generate Embedding Cost

```bash
python -u generate_cost.py \
  --config "/config_allo" \
  --netG './allo_1/netG_epoch__72.pth' \
  --datacover './dataset_sd_in_gray/'
```

Explanation of arguments:

* `--netG`: trained model checkpoint from the training stage
* `--datacover`: directory of cover images to be embedded
* The generated cost files (`.mat`) will be saved under `./root/config_allo/`

Internal procedure in this stage:

* First, use `train_cover` data to recalibrate BatchNorm statistics
* Then, generate embedding costs on `datacover`

---

### 2.2 MATLAB Embedding

Then execute:

```bash
module load matlab/R2018b
srun matlab -nodisplay -nosplash -nodesktop -r "clear;\
Payload = 0.4;\
cover_dir = './dataset_sd_in_gray';\
stego_dir = './stego/allo';\
cost_dir = './root/config_allo';\
run('embedding.m');\
exit;"
```

Explanation:

* `cover_dir`: path to the original cover images
* `cost_dir`: path to the generated embedding cost files
* `stego_dir`: directory for saving the output stego images
* `Payload`: embedding rate (e.g., 0.4 bpp)

After execution, the final stego images will be generated in:

```
./stego/allo/
```

---

## 3. Notes

1. Please make sure all paths are correctly configured:

   * Training dataset path (`train_cover`)
   * Testing cover dataset path (`datacover`)

2. The image resolution used in training and testing must be consistent (default: 512×512 grayscale images).

3. You may need to modify the paths in `train_gifdl.py` and `generate_cost.py` t
