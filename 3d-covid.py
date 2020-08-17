#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import miscnn
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.data_loading.data_io import Data_IO
from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling
from miscnn.processing.data_augmentation import Data_Augmentation
from miscnn.processing.preprocessor import Preprocessor
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.architecture.unet.dense import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, dice_crossentropy, tversky_loss
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, CSVLogger
from miscnn.evaluation.cross_validation import cross_validation

import requests
from tqdm import tqdm
import os
import zipfile
from IPython.display import Image


# In[ ]:


# Links to the data set
url_vol = "https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1"
url_seg = "https://zenodo.org/record/3757476/files/Lung_and_Infection_Mask.zip?download=1"


# In[ ]:


path_data = "data"


# In[ ]:


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        print("WARNING: Skipping download due to files are already there.")
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


# In[ ]:


# Create data structure
if not os.path.exists(path_data) : os.makedirs(path_data)


# In[ ]:


# Download CT volumes and save them into the data directory
path_vol_zip = os.path.join(path_data, "volumes.zip")
print("INFO:", "Downloading Volumes")
download_from_url(url_vol, path_vol_zip)
# Download segmentations and save them into the data directory
path_seg_zip = os.path.join(path_data, "segmentations.zip")
print("INFO:", "Downloading Segmentations")
download_from_url(url_seg, path_seg_zip)


# In[ ]:


# Extract sample list from the ZIP file
print("INFO:", "Obtain sample list from the volumes ZIP file")
with zipfile.ZipFile(path_vol_zip, "r") as zip_vol:
    sample_list = zip_vol.namelist()


# In[ ]:


# Iterate over the sample list and extract each sample from the ZIP files
print("INFO:", "Extracting data from ZIP files")
for sample in tqdm(sample_list):
    # Skip if file does not end with nii.gz
    if not sample.endswith(".nii.gz") : continue
    # Create sample directory
    path_sample = os.path.join(path_data, sample[:-len(".nii.gz")])
    if not os.path.exists(path_sample) : os.makedirs(path_sample)
    # Extract volume and store file into the sample directory
    with zipfile.ZipFile(path_vol_zip, "r") as zip_vol:
        zip_vol.extract(sample, path_sample)
    os.rename(os.path.join(path_sample, sample),
              os.path.join(path_sample, "imaging.nii.gz"))
    # Extract segmentation and store file into the sample directory
    with zipfile.ZipFile(path_seg_zip, "r") as zip_seg:
        zip_seg.extract(sample, path_sample)
    os.rename(os.path.join(path_sample, sample),
              os.path.join(path_sample, "segmentation.nii.gz"))


# In[ ]:


## We are using 4 classes due to [background, lung_left, lung_right, covid-19]
interface = NIFTI_interface(channels=1, classes=4)


# In[ ]:


data_io = miscnn.Data_IO(interface, path_data)


# In[ ]:


# Create and configure the Data Augmentation class
data_aug = miscnn.Data_Augmentation(cycles=1, scaling=True, rotations=True,
                                    elastic_deform=True, mirror=True,
                                    brightness=True, contrast=True,
                                    gamma=True, gaussian_noise=True)


# In[ ]:


# Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
sf_clipping = Clipping(min=-1250, max=250)
# Create a pixel value normalization Subfunction to scale between 0-255
sf_normalize = Normalization(mode="grayscale")
# Create a resampling Subfunction to voxel spacing 1.58 x 1.58 x 2.70
sf_resample = Resampling((1.58, 1.58, 2.70))
# Create a pixel value normalization Subfunction for z-score scaling
sf_zscore = Normalization(mode="z-score")


# In[ ]:


# Assemble Subfunction classes into a list
sf = [sf_clipping, sf_normalize, sf_resample, sf_zscore]


# In[ ]:


# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="patchwise-crop", patch_shape=(160, 160, 80))


# In[ ]:


# Adjust the patch overlap for predictions
pp.patchwise_overlap = (80, 80, 40)


# In[ ]:


# Initialize the Architecture
unet_dense = Architecture(activation="softmax")


# In[ ]:


# Create the Neural Network model
model = Neural_Network(preprocessor=pp, architecture=unet_dense,
                       loss=tversky_crossentropy,
                       metrics=[tversky_loss, dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=3, learninig_rate=0.001)


# In[ ]:


sample_list = data_io.get_indiceslist()
sample_list.sort()
sample_list = sample_list[:-2]
sample_list


# In[ ]:


cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, mode='min', min_delta=0.0001, 
                          cooldown=1, min_lr=0.00001)


# In[ ]:


# Run cross-validation function
cross_validation(sample_list, model, k_fold=5, epochs=5, iterations=50,
                 evaluation_path="evaluation", draw_figures=True, callbacks=[cb_lr], save_models=False, return_output=False)


# In[ ]:


Image(filename = "evaluation/fold_0/validation.dice_soft.png")


# In[ ]:


Image(filename = "evaluation/fold_0/validation.loss.png")


# In[ ]:


Image(filename = "evaluation/fold_0/validation.dice_crossentropy.png")

