import os
import nibabel as nib
import numpy as np
from denseinference import CRFProcessor
from keras.utils.np_utils import to_categorical

# please see https://github.com/mbickel/DenseInferenceWrapper/tree/f281143dd1af27fc57197bba8bcbea2f2aee9f74  for more details.

pro = CRFProcessor.CRF3DProcessor()
T1c=nib.load('/validation/Brats17_CBICA_AZA_1/T1c.nii').get_data()
seg = nib.load('/step2/val17_t1c/Brats17_CBICA_AZA_1.nii.gz').get_data() #step 2 output

seg=to_categorical(seg,4)
inputs_img=T1c/255
result = pro.set_data_and_run(inputs_img, seg)
T1.get_data()[:,:,:] = result
nib.save(T1,'/step2/Brats17_CBICA_AZA_1.nii.gz')
      
