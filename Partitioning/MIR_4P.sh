#!/bin/bash

#SBATCH --mem=50gb
#SBATCH --time=00:60:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH -e stderr.e
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=Claire-ROP
#SBATCH --output=%j.out

#module load  compiler/gnu/10
#module load  mpi/openmpi/4.0


TS=/home/.../TotalSegmentator/bin
source /home/.../claire/deps/env_source.sh

#Regularization parameter
betacont=("5e-3")

##Dataset
DATA=/path/to/dataset


export DATA

# Set flags to control which sections to run

#Partitioning(C1-C5), small images
run_section_1=0
#Partitioning(C6-C10), large images
run_section_2=0
#Run CLAIRE and get the warped images
run_section_3=1
#Cropping & Merging
run_section_4=1
#Padding (for large images)
run_section_5=1
#Mask
run_section_6=1
#Compute dice
run_section_7=1

if [ "$run_section_1" -eq "1" ]; then
    echo "Running the first section:Partitioning"

 # Call the Python script to partition the small images (256x256) to 4 partitions (128x128)   
        python_script='

import os
import time
import SimpleITK as sitk
from multiprocessing import Pool

def crop_and_save(input_file, input_image, start_indices, cropped_size, position_suffix):
    output_filenames = []
    for j, start_index in enumerate(start_indices):
        extract = sitk.ExtractImageFilter()
        extract.SetSize(cropped_size)
        extract.SetIndex(start_index)

        start = time.time()
        cropped_image = extract.Execute(input_image)
        end = time.time()
        print("Partitioning time:")
        print(end - start) 

        cropped_image.SetSpacing(input_image.GetSpacing())
        cropped_image.SetOrigin(input_image.GetOrigin())

        position = position_suffix[j]
        output_filename = f"{os.path.splitext(os.path.splitext(input_file)[0])[0]}_{position}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)

        sitk.WriteImage(cropped_image, output_path)
        output_filenames.append(output_filename)
    return output_filenames


def process_input_file(input_file, cropped_size, start_indices, position_suffix):
    input_path = os.path.join(base_dir, input_file)
    input_image = sitk.ReadImage(input_path)
    input_size = input_image.GetSize()
    print(f"Input image: {input_file} | Size: {input_size}")

    output_filenames = crop_and_save(input_file, input_image, start_indices, cropped_size, position_suffix)


base_dir = os.environ["DATA"]
output_dir = os.environ["DATA"]
input_files = ["mr.nii.gz", "mt.nii.gz"]
    
common_image = sitk.ReadImage(os.path.join(base_dir, "mr.nii.gz"))

start = time.time()

#2Patitions
start_indices_2 = [(0, 0, 0), (int(common_image.GetWidth() / 2) - 8, 0, 0)]
cropped_size_2 = (int(common_image.GetWidth()/2)+8, common_image.GetHeight(), common_image.GetDepth())
position_suffix_2 = ["r", "l"]


task = [(input_file, cropped_size_2, start_indices_2, position_suffix_2) for input_file in input_files]

with Pool() as pool:
    pool.starmap(process_input_file, task)

common_image_2 = sitk.ReadImage(os.path.join(base_dir, "mr_l.nii.gz"))

#4Partitions
cropped_size_4 = (common_image_2.GetWidth(), int(common_image_2.GetHeight()/2)+8, common_image_2.GetDepth())
start_indices_4 = [(0, 0, 0), (0, int(common_image_2.GetHeight()/2)-8, 0)]
position_suffix_4 = ["u", "l"]

input_files_4 = ["mr_r.nii.gz", "mr_l.nii.gz", "mt_r.nii.gz", "mt_l.nii.gz"]
input_task_4 = [(input_file, cropped_size_4, start_indices_4, position_suffix_4) for input_file in input_files_4]

with Pool() as pool:
    pool.starmap(process_input_file, input_task_4)

end = time.time()
print(end - start) 
'
    echo "${python_script}" | python 
fi    

if [ "$run_section_2" -eq "1" ]; then
      echo "Running the second section: Partitioning"
    # Call the Python script to partition the C6 to C10 images (512x512) to 4 partitions (256x128)     
    python_script='
import os
import time
import SimpleITK as sitk
from multiprocessing import Pool

def crop_and_save(input_file, input_image, start_indices, cropped_size, position_suffix):
    output_filenames = []
    for j, start_index in enumerate(start_indices):
        extract = sitk.ExtractImageFilter()
        extract.SetSize(cropped_size)
        extract.SetIndex(start_index)

        start = time.time()
        cropped_image = extract.Execute(input_image)
        end = time.time()
        print("Partitioning time:")
        print(end - start) 

        cropped_image.SetSpacing(input_image.GetSpacing())
        cropped_image.SetOrigin(input_image.GetOrigin())

        position = position_suffix[j]
        output_filename = f"{os.path.splitext(os.path.splitext(input_file)[0])[0]}_{position}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)

        sitk.WriteImage(cropped_image, output_path)
        output_filenames.append(output_filename)
    return output_filenames


def process_input_file(input_file, cropped_size, start_indices, position_suffix):
    input_path = os.path.join(base_dir, input_file)
    input_image = sitk.ReadImage(input_path)
    input_size = input_image.GetSize()
    print(f"Input image: {input_file} | Size: {input_size}")

    output_filenames = crop_and_save(input_file, input_image, start_indices, cropped_size, position_suffix)


base_dir = os.environ["DATA"]
output_dir = os.environ["DATA"]
input_files = ["mr.nii.gz", "mt.nii.gz"]
    
common_image = sitk.ReadImage(os.path.join(base_dir, "mr.nii.gz"))

start = time.time()

#w/overlap & remove background 
start_indices_2 = [(0, 100, 0), (int(common_image.GetWidth() / 2) - 8, 100, 0)]
cropped_size_2 = (int(common_image.GetWidth()/2)+8, int(common_image.GetHeight()/2), common_image.GetDepth())
position_suffix_2 = ["r", "l"]


task = [(input_file, cropped_size_2, start_indices_2, position_suffix_2) for input_file in input_files]


with Pool() as pool:
    pool.starmap(process_input_file, task)

common_image_2 = sitk.ReadImage(os.path.join(base_dir, "mr_l.nii.gz"))

#4P
cropped_size_4 = (common_image_2.GetWidth(), int(common_image_2.GetHeight()/2)+8, common_image_2.GetDepth())
start_indices_4 = [(0, 0, 0), (0, int(common_image_2.GetHeight()/2)-8, 0)]
position_suffix_4 = ["u", "l"]

input_files_4 = ["mr_r.nii.gz", "mr_l.nii.gz", "mt_r.nii.gz", "mt_l.nii.gz"]
input_task_4 = [(input_file, cropped_size_4, start_indices_4, position_suffix_4) for input_file in input_files_4]

with Pool() as pool:
    pool.starmap(process_input_file, input_task_4)    

end = time.time()
print(end - start)  

'

    echo "${python_script}" | python      
fi

if [ "$run_section_3" -eq "1" ]; then
   echo "Running the second section:Run CLAIRE and getting the warped image"


Partition_mt=("mt_r_u" "mt_l_u" "mt_r_l" "mt_l_l")
Partition_mr=("mr_r_u" "mr_l_u" "mr_r_l" "mr_l_l")
P=("RU" "LU" "RL" "LL")


#Define a function to run the registration command to get deformation maps
run_registration_defmap() {
    local Partition_mt="$1"
    local Partition_mr="$2"
    local index="$3"
    local defmap_name="$4"


    mpirun ./claire -mt "$DATA/$Partition_mt.nii.gz" -mr "$DATA/$Partition_mr.nii.gz" \
    -regnorm h1s-div -maxit 50 -krylovmaxit 100 -precond invreg -iporder 1 \
    -betacont "$betacont" -beta-div 1e-04 -diffpde finite -verbosity 2 \
     -x  "$DATA/${defmap_name}_"  -defmap
}

#Define a function to run the registration command to get the registration time
run_registration_time() {
    local Partition_mt="$1"
    local Partition_mr="$2"
    local index="$3"
    local defmap_name="$4"
    local betacont_filename="${betacont//./_}" 
    
    CUDA_VISIBLE_DEVICES=$index mpirun ./claire -mt "$DATA/$Partition_mt.nii.gz" -mr "$DATA/$Partition_mr.nii.gz" \
    -regnorm h1s-div -maxit 50 -krylovmaxit 100 -precond invreg -iporder 1 \
    -betacont "$betacont" -beta-div 1e-04 -diffpde finite -verbosity 2 \
    &> "$DATA/${defmap_name}_${betacont_filename}"
}


    for ((i = 0; i < ${#Partition_mr[@]}; i++)); do
        echo "Processing iteration $i with Partition"
        run_registration_defmap "${Partition_mt[i]}" "${Partition_mr[i]}" "$i" "${P[i]}"
        echo "mt: ${Partition_mt[i]}"
        echo "mr: ${Partition_mr[i]}"
        run_registration_time "${Partition_mt[i]}" "${Partition_mr[i]}" "$i" "${P[i]}"
    done
wait    
   
cases=("LL" "LU" "RL" "RU")
args_ifile=(
    $DATA/mt_l_l.nii.gz
    $DATA/mt_l_u.nii.gz
    $DATA/mt_r_l.nii.gz
    $DATA/mt_r_u.nii.gz
)
args_xfile=(
    $DATA/deformed_mt_ll.nii.gz
    $DATA/deformed_mt_lu.nii.gz
    $DATA/deformed_mt_rl.nii.gz
    $DATA/deformed_mt_ru.nii.gz
)

for i in "${!cases[@]}"; do
    case="${cases[i]}"
    case_args_ifile="${args_ifile[i]}"
    case_args_xfile="${args_xfile[i]}"   

 #Call the Python script_3 to get the warped image (input: moving image, deformation maps)
    python_script_3=$(cat <<EOF
import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
import os, sys
import easydict
import numpy as np
import SimpleITK as sitk
from ipywidgets import interact, fixed
import re

data_path = os.environ["DATA"]

def writeNII(img, filename, affine=None, ref_image=None):
    # function to write a nifti image, creates a new nifti object
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))
    nib.save(data, filename);

def readImage(filename):
    return nib.load(filename).get_fdata(), nib.load(filename)

if __name__ == "__main__":  
    args = easydict.EasyDict({
        "ifile": "${case_args_ifile}",   
        "xfile": "${case_args_xfile}",  
        "odir": os.path.join(data_path),
        "out": "result"
    })

    ext = ".nii.gz"
    print("loading images")
    odir = args.odir
    

    def1,_ = readImage(os.path.join(odir, "${case}_deformation-map-x1{}".format(ext)))
    def2,_ = readImage(os.path.join(odir, "${case}_deformation-map-x2{}".format(ext)))
    def3,_ = readImage(os.path.join(odir, "${case}_deformation-map-x3{}".format(ext)))   

    template, ref_image = readImage(args.ifile)

    nx = template.shape
    pi = np.pi
    h = 2*pi/np.asarray(nx);
    x = np.linspace(0, 2*pi-h[0], nx[0]);
    y = np.linspace(0, 2*pi-h[1], nx[1]);
    z = np.linspace(0, 2*pi-h[2], nx[2]);

    points = (x,y,z)

    query = np.stack([def3.flatten(), def2.flatten(), def1.flatten()], axis=1)

    print("evaluating interpolation")
    iporder = 1
    if iporder == 0:
        method = "nearest"
    if iporder == 1:
        method = "linear"
    output = interpn(points, template, query, method=method, bounds_error=False, fill_value=0)
    output = np.reshape(output, nx)
    print("writing output")
    writeNII(output, args.xfile, ref_image=ref_image)
    print("Done!")
EOF
)

   echo "${python_script_3}" | python 
   echo "Processing case: ${case}"
done

#End of section 2
fi

if [ "$run_section_4" -eq "1" ]; then
   echo "Running the third section: Cropping the images"


cases=("ru" "lu" "rl" "ll")

declare -A lower_crop_sizes
declare -A upper_crop_sizes

lower_crop_sizes["ll"]="8,8,0"
lower_crop_sizes["lu"]="8,0,0"
lower_crop_sizes["rl"]="0,8,0"
lower_crop_sizes["ru"]="0,0,0"

upper_crop_sizes["ll"]="0,0,0"
upper_crop_sizes["lu"]="0,8,0"
upper_crop_sizes["rl"]="8,0,0"
upper_crop_sizes["ru"]="8,8,0"

cropped_images=()

for case in "${cases[@]}"; do
    deformed_input_file="$DATA/deformed_mt_${case}.nii.gz"
    cropped_output_file="$DATA/deformed_mt_${case}_cropped.nii.gz"
    lower_crop_size="${lower_crop_sizes[$case]}"
    upper_crop_size="${upper_crop_sizes[$case]}"

    # Add the cropped image path to the list
    cropped_images+=("$DATA/deformed_mt_${case}_cropped.nii.gz")
    
    # Print some information for debugging
    echo "Processing case: $case"
    echo "Input file: $deformed_input_file"
    echo "Cropped output file: $cropped_output_file"
    echo "Lower crop size: $lower_crop_size"
    echo "Upper crop size: $upper_crop_size"

    # Call the Python script_4 to crop the warped images
    python_script_4=$(cat <<EOF
import SimpleITK as sitk
import time
import os

data_path = os.environ["DATA"]

img_fixed = sitk.ReadImage(os.path.join(data_path, "mr.nii.gz"))

deformed_image = sitk.ReadImage(os.path.join(data_path, '$deformed_input_file'))


crop = sitk.CropImageFilter()
lower_crop_size = [int(val) for val in '$lower_crop_size'.split(',')]
upper_crop_size = [int(val) for val in '$upper_crop_size'.split(',')]

print("Before cropping - Image size:", deformed_image.GetSize())
print("Lower crop size:", lower_crop_size)
print("Upper crop size:", upper_crop_size)

crop.SetLowerBoundaryCropSize(lower_crop_size)
crop.SetUpperBoundaryCropSize(upper_crop_size)

start = time.time()
cropped_image = crop.Execute(deformed_image)
print("Cropping time:")
end = time.time()
print(end - start)  


sitk.WriteImage(cropped_image, os.path.join(data_path, '$cropped_output_file'))

print("After cropping - Image size:", cropped_image.GetSize())

print("Total time:")
end = time.time()
print(end - start)  # time in seconds
print("Done!")
EOF
)
    echo "${python_script_4}" | python

done

echo "List of cropped images:"
for cropped_image_path in "${cropped_images[@]}"; do
    echo "$cropped_image_path"
done

export CROPPED_IMAGES="${cropped_images[*]}"

 # Call the Python script_5 to merge the warped images
    python_script_5='
import SimpleITK as sitk
import nibabel as nb
import numpy as np
import time
import os

data_path = os.environ["DATA"]
cropped_images = os.environ["CROPPED_IMAGES"].split()
img_fixed = sitk.ReadImage(os.path.join(data_path, "mr.nii.gz"))

cropped_deformed_images = [sitk.ReadImage(os.path.join(data_path, image_path)) for image_path in cropped_images]

print("Tiling!") 
start = time.time()
tile = sitk.TileImageFilter()
end = time.time()
print("Merging time:")
print(end - start)  # time in seconds

layout = [2, 2, 0]
tile.SetLayout(layout)

tiled_image = tile.Execute(cropped_deformed_images)

tiled_image.SetOrigin(img_fixed.GetOrigin())
tiled_image.SetDirection(img_fixed.GetDirection())
sitk.WriteImage(tiled_image, os.path.join(data_path, "tiled_deformed_mt.nii.gz"))

end = time.time()
print("Total time:")
print(end - start)  # time in seconds
print("Done!")
'

echo "${python_script_5}" | python

#end of section 3
fi

if [ "$run_section_5" -eq "1" ]; then
   echo "Running the fourth section: Padding the images"
# Call the Python script_6 for padding
            python_script_6='
import SimpleITK as sitk
from enum import Enum
import numpy as np
import os
from ipywidgets import interact, fixed

data_path = os.environ["DATA"]

pad = sitk.ConstantPadImageFilter()
input =  sitk.ReadImage(os.path.join(data_path, "tiled_deformed_mt.nii.gz")) 

pad.SetPadLowerBound((0,100,0)) 
pad.SetPadUpperBound((0,156,0))

pad_img = pad.Execute(input)
pad_img.SetOrigin((1,1,1))

sitk.WriteImage(pad_img, os.path.join(data_path, "tiled_deformed_mt.nii.gz"))

'
echo "${python_script_6}" | python
#End of section 4
fi

if [ "$run_section_6" -eq "1" ]; then
   echo "Running the fifth section: Get the masks"

#Masks
##Reference image (mr)
python3.8 $TS/TotalSegmentator -i $DATA/mr.nii.gz -o $DATA/mask_mr
python3.8 $TS/totalseg_combine_masks -i $DATA/mask_mr -o $DATA/mask_mr.nii.gz -m lung

##Moving image (mt)
python3.8 $TS/TotalSegmentator -i $DATA/mt.nii.gz -o $DATA/mask_mt
python3.8 $TS/totalseg_combine_masks -i $DATA/mask_mt -o $DATA/mask_mt.nii.gz -m lung

##Deformed image
python3.8 $TS/TotalSegmentator -i $DATA/tiled_deformed_mt.nii.gz  -o $DATA/tiled_deformed_mt
python3.8 $TS/totalseg_combine_masks -i $DATA/tiled_deformed_mt -o $DATA/mask_tiled_deformed.nii.gz -m lung

#End of section 5
fi

if [ "$run_section_7" -eq "1" ]; then
   echo "Running the sixth section: Compute dice"

# Call the Python script_7 to compute registration accuracy (Dice,...)
python_script_7='
import SimpleITK as sitk
from enum import Enum
import numpy as np
from ipywidgets import interact, fixed
import os 

data_path = os.environ["DATA"]

reference_segmentation = sitk.ReadImage(os.path.join(data_path, "mask_mr.nii.gz"))
reference_segmentation.SetOrigin((1,1,1))

#Before registration (moving image)
mask = sitk.ReadImage(os.path.join(data_path, "mask_mt.nii.gz"))
mask.SetOrigin((1,1,1))

#After registration (deformed moving image)
mask_deformed = sitk.ReadImage(os.path.join(data_path, "mask_tiled_deformed.nii.gz"))
mask_deformed.SetOrigin(mask.GetOrigin())

#Before and after
segmentations = [mask, mask_deformed]

class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)

class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)
    
# Empty numpy arrays to hold the results 
overlap_results = np.zeros((len(segmentations),len(OverlapMeasures.__members__.items())))  
 
# Compute the evaluation criteria
overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

for i, seg in enumerate(segmentations):
    overlap_measures_filter.Execute(reference_segmentation, seg)
    overlap_results[i,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    
print ("Overlap Results (Dice) before and after:" ,overlap_results)    
'

echo "${python_script_7}" | python > $DATA/overlap_"$betacont"
fi
