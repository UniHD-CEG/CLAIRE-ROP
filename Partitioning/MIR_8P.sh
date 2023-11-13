#!/bin/bash

#SBATCH --mem=40gb
#SBATCH --time=00:60:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH -e stderr.e
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=MIR_8P
#SBATCH --output=%j.out

#Load modules
#module load  compiler/gnu/10
#module load  mpi/openmpi/4.0
#module load  devel/cuda/11.6


#Regularization parameter
betacont=("5e-3")

#Dataset
DATA=/path/to/dataset

export DATA

#Set this flag to zero to ignore the edge partitions ["Case 9" and "Case 10"]
result=1

# Set flags to control which sections to run
#Partitioning(small dataset)
run_section_1=1
#Partitioning(large dataset)
run_section_2=0
#Run CLAIRE and get the warped images
run_section_3=1
#Cropping & Merging
run_section_4=1
#Padding (large dataset)
run_section_5=0
#Mask
run_section_6=1
#Compute dice
run_section_7=1

if [ "$run_section_1" -eq "1" ]; then
    echo "Running the first section: Partitioning"
    # Call the Python script to partition the image (C1 to C5)     
        python_script='

import os
import SimpleITK as sitk
import time
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

#2P
start_indices_2 = [(0, 0, 0), (int(common_image.GetWidth() / 2) - 8, 0, 0)]
cropped_size_2 = (int(common_image.GetWidth()/2)+8, common_image.GetHeight(), common_image.GetDepth())
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


common_image_3 = sitk.ReadImage(os.path.join(base_dir, "mr_l_l.nii.gz"))

#8P
input_files_8_r = ["mr_r_u.nii.gz", "mr_r_l.nii.gz","mt_r_u.nii.gz", "mt_r_l.nii.gz"]
start_indices_8_r_e = [(0, 0, 0)]
start_indices_8_r_i = [(int((common_image_3.GetWidth()-8) / 2) - 8, 0, 0)]
#######
input_files_8_l = ["mr_l_u.nii.gz", "mr_l_l.nii.gz","mt_l_u.nii.gz", "mt_l_l.nii.gz"]
start_indices_8_l_i = [(0, 0, 0)]
start_indices_8_l_e = [(int((common_image_3.GetWidth()-8) / 2) , 0, 0)]

cropped_size_8_i = (int((common_image_3.GetWidth()-8)//2)+16, common_image_3.GetHeight(), common_image_3.GetDepth())
cropped_size_8_e = (int((common_image_3.GetWidth()-8)//2)+8, common_image_3.GetHeight(), common_image_3.GetDepth())
position_suffix_8_i = ["i"]
position_suffix_8_e = ["e"]

input_task_8_r_i = [(input_file, cropped_size_8_i, start_indices_8_r_i, position_suffix_8_i) for input_file in input_files_8_r]
input_task_8_r_e = [(input_file, cropped_size_8_e, start_indices_8_r_e, position_suffix_8_e) for input_file in input_files_8_r]

input_task_8_l_i = [(input_file, cropped_size_8_i, start_indices_8_l_i, position_suffix_8_i) for input_file in input_files_8_l]
input_task_8_l_e = [(input_file, cropped_size_8_e, start_indices_8_l_e, position_suffix_8_e) for input_file in input_files_8_l]


with Pool() as pool:
    pool.starmap(process_input_file, input_task_8_r_i)
    pool.starmap(process_input_file, input_task_8_r_e)
    pool.starmap(process_input_file, input_task_8_l_e)
    pool.starmap(process_input_file, input_task_8_l_i)
    

end = time.time()
print(end - start) 
        '
         echo "${python_script}" | python  
fi         

if [ "$run_section_2" -eq "1" ]; then
    echo "Running the first section:Partitioning"
  
        # Call the Python script to partition the image (C6 to C10)     
        python_script='
import os
import SimpleITK as sitk
import time
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

#2P
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


common_image_3 = sitk.ReadImage(os.path.join(base_dir, "mr_l_l.nii.gz"))

#8P
input_files_8_r = ["mr_r_u.nii.gz", "mr_r_l.nii.gz","mt_r_u.nii.gz", "mt_r_l.nii.gz"]
start_indices_8_r_e = [(0, 0, 0)]
start_indices_8_r_i = [(int((common_image_3.GetWidth()-8) / 2) - 8, 0, 0)]
#######
input_files_8_l = ["mr_l_u.nii.gz", "mr_l_l.nii.gz","mt_l_u.nii.gz", "mt_l_l.nii.gz"]
start_indices_8_l_i = [(0, 0, 0)]
start_indices_8_l_e = [(int((common_image_3.GetWidth()-8) / 2) , 0, 0)]


cropped_size_8_i = (int((common_image_3.GetWidth()-8)/2)+16, common_image_3.GetHeight(), common_image_3.GetDepth())
cropped_size_8_e = (int((common_image_3.GetWidth()-8)/2)+8, common_image_3.GetHeight(), common_image_3.GetDepth())

position_suffix_8_i = ["i"]
position_suffix_8_e = ["e"]

input_task_8_r_i = [(input_file, cropped_size_8_i, start_indices_8_r_i, position_suffix_8_i) for input_file in input_files_8_r]
input_task_8_r_e = [(input_file, cropped_size_8_e, start_indices_8_r_e, position_suffix_8_e) for input_file in input_files_8_r]

input_task_8_l_i = [(input_file, cropped_size_8_i, start_indices_8_l_i, position_suffix_8_i) for input_file in input_files_8_l]
input_task_8_l_e = [(input_file, cropped_size_8_e, start_indices_8_l_e, position_suffix_8_e) for input_file in input_files_8_l]


with Pool() as pool:
    pool.starmap(process_input_file, input_task_8_r_i)
    pool.starmap(process_input_file, input_task_8_r_e)
    pool.starmap(process_input_file, input_task_8_l_i)
    pool.starmap(process_input_file, input_task_8_l_e)

end = time.time()
print(end - start) 
'
    echo "${python_script}" | python      
fi


if [ "$run_section_3" -eq "1" ]; then
   echo "Running the third section:Run CLAIRE and getting the warped image"

#inner partitions (4P)
Partition_mt=("mt_r_u_i" "mt_l_u_i" "mt_r_l_i" "mt_l_l_i")
Partition_mr=("mr_r_u_i" "mr_l_u_i" "mr_r_l_i" "mr_l_l_i")
P=("RU" "LU" "RL" "LL")

#all partitions
All_partition_mt=("mt_r_u_e" "mt_r_u_i" "mt_l_u_i" "mt_l_u_e" "mt_r_l_e" "mt_r_l_i" "mt_l_l_i" "mt_l_l_e")
All_partition_mr=("mr_r_u_e" "mr_r_u_i" "mr_l_u_i" "mr_l_u_e" "mr_r_l_e" "mr_r_l_i" "mr_l_l_i" "mr_l_l_e")
All_P=("RU_1" "RU_2" "LU_1" "LU_2" "RL_1" "RL_2" "LL_1" "LL_2")

#Define a function to run the registration command to get deofrmation maps
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

# Execute the appropriate registration command based on the mask value
if [ "$result" -eq "0" ]; then
    for ((i = 0; i < ${#Partition_mr[@]}; i++)); do
        echo "Processing iteration $i with Partition"
        run_registration_defmap "${Partition_mt[i]}" "${Partition_mr[i]}" "$i" "${P[i]}"
        echo "mt: ${Partition_mt[i]}"
        echo "mr: ${Partition_mr[i]}"
        run_registration_time "${Partition_mt[i]}" "${Partition_mr[i]}" "$i" "${P[i]}"
    done
else 
    for ((i = 0; i < ${#All_partition_mr[@]}; i++)); do
        echo "Processing iteration $i with All_partition"
        run_registration_defmap "${All_partition_mt[i]}" "${All_partition_mr[i]}" "$i" "${All_P[i]}"
        echo "mt: ${All_partition_mt[i]}"
        echo "mr: ${All_partition_mr[i]}"
        run_registration_time "${All_partition_mt[i]}" "${All_partition_mr[i]}" "$i" "${All_P[i]}"
    done    

fi    
wait
 
if [ "$result" -eq "0" ]; then
# Define the list of cases and corresponding arguments
cases=("LL" "LU" "RL" "RU")
args_ifile=(
    $DATA/mt_l_l_i.nii.gz
    $DATA/mt_l_u_i.nii.gz
    $DATA/mt_r_l_i.nii.gz
    $DATA/mt_r_u_i.nii.gz
)
args_xfile=(
    $DATA/deformed_mt_ll_1.nii.gz
    $DATA/deformed_mt_lu_1.nii.gz
    $DATA/deformed_mt_rl_2.nii.gz
    $DATA/deformed_mt_ru_2.nii.gz
)
else 

cases=("LL_1" "LL_2" "LU_1" "LU_2" "RL_1" "RL_2" "RU_1" "RU_2")
args_ifile=(
    $DATA/mt_l_l_i.nii.gz
    $DATA/mt_l_l_e.nii.gz
    $DATA/mt_l_u_i.nii.gz
    $DATA/mt_l_u_e.nii.gz
    $DATA/mt_r_l_e.nii.gz
    $DATA/mt_r_l_i.nii.gz
    $DATA/mt_r_u_e.nii.gz
    $DATA/mt_r_u_i.nii.gz
)
args_xfile=(
    $DATA/deformed_mt_ll_1.nii.gz
    $DATA/deformed_mt_ll_2.nii.gz
    $DATA/deformed_mt_lu_1.nii.gz
    $DATA/deformed_mt_lu_2.nii.gz
    $DATA/deformed_mt_rl_1.nii.gz
    $DATA/deformed_mt_rl_2.nii.gz
    $DATA/deformed_mt_ru_1.nii.gz
    $DATA/deformed_mt_ru_2.nii.gz
)
fi

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

#End of section 3
fi

if [ "$run_section_4" -eq "1" ]; then
   echo "Running the fourth section: Cropping the images"

if [ "$result" -eq "0" ]; then

cases=("ru_2" "lu_1" "rl_2" "ll_1")

declare -A lower_crop_sizes
declare -A upper_crop_sizes

lower_crop_sizes["ll_1"]="8,8,0"
lower_crop_sizes["lu_1"]="8,0,0"
lower_crop_sizes["rl_2"]="8,8,0"
lower_crop_sizes["ru_2"]="8,0,0"

upper_crop_sizes["ll_1"]="8,0,0"
upper_crop_sizes["lu_1"]="8,8,0"
upper_crop_sizes["rl_2"]="8,0,0"
upper_crop_sizes["ru_2"]="8,8,0"

layout="2"

else 

cases=("ru_1" "ru_2" "lu_1" "lu_2" "rl_1" "rl_2" "ll_1" "ll_2")

declare -A lower_crop_sizes
declare -A upper_crop_sizes

lower_crop_sizes["ll_1"]="8,8,0"
lower_crop_sizes["lu_1"]="8,0,0"
lower_crop_sizes["rl_2"]="8,8,0"
lower_crop_sizes["ru_2"]="8,0,0"
upper_crop_sizes["ll_1"]="8,0,0"
upper_crop_sizes["lu_1"]="8,8,0"
upper_crop_sizes["rl_2"]="8,0,0"
upper_crop_sizes["ru_2"]="8,8,0"

#edge partitions
lower_crop_sizes["ll_2"]="8,8,0"
lower_crop_sizes["lu_2"]="8,0,0"
lower_crop_sizes["rl_1"]="0,8,0"
lower_crop_sizes["ru_1"]="0,0,0"
upper_crop_sizes["ll_2"]="0,0,0"
upper_crop_sizes["lu_2"]="0,8,0"
upper_crop_sizes["rl_1"]="8,0,0"
upper_crop_sizes["ru_1"]="8,8,0"



layout="4"

fi

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
export layout="$layout"

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

tile = sitk.TileImageFilter()

layout = [int(os.environ["layout"]), int(os.environ["layout"]), 0]
tile.SetLayout(layout)

start = time.time() 
tiled_image = tile.Execute(cropped_deformed_images)
print("Merging time:")
end = time.time()
print(end - start)

tiled_image.SetOrigin(img_fixed.GetOrigin())
tiled_image.SetDirection(img_fixed.GetDirection())
sitk.WriteImage(tiled_image, os.path.join(data_path, "tiled_deformed_mt.nii.gz"))

print("Done!")

'

echo "${python_script_5}" | python

#end of section 4
fi


if [ "$run_section_5" -eq "1" ]; then
   echo "Running the sixth section: Padding the images"
   export result="$result"

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

if int(os.environ["result"]) == 0:
    #pad.SetPadLowerBound((128,128,0)) 
    #pad.SetPadUpperBound((128,128,0))

    pad.SetPadLowerBound((128,100,0)) 
    pad.SetPadUpperBound((128,156,0))

else:
    #pad.SetPadLowerBound((0,128,0)) 
    #pad.SetPadUpperBound((0,128,0))

    pad.SetPadLowerBound((0,100,0)) 
    pad.SetPadUpperBound((0,156,0))

pad_img = pad.Execute(input)
pad_img.SetOrigin((1,1,1))

sitk.WriteImage(pad_img, os.path.join(data_path, "tiled_deformed_mt.nii.gz"))

'
echo "${python_script_6}" | python

#End of section 5
fi

if [ "$run_section_6" -eq "1" ]; then
   echo "Running the sixth section: Get the masks"

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

#End of section 6
fi

if [ "$run_section_7" -eq "1" ]; then
   echo "Running the seventh section: Compute dice"

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
