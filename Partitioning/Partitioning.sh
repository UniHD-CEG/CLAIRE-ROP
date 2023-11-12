#!/bin/bash

#SBATCH --mem=50gb
#SBATCH --time=00:60:00
#SBATCH --gres=gpu:1     
#SBATCH --partition=accelerated
#SBATCH -e stderr.e
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=CLAIRE-ROP
#SBATCH --output=%j.out

#module load  compiler/gnu/10
#module load  mpi/openmpi/4.0


DATA=/home/hk-project-irmulti/hd_fa163/Dataset/Lung/MIR/C1/2P
export DATA

export SLURM_GPUS=2  # Set SLURM_GPUS to the desired GPU count
export DATA_SET=S    # Set DATA_SET to small (S) or large (L)

echo "SLURM_GPUS set to $SLURM_GPUS GPUs."


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
   
if os.environ.get("SLURM_GPUS"):
    gpu_count = int(os.environ.get("SLURM_GPUS"))
    print(f"Proceed with {gpu_count} partitions")

    if 2 <= gpu_count:

        common_image = sitk.ReadImage(os.path.join(base_dir, "mr.nii.gz"))
        
        if os.environ.get("DATA_SET") == "S":
            start_indices_2 = [(0, 0, 0), (int(common_image.GetWidth() / 2) - 8, 0, 0)]
            cropped_size_2 = (int(common_image.GetWidth()/2)+8, common_image.GetHeight(), common_image.GetDepth())
        
        else:
            start_indices_2 = [(0, 100, 0), (int(common_image.GetWidth() / 2) - 8, 100, 0)]
            cropped_size_2 = (int(common_image.GetWidth()/2)+8, int(common_image.GetHeight()/2), common_image.GetDepth())

        position_suffix_2 = ["r", "l"]

        task = [(input_file, cropped_size_2, start_indices_2, position_suffix_2) for input_file in input_files]

        with Pool() as pool:
            pool.starmap(process_input_file, task)
   
    if 4 <= gpu_count:

        common_image_2 = sitk.ReadImage(os.path.join(base_dir, "mr_l.nii.gz"))

        cropped_size_4 = (common_image_2.GetWidth(), int(common_image_2.GetHeight()/2)+8, common_image_2.GetDepth())
        start_indices_4 = [(0, 0, 0), (0, int(common_image_2.GetHeight()/2)-8, 0)]
        position_suffix_4 = ["u", "l"]

        input_files_4 = ["mr_r.nii.gz", "mr_l.nii.gz", "mt_r.nii.gz", "mt_l.nii.gz"]
        input_task_4 = [(input_file, cropped_size_4, start_indices_4, position_suffix_4) for input_file in input_files_4]

        with Pool() as pool:
            pool.starmap(process_input_file, input_task_4)

    if 8 <= gpu_count:        

        common_image_3 = sitk.ReadImage(os.path.join(base_dir, "mr_l_l.nii.gz"))

        input_files_8_r = ["mr_r_u.nii.gz", "mr_r_l.nii.gz","mt_r_u.nii.gz", "mt_r_l.nii.gz"]
        start_indices_8_r_e = [(0, 0, 0)]
        start_indices_8_r_i = [(int((common_image_3.GetWidth()-8) / 2) - 8, 0, 0)]
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

'

echo "${python_script}" | python      

if [ "$SLURM_GPUS" -ge 2 ] && [ "$SLURM_GPUS" -lt 4 ]; then
    echo "Running MIR_2P.sh"
    sbatch MIR_2P.sh

elif [ "$SLURM_GPUS" -ge 4 ] && [ "$SLURM_GPUS" -lt 8 ]; then
    echo "Running MIR_4P.sh"
    sbatch MIR_4P.sh       

elif [ "$SLURM_GPUS" -ge 8 ]; then
    echo "Running MIR_8P.sh"
    sbatch MIR_8P.sh  

fi

