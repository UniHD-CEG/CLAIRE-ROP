#!/bin/bash

#SBATCH --mem=50gb
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1     
#SBATCH --partition=accelerated
#SBATCH -e stderr.e
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=CLAIRE-ROP
#SBATCH --output=%j.out

#module load  compiler/gnu/10
#module load  mpi/openmpi/4.0


DATA=/home/hk-project-irmulti/hd_fa163/Dataset/Lung/MIR/C1/128x64x94_8P/            # Path to dataset
DATA_MASK=/home/hk-project-irmulti/hd_fa163/Dataset/Lung/MIR/C1/128x64x94_8P/Masks  # Path to masks
export DATA
export DATA_MASK
export SLURM_GPUS=8       #Set SLURM_GPUS to the desired GPU count
export DATA_SET=L         #Set DATA_SET to small (S) or large (L)
export OVERLAP_VALUE=8    #Set OVERLAP_VALUE (default is 8)

#TS=/home/hk-project-irmulti/hd_fa163/TotalSegmentator/bin
echo "SLURM_GPUS set to $SLURM_GPUS GPUs."

#Partitioning
run_section_1=0
#Mask Partitioning & Calculate lung percentage 
run_section_2=1
#Proceed to MIR
run_section_3=0

if [ "$run_section_1" -eq "1" ]; then
   echo "Partitioning"

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
overlap_value = int(os.environ["OVERLAP_VALUE"])
   
if os.environ.get("SLURM_GPUS"):
    gpu_count = int(os.environ.get("SLURM_GPUS"))
    print(f"Proceed with {gpu_count} partitions")

    if 2 <= gpu_count:

        common_image = sitk.ReadImage(os.path.join(base_dir, "mr.nii.gz"))
        
        if os.environ.get("DATA_SET") == "S":
            start_indices_2 = [(0, 0, 0), (int(common_image.GetWidth() / 2) - overlap_value, 0, 0)]
            cropped_size_2 = (int(common_image.GetWidth()/2)+overlap_value, common_image.GetHeight(), common_image.GetDepth())
        
        else:
            start_indices_2 = [(0, 100, 0), (int(common_image.GetWidth() / 2) - overlap_value, 100, 0)]
            cropped_size_2 = (int(common_image.GetWidth()/2)+overlap_value, int(common_image.GetHeight()/2), common_image.GetDepth())

        position_suffix_2 = ["r", "l"]

        task = [(input_file, cropped_size_2, start_indices_2, position_suffix_2) for input_file in input_files]

        with Pool() as pool:
            pool.starmap(process_input_file, task)
   
    if 4 <= gpu_count:

        common_image_2 = sitk.ReadImage(os.path.join(base_dir, "mr_l.nii.gz"))
    
        cropped_size_4 = (common_image_2.GetWidth(), int(common_image_2.GetHeight()/2)+overlap_value, common_image_2.GetDepth())
        start_indices_4 = [(0, 0, 0), (0, int(common_image_2.GetHeight()/2)-overlap_value, 0)]
        position_suffix_4 = ["u", "l"]

        input_files_4 = ["mr_r.nii.gz", "mr_l.nii.gz", "mt_r.nii.gz", "mt_l.nii.gz"]
        input_task_4 = [(input_file, cropped_size_4, start_indices_4, position_suffix_4) for input_file in input_files_4]

        with Pool() as pool:
            pool.starmap(process_input_file, input_task_4)

    if 8 <= gpu_count:        

        common_image_3 = sitk.ReadImage(os.path.join(base_dir, "mr_l_l.nii.gz"))

        input_files_8_r = ["mr_r_u.nii.gz", "mr_r_l.nii.gz","mt_r_u.nii.gz", "mt_r_l.nii.gz"]
        start_indices_8_r_e = [(0, 0, 0)]
        start_indices_8_r_i = [(int((common_image_3.GetWidth()-overlap_value) / 2) - overlap_value, 0, 0)]
        input_files_8_l = ["mr_l_u.nii.gz", "mr_l_l.nii.gz","mt_l_u.nii.gz", "mt_l_l.nii.gz"]
        start_indices_8_l_i = [(0, 0, 0)]
        start_indices_8_l_e = [(int((common_image_3.GetWidth()-overlap_value) / 2) , 0, 0)]


        cropped_size_8_i = (int((common_image_3.GetWidth()-overlap_value)/2)+ (2*overlap_value), common_image_3.GetHeight(), common_image_3.GetDepth())
        cropped_size_8_e = (int((common_image_3.GetWidth()-overlap_value)/2)+overlap_value, common_image_3.GetHeight(), common_image_3.GetDepth())
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

    if 16 <= gpu_count:

        common_image_4 = sitk.ReadImage(os.path.join(base_dir, "mr_l_l_e.nii.gz"))

        input_files_16_e = ["mr_r_u_e.nii.gz", "mr_r_l_e.nii.gz", "mr_l_u_e.nii.gz", "mr_l_l_e.nii.gz",
                            "mt_r_u_e.nii.gz", "mt_r_l_e.nii.gz", "mt_l_u_e.nii.gz", "mt_l_l_e.nii.gz"]

        input_files_16_i = ["mr_r_u_i.nii.gz", "mr_r_l_i.nii.gz", "mr_l_u_i.nii.gz", "mr_l_l_i.nii.gz",
                            "mt_r_u_i.nii.gz", "mt_r_l_i.nii.gz", "mt_l_u_i.nii.gz", "mt_l_l_i.nii.gz"]

        cropped_size_16_e = (common_image_4.GetWidth(), int(common_image_4.GetHeight()/2)+overlap_value, common_image_4.GetDepth())
        cropped_size_16_i = (common_image_4.GetWidth()+overlap_value, int(common_image_4.GetHeight()/2)+overlap_value, common_image_4.GetDepth())

        start_indices_16 = [(0, 0, 0), (0, int(common_image_4.GetHeight()/2)-overlap_value, 0)]

        position_suffix_16 = ["u", "l"]

        input_task_16_e = [(input_file, cropped_size_16_e, start_indices_16, position_suffix_16) for input_file in input_files_16_e]
        input_task_16_i = [(input_file, cropped_size_16_i, start_indices_16, position_suffix_16) for input_file in input_files_16_i]

        with Pool() as pool:
            pool.starmap(process_input_file, input_task_16_e)
            pool.starmap(process_input_file, input_task_16_i)
'
echo "${python_script}" | python    
fi


if [ "$run_section_2" -eq "1" ]; then
    echo "Mask Partitioning & Calculate Lung Percentage"

    #python3.8 $TS/TotalSegmentator -i $DATA/mt.nii.gz -o $DATA/mask_mt
    #python3.8 $TS/totalseg_combine_masks -i $DATA/mask_mt -o $DATA/mask_mt.nii.gz -m lung
    result=0

    python_script_1=$(cat <<EOF
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

        #start = time.time()
        cropped_image = extract.Execute(input_image)
        #end = time.time()
        #print("Partitioning time:")
        #print(end - start) 

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
    #print(f"Input image: {input_file} | Size: {input_size}")

    output_filenames = crop_and_save(input_file, input_image, start_indices, cropped_size, position_suffix)
    return output_filenames

def calculate_lung_percentage(input_file):
    input_path = os.path.join(output_dir, input_file)
    input_image = sitk.ReadImage(input_path)
    
    #start = time.time()
    # Convert to a numpy array
    mask_array = sitk.GetArrayFromImage(input_image)
    
    # Calculate total number of pixels
    total_pixels = mask_array.size
    
    # Calculate the number of pixels in the lung (values of 1 in the mask)
    lung_pixels = mask_array.sum()
    
    # Calculate percentage of lung area
    lung_percentage = (lung_pixels / total_pixels) * 100
    #end = time.time()
    #print("Calculating Lung Content time:")
    #print(end - start) 
    
    #Uncomment the line below if you want to see the percentage of lung content in each edge partition
    print(f"Lung percentage for {input_file}: {lung_percentage:.2f}%")

    # Determine result
    return 1 if lung_percentage > 0 else 0
        
base_dir = os.environ["DATA_MASK"]
output_dir = os.environ["DATA_MASK"]
input_files = ["mask_mt.nii.gz"]
overlap_value = int(os.environ["OVERLAP_VALUE"])
   
if os.environ.get("SLURM_GPUS"):
    gpu_count = int(os.environ.get("SLURM_GPUS"))
    #print(f"Proceed with {gpu_count} partitions")

    if 2 <= gpu_count:

        common_image = sitk.ReadImage(os.path.join(base_dir, "mask_mt.nii.gz"))
        
        if os.environ.get("DATA_SET") == "S":
            start_indices_2 = [(0, 0, 0), (int(common_image.GetWidth() / 2) - overlap_value, 0, 0)]
            cropped_size_2 = (int(common_image.GetWidth()/2)+overlap_value, common_image.GetHeight(), common_image.GetDepth())
        
        else:
            start_indices_2 = [(0, 100, 0), (int(common_image.GetWidth() / 2) - overlap_value, 100, 0)]
            cropped_size_2 = (int(common_image.GetWidth()/2)+overlap_value, int(common_image.GetHeight()/2), common_image.GetDepth())

        position_suffix_2 = ["r", "l"]

        task = [(input_file, cropped_size_2, start_indices_2, position_suffix_2) for input_file in input_files]

        with Pool() as pool:
            pool.starmap(process_input_file, task)
   
    if 4 <= gpu_count:

        common_image_2 = sitk.ReadImage(os.path.join(base_dir, "mask_mt_l.nii.gz"))
    
        cropped_size_4 = (common_image_2.GetWidth(), int(common_image_2.GetHeight()/2)+overlap_value, common_image_2.GetDepth())
        start_indices_4 = [(0, 0, 0), (0, int(common_image_2.GetHeight()/2)-overlap_value, 0)]
        position_suffix_4 = ["u", "l"]

        input_files_4 = ["mask_mt_r.nii.gz", "mask_mt_l.nii.gz"]
        input_task_4 = [(input_file, cropped_size_4, start_indices_4, position_suffix_4) for input_file in input_files_4]

        with Pool() as pool:
            pool.starmap(process_input_file, input_task_4)

    if 8 <= gpu_count:        

        common_image_3 = sitk.ReadImage(os.path.join(base_dir, "mask_mt_l_l.nii.gz"))

        input_files_8_r = ["mask_mt_r_u.nii.gz", "mask_mt_r_l.nii.gz"]
        start_indices_8_r_e = [(0, 0, 0)]
        start_indices_8_r_i = [(int((common_image_3.GetWidth()-overlap_value) / 2) - overlap_value, 0, 0)]
        input_files_8_l = ["mask_mt_l_u.nii.gz", "mask_mt_l_l.nii.gz"]
        start_indices_8_l_i = [(0, 0, 0)]
        start_indices_8_l_e = [(int((common_image_3.GetWidth()-overlap_value) / 2) , 0, 0)]


        cropped_size_8_i = (int((common_image_3.GetWidth()-overlap_value)/2)+ (2*overlap_value), common_image_3.GetHeight(), common_image_3.GetDepth())
        cropped_size_8_e = (int((common_image_3.GetWidth()-overlap_value)/2)+overlap_value, common_image_3.GetHeight(), common_image_3.GetDepth())
        position_suffix_8_i = ["i"]
        position_suffix_8_e = ["e"]

        input_task_8_r_i = [(input_file, cropped_size_8_i, start_indices_8_r_i, position_suffix_8_i) for input_file in input_files_8_r]
        input_task_8_r_e = [(input_file, cropped_size_8_e, start_indices_8_r_e, position_suffix_8_e) for input_file in input_files_8_r]

        input_task_8_l_i = [(input_file, cropped_size_8_i, start_indices_8_l_i, position_suffix_8_i) for input_file in input_files_8_l]
        input_task_8_l_e = [(input_file, cropped_size_8_e, start_indices_8_l_e, position_suffix_8_e) for input_file in input_files_8_l]


        with Pool() as pool:
            output_files_r_i = pool.starmap(process_input_file, input_task_8_r_i)
            output_files_r_e = pool.starmap(process_input_file, input_task_8_r_e)
            output_files_l_e = pool.starmap(process_input_file, input_task_8_l_i)
            output_files_l_e = pool.starmap(process_input_file, input_task_8_l_e)
        
        output_files = [file for sublist in output_files_r_e + output_files_l_e for file in sublist]
        
        with Pool() as pool:
          lung_results =  pool.map(calculate_lung_percentage, output_files)
        
        # Determine final result
        result = 1 if any(lung_results) else 0
        print(result)     # Ensure this line is uncommented and that there are no other print statements.
if 16 <= gpu_count:

        common_image_4 = sitk.ReadImage(os.path.join(base_dir, "mask_mt_l_l_e.nii.gz"))

        input_files_16_e = ["mask_mt_r_u_e.nii.gz", "mask_mt_r_l_e.nii.gz", "mask_mt_l_u_e.nii.gz", "mask_mt_l_l_e.nii.gz"]

        input_files_16_i = ["mask_mt_r_u_i.nii.gz", "mask_mt_r_l_i.nii.gz", "mask_mt_l_u_i.nii.gz", "mask_mt_l_l_i.nii.gz"]

        cropped_size_16_e = (common_image_4.GetWidth(), int(common_image_4.GetHeight()/2)+overlap_value, common_image_4.GetDepth())
        cropped_size_16_i = (common_image_4.GetWidth()+overlap_value, int(common_image_4.GetHeight()/2)+overlap_value, common_image_4.GetDepth())

        start_indices_16 = [(0, 0, 0), (0, int(common_image_4.GetHeight()/2)-overlap_value, 0)]

        position_suffix_16 = ["u", "l"]

        input_task_16_e = [(input_file, cropped_size_16_e, start_indices_16, position_suffix_16) for input_file in input_files_16_e]
        input_task_16_i = [(input_file, cropped_size_16_i, start_indices_16, position_suffix_16) for input_file in input_files_16_i]

        with Pool() as pool:
            pool.starmap(process_input_file, input_task_16_e)
            pool.starmap(process_input_file, input_task_16_i)
            pool.starmap(calculate_lung_percentage, input_task_16_e)  

EOF
)

    result=$(echo "${python_script_1}" | python)
    echo "Result: ${result}"


    if [[ "$result" -eq 1 ]]; then
        echo "Non-zero lung content detected."
    elif [[ "$result" -eq 0 ]]; then
        echo "All lung percentages are zero."
    fi
fi  


if [ "$run_section_3" -eq "1" ]; then
   echo "Proceed to MIR"

    if [ "$SLURM_GPUS" -ge 2 ] && [ "$SLURM_GPUS" -lt 4 ]; then
        echo "Running MIR_2P.sh"
        sbatch MIR_2P.sh

    elif [ "$SLURM_GPUS" -ge 4 ] && [ "$SLURM_GPUS" -lt 8 ]; then
        echo "Running MIR_4P.sh"
        sbatch MIR_4P.sh       

    elif [ "$SLURM_GPUS" -ge 8 ] && [ "$SLURM_GPUS" -lt 16 ]; then
        echo "Running MIR_8P.sh"
        echo "Running MIR_8P.sh with result: $result"
        sbatch MIR_8P.sh  "$result"

    elif [ "$SLURM_GPUS" -ge 16 ]; then
        echo "Running MIR_16P.sh"
        sbatch MIR_16P.sh      

    fi

fi

