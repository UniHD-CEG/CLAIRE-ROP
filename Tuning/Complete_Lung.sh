#!/bin/bash

#SBATCH --mem=50gb
#SBATCH --time=00:60:00
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated
#SBATCH -e stderr.e
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --job-name=CLAIRE
#SBATCH --output=%j.out

#module load  compiler/gnu/10
#module load  mpi/openmpi/4.0 

TS=/home/hk-project-irmulti/hd_fa163/TotalSegmentator/bin
source /home/hk-project-irmulti/hd_fa163/Claire/claire/deps/env_source.sh

betacont_values=("5e-2" "1e-2" "5e-3" "1e-3")


DATA=/home/hk-project-irmulti/hd_fa163/Dataset/Lung/MIR/C9/Complete_Lung

export DATA

##Moving image before registration
python3.8 $TS/TotalSegmentator -i $DATA/mt.nii.gz -o $DATA/mask_mt
python3.8 $TS/totalseg_combine_masks -i $DATA/mask_mt -o $DATA/mask_mt.nii.gz -m lung
##Reference image (mr)
python3.8 $TS/TotalSegmentator -i $DATA/mr.nii.gz -o $DATA/mask_mr
python3.8 $TS/totalseg_combine_masks -i $DATA/mask_mr -o $DATA/mask_mr.nii.gz -m lung

for betacont in "${betacont_values[@]}"; do

    #beta="${betacont_values[betacont]}"

    output="Complete_$betacont"
    mpirun ./claire -mt $DATA/mt.nii.gz -mr $DATA/mr.nii.gz -mask $DATA/mask_mr.nii.gz  -regnorm h1s-div -maxit 50 -krylovmaxit 100 -precond invreg  -iporder 1  -betacont $betacont  -beta-div 1e-04 -diffpde finite -verbosity 2   -x  "$DATA/${output}_"   -defmap  

    log_file="Complete_$betacont.log"
    mpirun ./claire -mt $DATA/mt.nii.gz -mr $DATA/mr.nii.gz -mask $DATA/mask_mr.nii.gz   -regnorm h1s-div -maxit 50 -krylovmaxit 100 -precond invreg  -iporder 1  -betacont $betacont -beta-div 1e-04 -diffpde finite   -verbosity 2     &> "$DATA/$log_file" 2>&1

    python_script_1=$(cat <<EOF
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
        "ifile":  os.path.join(data_path,"mt.nii.gz"),   
        "xfile": os.path.join(data_path,"mt_deformed_w_defmap.nii.gz"),  
        "odir": os.path.join(data_path),
        "out": "result"
    })

    ext = ".nii.gz"
    print("loading images")
    odir = args.odir
    

    def1,_ = readImage(os.path.join(odir,"${output}_deformation-map-x1{}".format(ext)))
    def2,_ = readImage(os.path.join(odir,"${output}_deformation-map-x2{}".format(ext)))
    def3,_ = readImage(os.path.join(odir,"${output}_deformation-map-x3{}".format(ext)))   

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
    echo "${python_script_1}" | python     

    python3.8 $TS/TotalSegmentator -i $DATA/mt_deformed_w_defmap.nii.gz  -o $DATA/mask_mt_deformed_w_defmap
    python3.8 $TS/totalseg_combine_masks -i $DATA/mask_mt_deformed_w_defmap -o $DATA/mask_mt_deformed_w_defmap.nii.gz -m lung

    python_script_2='
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
mask_deformed = sitk.ReadImage(os.path.join(data_path, "mask_mt_deformed_w_defmap.nii.gz"))
mask_deformed.SetOrigin(mask.GetOrigin())

#Before and after
segmentations = [mask, mask_deformed]

class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)

class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)
    
# Empty numpy arrays to hold the results 
overlap_results = np.zeros((len(segmentations),len(OverlapMeasures.__members__.items())))  
surface_distance_results = np.zeros((len(segmentations),len(SurfaceDistanceMeasures.__members__.items())))  

# Compute the evaluation criteria

# Note that for the overlap measures filter, because we are dealing with a single label we 
# use the combined, all labels, evaluation measures without passing a specific label to the methods.
overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

# Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside 
# relationship, is irrelevant)
label = 1
reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False))
reference_surface = sitk.LabelContour(reference_segmentation)

statistics_image_filter = sitk.StatisticsImageFilter()
# Get the number of pixels in the reference surface by counting all pixels that are 1.
statistics_image_filter.Execute(reference_surface)
num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 

for i, seg in enumerate(segmentations):
    # Overlap measures
    overlap_measures_filter.Execute(reference_segmentation, seg)
    #overlap_results[i,OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
    overlap_results[i,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    #overlap_results[i,OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
    #overlap_results[i,OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
    #overlap_results[i,OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()
    
    # Hausdorff distance
    #hausdorff_distance_filter.Execute(reference_segmentation, seg)
    #surface_distance_results[i,SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    #segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False))
    #segmented_surface = sitk.LabelContour(seg)
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    #seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    #ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the segmented surface by counting all pixels that are 1.
    #statistics_image_filter.Execute(segmented_surface)
    #num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    #seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    #seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    #seg2ref_distances = seg2ref_distances + \
     #                   list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    #ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    #ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    #ref2seg_distances = ref2seg_distances + \
      #                  list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    #all_surface_distances = seg2ref_distances + ref2seg_distances
    
    #surface_distance_results[i,SurfaceDistanceMeasures.mean_surface_distance.value] = np.mean(all_surface_distances)
    #surface_distance_results[i,SurfaceDistanceMeasures.median_surface_distance.value] = np.median(all_surface_distances)
    #surface_distance_results[i,SurfaceDistanceMeasures.std_surface_distance.value] = np.std(all_surface_distances)
    #surface_distance_results[i,SurfaceDistanceMeasures.max_surface_distance.value] = np.max(all_surface_distances)

print ("Overlap Results (Dice) before and after:" ,overlap_results)    
'

    echo "${python_script_2}" | python > $DATA/overlap_"$betacont"
done
