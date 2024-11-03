#!/bin/bash
set -e

#T1=data/datasets/ismrm2015_2mm/anat/ismrm2015_T1.nii.gz
T1=data/datasets/TractoInferno/derivatives/testset/sub-1006/anat/sub-1006__T1w.nii.gz
output_dir=fsl_tissues_tractoinferno
#subject_id=ismrm2015
subject_id=tractoinferno
mkdir -p ${output_dir}

# Run FSL's FAST to segment the T1 image into GM, WM, and CSF
fast \
	-t 1 \
	-n 3 \
	-H 0.1 \
	-I 4 \
	-l 20.0 \
	-g \
	-b \
	-B \
	-o ${output_dir}/t1.nii.gz \
	--verbose \
	${T1}

scil_volume_math.py convert ${output_dir}/t1_seg_2.nii.gz ${output_dir}/${subject_id}_mask_wm.nii.gz --data_type uint8 -f

scil_volume_math.py convert ${output_dir}/t1_seg_1.nii.gz ${output_dir}/${subject_id}_mask_gm.nii.gz --data_type uint8 -f
scil_volume_math.py convert ${output_dir}/t1_seg_0.nii.gz ${output_dir}/${subject_id}_mask_csf.nii.gz --data_type uint8 -f

mv ${output_dir}/t1_pve_2.nii.gz ${output_dir}/${subject_id}_map_wm.nii.gz
mv ${output_dir}/t1_pve_1.nii.gz ${output_dir}/${subject_id}_map_gm.nii.gz
mv ${output_dir}/t1_pve_0.nii.gz ${output_dir}/${subject_id}_map_csf.nii.gz

echo "Tissue segmentation done!, results in ${output_dir}"


