
# Things to note:
# - We need to provide a T1 image for the registration process.
#   The T1 should be in the diffusion space (same space as the TRK files.)
#   The T1 file ends with "_t1.nii.gz".
# - The input directory contains the following:
#     - S1/
#         - S1_trk.trk
#         - S1_t1.nii.gz # Same space as the trk file.
#     - S2/
#         - S2_trk.trk
#         - S2_t1.nii.gz # Same space as the trk file.
#     - ...
# 
# - The output directory will contain the following:
#   final_outputs/
#       - S1/
#           - orig_space/
#               - S1__plausible_orig_space.trk
#               - S1__unplausible_orig_space.trk
#           - mni_space/
#               - S1__output0GenericAffine.mat
#               - S1__output1InverseWarp.nii.gz
#               - S1__output1Warp.nii.gz
#               - S1__plausible_mni_space.trk
#               - S1__unplausible_mni_space.trk
#               - S1__t1_mni_space.nii.gz
#               - sub-100206__tracking_mni_space.trk
#       - S2/
#           - ...
# 
# In order to make the registration process faster, we can use the "--quick_registration" flag.
# Also, what we can do is to calculate the registration for all the subjects once in advance.
# Then, we can transform the space of the trk files to the MNI space using the calculated transformations.
# This way, we won't have to input the T1 for each subject.
#
# How to do this:
# 
# 1. Pass all the subjects through extractor_flow once and retreive the calculated transformations
#    for each subject (XX__output0GenericAffine.mat, XX__output1InverseWarp.nii.gz, XX__output1Warp.nii.gz)
# 2. Use scil_apply_transform_to_tractogram.py to transform the trk files to the MNI space.
#    The command should look like this:
#        scil_apply_transform_to_tractogram.py \
#            <tractogram_to_score>.trk \
#            /extractor_flow/templates_and_ROIs/JHU_MNI_SS_T1_brain_182x218x182_conv_f.nii.gz \
#            <subid>__output0GenericAffine.mat \
#            <out_tractogram>_mni.trk \
#            --remove_invalid \
#            --inverse \
#            --in_deformation XX__output1InverseWarp.nii.gz
# 3. 

# INPUT_DIR=~/Documents/test_extractor/i_extractor_flow
# INPUT_DIR=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/hcp/tractoflow_organized
INPUT_DIR=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/TractoInferno/validset
OUTPUT_DIR=~/Documents/FineTrack/results_extractor_flow


# Get all the subjects' T1 images which are in the INPUT_DIR/<subid>/anat/*_T1.nii.gz
mkdir -p $OUTPUT_DIR/subjects_anats
for subj in $(ls $INPUT_DIR); do
    # Make sure subj is a directory
    if [ ! -d $INPUT_DIR/$subj ]; then
        continue
    fi

    mkdir -p $OUTPUT_DIR/subjects_anats/$subj
    cp $INPUT_DIR/$subj/anat/*_[Tt]1*.nii.gz $OUTPUT_DIR/subjects_anats/$subj/${subj}_T1.nii.gz
done

REGISTERING_INPUT_DIR=$OUTPUT_DIR/subjects_anats

# nextflow ~/Documents/extractor_flow/register_only.nf \
nextflow run ~/Documents/FineTrack/configs/nextflow/register_only.nf \
    --input $REGISTERING_INPUT_DIR \
    -with-docker mrzarfir/extractorflow:latest \
    --quick_registration \
    --orig \
    --output_dir $OUTPUT_DIR
    

# Now, we want to copy the mni_space resulting from the registration process to each subject's directory.
for subj in $(ls $OUTPUT_DIR/final_outputs); do
    # Make sure subj is a directory
    if [ ! -d $OUTPUT_DIR/final_outputs/$subj ]; then
        continue
    fi

    cp -r $OUTPUT_DIR/final_outputs/$subj/mni_space $INPUT_DIR/$subj/
done

exit 0
