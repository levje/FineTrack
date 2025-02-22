#!/usr/bin/env nextflow

params.input = false
params.help = false
params.debug = true


if(params.help) {
    usage = file("$baseDir/USAGE")
    cpu_count = Runtime.runtime.availableProcessors()
    bindings = ["rois_folder":"$params.rois_folder",
                "FLF": "$params.FLF",
                "run_bet":"$params.run_bet",
                "distance": "$params.distance",
                "orig":"$params.orig",
                "extended":"$params.extended",
                "keep_intermediate_steps":"$params.keep_intermediate_steps",
                "quick_registration": "$params.quick_registration",
                "cpu_count":"$cpu_count",
                "processes_bet_register_t1":"$params.processes_bet_register_t1",
                "processes_major_filtering":"$params.processes_major_filtering"]  

    engine = new groovy.text.SimpleTemplateEngine()
    template = engine.createTemplate(usage.text).make(bindings)
    print template.toString()
    return
    }

log.info "Extractor_flow pipeline"
log.info "==================="
log.info "Start time: $workflow.start"
log.info ""

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}

if (!params.keep_intermediate_steps) {
  log.info "Warning: You won't be able to resume your processing if you don't use the option --keep_intermediate_steps"
  log.info ""
}

if (params.input){
    log.info "Input: $params.input"
    root = file(params.input)

    Channel
    .fromPath("$root/**/*_T1.nii.gz",
              maxDepth:1)
             .map{[it.parent.name, it]}
             .into{t1s_for_register;
                   t1s_for_register_back;
                   t1s_for_copy_to_orig;
                   check_t1s;
                   t1s_empty}
}
else {
    error "Error ~ Please use --input for the input data."
}

check_t1s.count().into{number_t1s_for_compare; number_t1s_check_with_orig}


if (params.orig){
    number_t1s_check_with_orig
      .subscribe{a -> if (a == 0)
      error "Error ~ You cannot use --orig without having any T1w in the orig space."}
}
    
number_t1s_for_compare
    .subscribe { a -> if (a == 0)
    error "Error ~ No T1 images found for comparison." }

/* BEGINNING TRANSFO */

process Register_T1 {
    publishDir = params.final_output_mni_space
    cpus params.processes_bet_register_t1

    input:
    set sid, file(t1) from t1s_for_register

    output:
    set sid, "${sid}__output0GenericAffine.mat", "${sid}__output1InverseWarp.nii.gz", "${sid}__output1Warp.nii.gz" into transformation_for_trk
    file "${sid}__t1_${params.template_space}.nii.gz"
    file "${sid}__t1_bet_mask.nii.gz" optional true
    file "${sid}__t1_bet.nii.gz" optional true

    script:
    if (params.run_bet){
    """
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export ANTS_RANDOM_SEED=1234

        antsBrainExtraction.sh -d 3 -a $t1 -e $params.template_t1/t1_template.nii.gz\
            -o bet/ -m $params.template_t1/t1_brain_probability_map.nii.gz -u 0
        scil_image_math.py convert bet/BrainExtractionMask.nii.gz ${sid}__t1_bet_mask.nii.gz --data_type uint8
        scil_image_math.py multiplication $t1 ${sid}__t1_bet_mask.nii.gz ${sid}__t1_bet.nii.gz

        ${params.registration_script} -d 3 -m ${sid}__t1_bet.nii.gz -f ${params.rois_folder}${params.atlas.template} -n ${task.cpus} -o "${sid}__output" -t s
        mv ${sid}__outputWarped.nii.gz ${sid}__t1_${params.template_space}.nii.gz
    """
    }
    else{
    """
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export ANTS_RANDOM_SEED=1234

        ${params.registration_script} -d 3 -m ${t1} -f ${params.rois_folder}${params.atlas.template} -n ${task.cpus} -o "${sid}__output" -t s
        mv ${sid}__outputWarped.nii.gz ${sid}__t1_${params.template_space}.nii.gz
    """
    }
}