import nextflow
import nextflow.io
import tempfile

path_to_atlas = "/home/local/USHERBROOKE/levj1404/Documents/rbx_flow/atlas"  # Fixed
path_to_subjs = "/home/local/USHERBROOKE/levj1404/Documents/rbx_flow/subjs"
config_path = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack/configs/nextflow.config"  # Fixed

# Replace those by tmp_dir
run_dir = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack"
output_path = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack/nf_output_dir"

print("Running pipeline...")
execution = nextflow.run(pipeline_path="scilus/rbx_flow",
                         run_path=run_dir,
                         output_path=output_path,
                         configs=[config_path],
                         params={"atlas_directory": path_to_atlas, "input": path_to_subjs},
                         profiles=['finetrack']
                         )
print("Pipeline finished.")
print("Execution:", execution)
print("return_code:", execution.return_code)
print("stdout:", execution.stdout)
print("stderr:", execution.stderr)