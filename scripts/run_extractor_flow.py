from FineTrack.filterers.extractor.extractor_filterer import ExtractorFilterer
from FineTrack.utils.logging import get_logger, setup_logging
from argparse import Namespace

args = Namespace(log_file=None, log_level='INFO')
setup_logging(args)
LOGGER = get_logger(__name__)

LOGGER.info("Running the extractor flow pipeline...")

root_dir = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/hcp/extractor/input"
out_dir = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/hcp/extractor/output"

extractor = ExtractorFilterer(end_space="orig",
                              keep_intermediate_steps=False,
                              quick_registration=False)
extractor(root_dir, [], out_dir=out_dir)

print("Done.")
