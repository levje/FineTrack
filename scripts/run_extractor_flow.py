from FineTrack.filterers.extractor.extractor_filterer import ExtractorFilterer
from FineTrack.utils.logging import get_logger, setup_logging, add_logging_args
from argparse import Namespace, ArgumentParser

args = Namespace(log_file=None, log_level='INFO')
setup_logging(args)
LOGGER = get_logger(__name__)

def parse_args():
    parser = ArgumentParser(description="Run the extractor flow pipeline.")
    parser.add_argument("--root_dir", required=True, type=str, help="The root directory of the input data.")
    parser.add_argument("--out_dir", required=True, type=str, help="The output directory.")
    parser.add_argument("--quick_registration", action="store_true", help="Use quick registration.")
    parser.add_argument("--keep_intermediate_steps", action="store_true", help="Keep intermediate steps.")
    add_logging_args(parser)
    return parser.parse_args()

def main():
    LOGGER.info("Running the extractor flow pipeline...")
    args = parse_args()
    # root_dir = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/hcp/extractor/input"
    # out_dir = "/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/hcp/extractor/output"

    extractor = ExtractorFilterer(end_space="orig",
                                keep_intermediate_steps=False,
                                quick_registration=False)
    extractor(args.root_dir, [], out_dir=args.out_dir)

    print("Done.")

if __name__ == "__main__":
    main()
