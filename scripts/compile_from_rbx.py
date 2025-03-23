import argparse
import json
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tractogram_in', type=str, help='The tractogram file to split')
    parser.add_argument('results_json', type=str, help='The results json file containing the recognized indices from RBX')
    parser.add_argument('tractogram_out_recognized', type=str, help='The tractogram file to save the recognized streamlines')
    parser.add_argument('tractogram_out_unrecognized', type=str, help='The tractogram file to save the unrecognized streamlines')
    parser.add_argument('--reference', type=str, default="same", help='The reference tractogram to use for saving the recognized and unrecognized streamlines')
    return parser.parse_args()

def main():
    args = parse_args()
    tractogram_in = args.tractogram_in
    results_json = args.results_json
    tractogram_out_recognized = args.tractogram_out_recognized
    tractogram_out_unrecognized = args.tractogram_out_unrecognized
    reference = args.reference

    with open(results_json, 'r') as f:
        results = json.load(f)

    all_recognized_indices = set()
    for k in results.keys():
        for i in results[k]['indices']:
            all_recognized_indices.add(i)
    
    all_recognized_indices = list(all_recognized_indices)
    all_recognized_indices.sort()
    all_recognized_indices = np.array(all_recognized_indices, dtype=int)

    unrecognized_indices = np.setdiff1d(np.arange(len(sft)), all_recognized_indices)

    assert len(all_recognized_indices) + len(unrecognized_indices) == len(sft)

    sft = load_tractogram(tractogram_in, reference)
    recognized_sft = sft[all_recognized_indices]
    unrecognized_sft = sft[unrecognized_indices]

    save_tractogram(recognized_sft, tractogram_out_recognized, reference)
    print("Recognized streamlines saved to: ", tractogram_out_recognized)
    save_tractogram(unrecognized_sft, tractogram_out_unrecognized, reference)
    print("Unrecognized streamlines saved to: ", tractogram_out_unrecognized)

    print("Done!")

if __name__ == '__main__':
    main()