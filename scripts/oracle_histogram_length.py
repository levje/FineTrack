
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import length
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import argparse
from FineTrack.oracles.oracle import OracleSingleton
from FineTrack.oracles.transformer_oracle import TransformerOracle
from FineTrack.utils.torch_utils import get_device_str
from FineTrack.algorithms.shared.utils import (
    add_item_to_means, mean_losses)
from FineTrack.trainers.oracle.oracle_trainer import to_device
import seaborn as sns
import os
import h5py
import torch
from tqdm import tqdm
from collections import defaultdict
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import set_number_of_points
from FineTrack.environments.utils import resample_streamlines_if_needed

sns.set_style("darkgrid")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot histogram of lengths of streamlines")
    parser.add_argument("checkpoint", help="Path to the checkpoint of the oracle.")
    parser.add_argument("tractograms", nargs="+", help="Glob pattern to tractograms.")
    parser.add_argument("--reference", help="Reference tractogram to get the affine matrix.")
    parser.add_argument("--score_override", type=int, help="Override the score of the streamlines.")
    parser.add_argument("--sup_title", help="Choose between, ['all', 'positive', 'negative'].",
                        choices=['all', 'positive', 'negative'], default='all')
    parser.add_argument("--bin_width", type=int, help="Width of the bins in millimeters.", default=2)
    parser.add_argument("--save_dir", type=str, help="Directory to save the plot.", default=".")
    args = parser.parse_args()
    return args

def plot_length_freq_histogram(tractograms, bins, ax=None, reference=None, filter_for_score=None):
    lengths = []
    for tractogram in tractograms:
        sft = load_sft(tractogram, reference, filter_for_score)
        sft.to_rasmm()
        sft.to_corner()

        sft_lengths = list(length(sft.streamlines))
        lengths.extend(sft_lengths)

    if ax is None:
        fig, ax = plt.subplots(1)

    ax.hist(lengths, bins=bins)
    ax.set_title("Histogram of lengths of streamlines")
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Frequency")
    
    if ax is None:
        plt.savefig("histogram_of_lengths.png")

# def test_step_wtf(model, test_batch, batch_idx):
#     x, y = test_batch
#     with torch.no_grad():
#         with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
#             y_hat = model(x)
#             y_int = torch.round(y)

#     preds = (y_hat > 0.5).int()

#     is_correct = (preds == y_int).cpu().numpy()
#     indices_not_correct = np.arange(is_correct.shape[0])[~is_correct]
#     nb_corrects = (preds == y_int).sum().item()
#     nb_total = y_int.size(0)
#     hand_computed_acc = nb_corrects / nb_total
#     print("wtf hand_computed_acc: {} ({}/{})".format(hand_computed_acc, nb_corrects, nb_total))
#     return hand_computed_acc, preds

# def streamlines_to_numpy(streamlines, nb_points):
#     if isinstance(streamlines, np.ndarray):
#         return streamlines
#     elif isinstance(streamlines, list):
#         return np.array(streamlines)
#     elif isinstance(streamlines, ArraySequence):
#         return np.array(set_number_of_points(streamlines, nb_points))
#     else:
#         raise ValueError("Unsupported type for streamlines.")
        

def predict(streamlines, scores, model: TransformerOracle):
    model.eval()  # Set model to evaluation mode
    model.to(get_device_str())
    streamlines = resample_streamlines_if_needed(streamlines, (model.input_size // 3) + 1)
    dirs = np.diff(streamlines, axis=1)

    batch_size = 2048
    preds = np.zeros(len(streamlines))
    exp_np_loops = len(streamlines) // batch_size
    for i, batch in enumerate(tqdm(range(exp_np_loops), desc="testing oracle")):
        start = i * batch_size
        end = min(start + batch_size, len(dirs))

        batch = torch.from_numpy(dirs[start:end])
        batch = to_device(batch, get_device_str())
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(batch)
                predictions = (logits > 0.5).int().cpu().numpy()
        
        preds[start:end] = predictions

    # print("Mean accuracy: ", np.mean(preds == scores))

    return preds

def load_model(checkpoint: str, singleton=False):
    if singleton:
        model = OracleSingleton(checkpoint, device="cuda")
    else:
        model = TransformerOracle.load_from_checkpoint(torch.load(checkpoint))
    return model

def load_sft(tractogram, reference, filter_for_score=1):
    assert len(tractogram) > 4, "Tractogram should have a valid extension."

    _, file_extension = os.path.splitext(tractogram)

    if file_extension == ".trk":
        sft = load_tractogram(tractogram, 'same' if reference is None else reference)
    elif file_extension == ".tck":
        assert reference is not None and os.path.exists(reference), \
            "Reference should be provided for tck files."
        sft = load_tractogram(tractogram, reference)
    elif file_extension == ".hdf5":
        assert reference is not None and os.path.exists(reference), \
            "Reference should be provided for hdf5 files."
        with h5py.File(tractogram, 'r') as f:
            group = f["test"] if "test" in f else f["train"]
            streamlines = group["data"][:]
            scores = group["scores"][:]

            if filter_for_score is not None:
                print("Nb streamlines before filtering: ", len(streamlines))
                is_score = scores == filter_for_score
                streamlines = streamlines[is_score]
                scores = scores[is_score]
                print("Nb streamlines after filtering: ", len(streamlines))

            sft = StatefulTractogram(streamlines, reference, Space.VOX, origin=Origin.TRACKVIS, data_per_streamline={"score": scores})
    else:
        raise ValueError("Unsupported extension for tractogram.")
    
    assert isinstance(sft, StatefulTractogram), \
        "There was an error loading the tractogram."
    return sft

def plot_length_acc_histogram(model: OracleSingleton, tractograms,
                              bins, ax=None, reference=None,
                              score_override=None, bin_width=2,
                              filter_for_score=None):
    if ax is None:
        fig, ax = plt.subplots(1)

    accuracies = np.zeros(len(bins) - 1)
    lengths = np.zeros(len(bins) - 1, dtype=int)
    min_length, max_length = np.inf, -np.inf
    for tractogram in tractograms:
        sft = load_sft(tractogram, reference, filter_for_score)

        print("Number of streamlines: ", len(sft.streamlines))
        
        if len(sft.streamlines) == 0:
            continue

        if score_override is not None:
            scores = np.ones(len(sft.streamlines)) * score_override
        else:
            scores = np.array(sft.data_per_streamline["score"]).squeeze(1)
        
        #################################
        # VOX space: Predict the scores using the oracle, to get it's accuracy
        # later on below.
        #################################
        sft.to_vox()
        sft.to_corner()
        
        predictions = model.predict(sft.streamlines, prefetch_streamlines=False)
        predictions = (predictions > 0.5).astype(np.uint8)

        #################################
        # RASMM space: Compute the lengths
        #################################
        sft.to_rasmm()
        sft.to_corner()

        # Count the number of each bin we have first
        sft_lengths = list(length(sft.streamlines))
        min_length = min(min_length, np.min(sft_lengths))
        max_length = max(max_length, np.max(sft_lengths))

        # Only used to recreate the same histogram as in the other function.
        count_per_length, _ = np.histogram(sft_lengths, bins=bins)
        lengths += count_per_length

        # Use the predictions above to get the accuracy.
        t_acc = (predictions == scores).astype(np.uint8)

        for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            within_bin = np.logical_and(sft_lengths >= start, sft_lengths < end)
            accuracies[i] += np.sum(t_acc[within_bin])

    avg_accuracy = np.sum(accuracies) / np.sum(lengths)
    print("Average accuracy: ", avg_accuracy)

    accuracies[lengths > 0] /= lengths[lengths > 0]
    
    ax.set_title("Accuracy of lengths of streamlines (avg_acc: {:.1f}%, min/max lengths: ({:.0f}, {:.0f}))".format(avg_accuracy*100, min_length, max_length))
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Accuracy")
    ax.margins(y=0.1)
    bars = ax.bar(bins[:-1], accuracies, width=bin_width)
    for i, rect in enumerate(bars):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f" {lengths[i]}", ha='center', va='bottom', rotation=90, fontsize=6)
    
    if ax is None:
        plt.savefig("histogram_of_lengths.png")

def main():
    args = parse_args()
    tractograms = args.tractograms
    model = load_model(checkpoint=args.checkpoint, singleton=True)

    # The bins should be 2mm long and the range should be from 0 to 200mm
    bins = np.arange(0, 200, args.bin_width)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    if args.sup_title == 'positive' or args.score_override == 1:
        print("For positive streamlines")
        title = "For positive streamlines"
        postfix = "_pos"
        filter_for = 1
    elif args.sup_title == 'negative' or args.score_override == 0:
        print("For negative streamlines")
        title = "For negative streamlines"
        postfix = "_neg"
        filter_for = 0
    else:
        print("For all streamlines")
        title = "For all streamlines"
        postfix = "_all"
        filter_for = None
    
    fig.suptitle(title)
    
    plot_length_freq_histogram(tractograms, bins, ax=ax[0], reference=args.reference, filter_for_score=filter_for)
    plot_length_acc_histogram(model, tractograms, bins, ax=ax[1], reference=args.reference, score_override=args.score_override, bin_width=args.bin_width, filter_for_score=filter_for)

    fig.savefig(os.path.join(args.save_dir, f"histogram_of_lengths{postfix}.png"))

if __name__ == '__main__':
    main()


