import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from matplotlib.widgets import Slider, RadioButtons
from nibabel.nifti1 import Nifti1Image
from pathlib import Path
from enum import Enum


class View(Enum):
    axiale = 0
    coronale = 1
    sagitale = 2

    def __str__(self):
        if self.value == 0:
            return "Axiale"
        elif self.value == 1:
            return "Coronale"
        else:
            return "Sagitale"

    @staticmethod
    def get_ordered_view_string_list():
        return [str(View.axiale), str(View.coronale), str(View.sagitale)]

    @staticmethod
    def get_view_from_string(view_string: str):
        if view_string == "Axiale":
            return View.axiale
        elif view_string == "Coronale":
            return View.coronale
        else:
            return View.sagitale


def generate_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_name", help="Path of the .nii image to visualize.")
    return parser


def get_image_data_for_view_and_slice(image_fdata: np.ndarray, view: View, slice_number: int, sh_coef: int = 0):
    if view == View.axiale:
        return image_fdata[sh_coef, :, :, slice_number]
    elif view == View.coronale:
        return image_fdata[sh_coef, :, slice_number, :]
    elif view == View.sagitale:
        return image_fdata[sh_coef, slice_number, :, :]
    else:
        raise RuntimeError("4th dimension is not yet supported")


def get_aspect_for_view(image: nib.Nifti1Image, view: View):
    zooms = image.header.get_zooms()
    width = 1
    height = 1
    if view == View.axiale:
        width = zooms[1]
        height = zooms[0]
    elif view == View.coronale:
        width = zooms[2]
        height = zooms[0]
    elif view == View.sagitale:
        width = zooms[2]
        height = zooms[1]

    aspect = 1
    if width > 0 and height > 0:
        aspect = width / height

    print("Computed aspect: ", aspect, ". With voxels of size (X, Y) = (", width, ', ', height, ').')

    return aspect


def display_image(image: nib.Nifti1Image, default_slice=75, save_to=None) -> None:
    current_view = View.sagitale
    current_slice = default_slice
    current_sh_coef = 0

    axcodes = nib.aff2axcodes(image.affine)
    origin = 'lower' if axcodes[0] == 'R' else 'upper'


    image_fdata = image.get_fdata()
    image_shape = image_fdata.shape

    # if len(image_shape) == 4:
    #     image_fdata = np.squeeze(image_fdata, axis=3)

    max_slices = image_shape[current_view.value] - 1

    # print("image_shape: ", image_shape)
    # print("current_view: ", current_view)
    # print("max_slices: ", max_slices)
    # print("zooms: ", image.header.get_zooms())

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.title("NIFTI Visualizer")
    data = get_image_data_for_view_and_slice(image_fdata, current_view, current_slice, current_sh_coef)
    ax_image = ax.imshow(
        data,
        # cmap='gray',
        vmin=np.min(image_fdata),
        vmax=np.max(image_fdata),
        origin=origin,
        aspect=get_aspect_for_view(image, current_view)
    )

    slice_axes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
    slice_slider = Slider(
        ax=slice_axes,
        label="Slice",
        valmin=0,
        valmax=max_slices,
        valinit=current_slice,
        orientation='vertical',
        valstep=1
    )

    # Color bar
    plt.colorbar(ax_image, ax=ax)

    view_axes = fig.add_axes([0.05, 0.85, 0.15, 0.1])
    view_radio_buttons = RadioButtons(
        view_axes,
        View.get_ordered_view_string_list(),
        active=current_view.value,
    )

    def update_current_view(current_slice_float: float):
        nonlocal current_slice
        nonlocal current_view
        nonlocal current_sh_coef

        current_slice = round(current_slice_float)
        image_data_view_slice = get_image_data_for_view_and_slice(image_fdata, current_view, current_slice, current_sh_coef)
        ax_image.set_data(image_data_view_slice)
        fig.canvas.draw()

    def update_view(view: str):
        nonlocal current_view
        nonlocal ax_image
        current_view = View.get_view_from_string(view)
        image_data_view_slice = get_image_data_for_view_and_slice(image_fdata, current_view, current_slice)

        # Sembler fonctionner uniquement en Debug.
        ax_image = ax.imshow(
            image_data_view_slice,
            # cmap='gray',
            vmin=np.min(image_fdata),
            vmax=np.max(image_fdata),
            origin=origin,
            aspect=get_aspect_for_view(image, current_view)
        )
        fig.canvas.draw()

    view_radio_buttons.on_clicked(update_view)
    slice_slider.on_changed(update_current_view)
    # fodf_slider.on_changed(update_current_view)

    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()


def main():
    parser = generate_argument_parser()
    args = parser.parse_args()

    image_path = Path(args.image_name)

    print("Loading image... ", image_path)

    img: Nifti1Image = nib.nifti1.load(str(image_path))

    display_image(img)


if __name__ == "__main__":
    main()