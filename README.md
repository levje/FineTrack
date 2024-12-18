# FineTrack

FineTrack is a reinforcement learning (RL) framework applied to [tractography](https://tractography.io/about/) where the **reward model is iteratively aligned with the policy** during training. This work also bring some architectural improvements along with an added module for Monte-Carlo exploration during training and inference/tracking.

It is a continuation starting from previous works of *Théberge et al.* ([see below](#reference-work)).

## Getting started

**Right now, only python 3.10 is supported.**

It is recommended to use a python [virtual environment](https://virtualenv.pypa.io/en/latest/user_guide.html) to run the code.


``` bash
virtualenv .env --python=python3.10
source .env/bin/activate
```

Then, install the dependencies and setup the repo with

``` bash
./install.sh
```

Getting errors during installation? Open an issue!

### Tracking

You will need a trained agent for tracking. One is provided in the `models` folder and is loaded autmatically when tracking. You can then track by running `ttl_track.py`.

You will need to provide fODFs, a seeding mask and a WM mask. The seeding mask **must** represent the interface of white matter and gray matter.

Agents used for tracking are constrained by their training regime. For example, the agents provided in `models` were trained on a volume with a resolution of ~1mm iso voxels and a step size of 0.75mm using fODFs of order 8, `descoteaux07` basis. When tracking on arbitrary data, the step-size and fODF order and basis will be adjusted accordingly automatically (i.e resulting in a step size of 0.375mm on 0.5mm iso diffusion data). **However**, if using fODFs in the `tournier07` basis, you will need to set the `--sh_basis` argument accordingly.

### Contributing

Contributions are welcome. Please refer to the [contribution guidelines](./docs/CONTRIBUTING.md)

## Reference work

This work is a **continuation** of the [TrackToLearn/TractOracle-RL framework](https://github.com/scil-vital/TrackToLearn) from *Théberge et al.*.

> Théberge, A., Descoteaux, M., & Jodoin, P. M. (2024). TractOracle: towards an anatomically-informed reward function for RL-based tractography. Accepted at MICCAI 2024.

To see the reference version or previous versions of this work, please visit [this work](https://github.com/scil-vital/TrackToLearn).
