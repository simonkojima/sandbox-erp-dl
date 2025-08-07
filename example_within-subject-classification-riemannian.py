"""
Example: Within-subject classification with riemannian classifier
=================================================================
"""

# %%
import functools
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import mne
import tag_mne as tm

import moabb.datasets

import torch
import pyriemann
import rosoku

# %%

subject = 56
resample = 128

# %%

# load dataset and generate epochs


def epochs_from_raws(
    raws, runs, rtypes, tmin, tmax, l_freq, h_freq, order_filter, subject
):
    epochs_list = list()
    for raw, run, rtype in zip(raws, runs, rtypes):

        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="iir",
            iir_params={"ftype": "butter", "order": 4, "btype": "bandpass"},
        )

        # eog and emg mapping
        mapping = dict()
        for ch in raw.ch_names:
            if "EOG" in ch:
                mapping[ch] = "eog"
            elif "EMG" in ch:
                mapping[ch] = "emg"

        raw.set_channel_types(mapping)
        raw.set_montage("standard_1020")

        events, event_id = mne.events_from_annotations(raw)

        samples, markers = tm.markers_from_events(events, event_id)
        markers = tm.add_tag(markers, f"subject:{subject}")
        markers = tm.add_event_names(
            markers, {"left": ["left_hand"], "right": ["right_hand"]}
        )
        markers = tm.add_tag(markers, f"run:{run}")
        markers = tm.add_tag(markers, f"rtype:{rtype}")

        samples, markers = tm.remove(samples, markers, "event:misc")

        events, event_id = tm.events_from_markers(samples, markers)
        epochs = mne.Epochs(
            raw=raw,
            tmin=tmin,
            tmax=tmax,
            events=events,
            event_id=event_id,
            baseline=None,
        )

        epochs_list.append(epochs)

    epochs = tm.concatenate_epochs(epochs_list)

    return epochs


dataset = moabb.datasets.Dreyer2023()
sessions = dataset.get_data(subjects=[subject])
raws = sessions[subject]["0"]

epochs_acquisition = epochs_from_raws(
    raws=[raws[key] for key in ["0R1acquisition", "1R2acquisition"]],
    runs=[1, 2],
    rtypes=["acquisition", "acquisition"],
    tmin=-1.0,
    tmax=5.5,
    l_freq=8.0,
    h_freq=30.0,
    order_filter=4,
    subject=subject,
).resample(resample)

epochs_online = epochs_from_raws(
    raws=[raws[key] for key in ["2R3online", "3R4online", "4R5online"]],
    runs=[3, 4, 5],
    rtypes=["online", "online", "online"],
    tmin=-1.0,
    tmax=5.5,
    l_freq=8.0,
    h_freq=30.0,
    order_filter=4,
    subject=subject,
).resample(resample)

epochs = tm.concatenate_epochs([epochs_acquisition, epochs_online])

# %%


def func_proc_epochs(epochs, mode, tmin=0.5, tmax=4.5):
    epochs = epochs.pick(picks="eeg").crop(tmin=tmin, tmax=tmax)
    return epochs


def func_load_epochs(keywords, mode, epochs):
    return epochs[keywords]


def convert_epochs_to_ndarray(
    epochs_train,
    epochs_test,
    label_keys,
):

    X_train = epochs_train.get_data()
    X_test = epochs_test.get_data()

    X_train = pyriemann.estimation.Covariances().transform(X_train)
    X_test = pyriemann.estimation.Covariances().transform(X_test)

    y_train = rosoku.utils.get_labels_from_epochs(epochs_train, label_keys)
    y_test = rosoku.utils.get_labels_from_epochs(epochs_test, label_keys)

    return X_train, X_test, y_train, y_test


# %%
label_keys = {"event:left": 0, "event:right": 1}

results = rosoku.conventional(
    keywords_train=["run:1", "run:2"],
    keywords_test=["run:3", "run:4", "run:5"],
    func_load_epochs=functools.partial(func_load_epochs, epochs=epochs),
    func_proc_epochs=func_proc_epochs,
    func_convert_epochs_to_ndarray=functools.partial(
        convert_epochs_to_ndarray, label_keys=label_keys
    ),
)

print(results)
