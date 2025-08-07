import os
import moabb
import rosoku
import torch
from pathlib import Path
import mne
import tag_mne as tm
import braindecode
import functools
import sklearn.metrics


def epochs_from_raws(raws, tmin, tmax, l_freq, h_freq, order_filter):
    epochs_list = list()
    for run, raw in raws.items():

        run = run[0:-7]

        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="iir",
            iir_params={"ftype": "butter", "order": order_filter, "btype": "bandpass"},
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
        markers = tm.add_tag(markers, f"run:{run}")

        events, event_id = tm.events_from_markers(samples, markers)

        epochs = mne.Epochs(
            raw, tmin=tmin, tmax=tmax, baseline=None, events=events, event_id=event_id
        )

        epochs_list.append(epochs["marker:Target", "marker:NonTarget"])

    epochs = tm.concatenate_epochs(epochs_list)

    return epochs


def func_proc_epochs(epochs, mode):
    epochs = epochs.pick(picks="eeg")
    return epochs


def func_load_epochs(keywords, mode, epochs):
    return epochs[keywords]


def func_get_model(X, y):
    _, n_chans, n_times = X.shape
    F1 = 4
    D = 2
    F2 = F1 * D

    model = braindecode.models.EEGNetv4(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        F1=F1,
        D=D,
        F2=F2,
        drop_prob=0.5,
    )

    return model


if __name__ == "__main__":

    subject = 1
    tmin = 0.0
    tmax = 1.0
    l_freq = 0.1
    h_freq = 8
    order_filter = 2
    resample = 128

    force_epoching = False

    lr = 1e-3
    weight_decay = 1e-2
    n_epochs = 500
    batch_size = 64
    patience = 75
    enable_euclidean_alignment = False
    enable_normalization = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    enable_ddp = False
    enable_dp = False

    seed = 42

    save_base = Path("~").expanduser() / "rosoku-log"
    (save_base / "checkpoint").mkdir(parents=True, exist_ok=True)
    (save_base / "history").mkdir(parents=True, exist_ok=True)

    dataset = moabb.datasets.Kojima2024B_2stream()

    if (
        os.path.exists(
            save_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
        )
    ) and (force_epoching is False):
        epochs = mne.read_epochs(
            save_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
        )
        epochs.load_data()
    else:
        sessions = dataset.get_data(subjects=[subject])
        raws = sessions[subject]["0"]
        epochs = epochs_from_raws(
            raws=raws,
            tmin=tmin,
            tmax=tmax,
            l_freq=l_freq,
            h_freq=h_freq,
            order_filter=order_filter,
        ).resample(resample)
        epochs.save(
            (save_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"),
            overwrite=True,
        )

        epochs = mne.read_epochs(
            save_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
        )
        epochs.load_data()

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0]).to(device))
    # criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params = {"T_max": n_epochs, "eta_min": 1e-6}
    optimizer = torch.optim.AdamW
    optimizer_params = {"lr": lr, "weight_decay": weight_decay}
    early_stopping = rosoku.utils.EarlyStopping(patience=patience)

    results = rosoku.deeplearning(
        keywords_train=[f"run:{m}" for m in [1, 3, 5, 8]],
        keywords_valid=[f"run:10"],
        keywords_test=[f"run:12"],
        func_load_epochs=functools.partial(func_load_epochs, epochs=epochs),
        func_convert_epochs_to_ndarray=functools.partial(
            rosoku.utils.convert_epochs_to_ndarray,
            label_keys={"marker:Target": 1, "marker:NonTarget": 0},
        ),
        apply_func_proc_per_obj=True,
        batch_size=batch_size,
        n_epochs=n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        func_get_model=func_get_model,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        device=device,
        enable_ddp=enable_ddp,
        func_proc_epochs=func_proc_epochs,
        early_stopping=early_stopping,
        enable_normalization=enable_normalization,
        name_classifier="eegnet4.2",
        history_fname=(save_base / "history" / f"sub-{subject}"),
        checkpoint_fname=(save_base / "checkpoint" / f"sub-{subject}.pth"),
        desc="eegnet4.2/drop_prob=0.25",
        enable_wandb_logging=False,
        wandb_params={
            "config": {
                "lr": lr,
                "weight_decay": weight_decay,
                "n_epochs": n_epochs,
                "force_epoching": force_epoching,
                "device": device,
            },
            "project": "sandbox-kojima2024b",
            "name": f"sub-{subject}",
        },
        seed=seed,
    )

    print(results.loc[0])

    probas = results.loc[0]["probas"]
    labels = results.loc[0]["labels"]
    preds = results.loc[0]["preds"]

    f1 = sklearn.metrics.f1_score(labels, preds)
    bacc = sklearn.metrics.balanced_accuracy_score(labels, preds)

    auc = sklearn.metrics.roc_auc_score(labels, probas[:, 1])
    print(f"BACC: {bacc}")
    print(f"F1: {f1}")
    print(f"AUC: {auc}")
