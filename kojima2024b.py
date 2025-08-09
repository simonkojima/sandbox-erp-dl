import os
import argparse
import moabb
import datetime
import rosoku
import torch
from pathlib import Path
import pandas as pd
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
    epochs = epochs.pick(picks="eeg").crop(tmin=0.0, tmax=1.0)
    return epochs


def func_load_epochs(keywords, mode, epochs):
    return epochs[keywords]


def func_get_model(X, y, classifier):
    _, n_chans, n_times = X.shape

    match classifier:
        case "eegnet4.2":
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
        case "eegnet8.2":
            F1 = 8
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

        case "EEGInceptionERP":
            model = braindecode.models.EEGInceptionERP(
                n_chans=n_chans,
                n_outputs=2,
                n_times=n_times,
                sfreq=128,
            )

        case "DeepConvNet":
            model = braindecode.models.Deep4Net(
                n_chans=n_chans,
                n_outputs=2,
                n_times=n_times,
                final_conv_length="auto",
                pool_time_length=2,
                pool_time_stride=2,
                filter_time_length=5,
            )

        case "ShallowConvNet":
            model = braindecode.models.ShallowFBCSPNet(
                n_chans=n_chans,
                n_outputs=2,
                n_times=n_times,
                final_conv_length="auto",
            )

    return model


def main(timestamp, subject, classifier, resample):
    tmin = 0.0
    tmax = 1.2
    l_freq = 0.1
    h_freq = 40
    order_filter = 2
    # resample = 128

    force_epoching = False

    n_epochs = 500
    patience = 75
    enable_normalization = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_ddp = False

    seed = 42

    save_base = (
        Path("~").expanduser()
        / "Documents"
        / "results"
        / "classification"
        / "erp-dl"
        / timestamp
        / classifier
    )
    (save_base / "checkpoint").mkdir(parents=True, exist_ok=True)
    (save_base / "history").mkdir(parents=True, exist_ok=True)

    dataset = moabb.datasets.Kojima2024B_2stream()
    epochs_base = (
        Path("~").expanduser()
        / "mne_data"
        / f"MNE-{dataset.code}-data"
        / "epochs"
        / f"tmin-{tmin}_tmax-{tmax}_l_freq-{l_freq}_h_freq-{h_freq}_order-{order_filter}_resample-{resample}"
    )
    epochs_base.mkdir(parents=True, exist_ok=True)

    if (
        os.path.exists(
            epochs_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
        )
    ) and (force_epoching is False):
        epochs = mne.read_epochs(
            epochs_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
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
            (epochs_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"),
            overwrite=True,
        )

        epochs = mne.read_epochs(
            epochs_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
        )
        epochs.load_data()

    for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        for batch_size in [4, 8, 16, 32, 64]:
            for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
                criterion = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor([1.0, 3.0]).to(device)
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
                scheduler_params = {"T_max": n_epochs, "eta_min": 1e-6}
                optimizer = torch.optim.AdamW
                optimizer_params = {"lr": lr, "weight_decay": weight_decay}
                early_stopping = rosoku.utils.EarlyStopping(
                    patience=patience, min_delta=1e-3
                )

                results = rosoku.deeplearning(
                    keywords_train=[f"run:{m}" for m in [1, 3, 5, 8]],
                    keywords_valid=[f"run:10"],
                    keywords_test=[f"run:12"],
                    func_load_epochs=functools.partial(
                        func_load_epochs, epochs=epochs.copy()
                    ),
                    func_convert_epochs_to_ndarray=functools.partial(
                        rosoku.utils.convert_epochs_to_ndarray,
                        label_keys={"marker:Target": 1, "marker:NonTarget": 0},
                    ),
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    criterion=criterion,
                    optimizer=optimizer,
                    optimizer_params=optimizer_params,
                    func_get_model=functools.partial(
                        func_get_model, classifier=classifier
                    ),
                    scheduler=scheduler,
                    scheduler_params=scheduler_params,
                    device=device,
                    enable_ddp=enable_ddp,
                    func_proc_epochs=func_proc_epochs,
                    early_stopping=early_stopping,
                    enable_normalization=enable_normalization,
                    name_classifier=classifier,
                    history_fname=(
                        save_base
                        / "history"
                        / f"sub-{subject}_lr-{lr}_bs-{batch_size}_wd-{weight_decay}"
                    ),
                    checkpoint_fname=(
                        save_base
                        / "checkpoint"
                        / f"sub-{subject}_lr-{lr}_bs-{batch_size}_wd-{weight_decay}.pth"
                    ),
                    desc=classifier,
                    enable_wandb_logging=True,
                    wandb_params={
                        "config": {
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "n_epochs": n_epochs,
                            "batch_size": batch_size,
                            "force_epoching": force_epoching,
                            "device": device,
                            "subject": subject,
                            "model": classifier,
                        },
                        "project": "sandbox-kojima2024b-params-search",
                        "name": f"sub-{subject}_lr-{lr}_bs-{batch_size}_wd-{weight_decay}",
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

                df_save_base = (
                    Path("~").expanduser()
                    / "Documents"
                    / "results"
                    / "classification"
                    / "erp-dl"
                    / timestamp
                )

                results["subject"] = [subject]
                results["bacc"] = [bacc]
                results["f1"] = [f1]
                results["auc"] = [auc]
                results["lr"] = [lr]
                results["weight_decay"] = [weight_decay]
                results["batch_size"] = [batch_size]

                df_all.append(results)

                df = pd.concat(df_all, axis=0, ignore_index=True)
                df.to_pickle(df_save_base / "classification-results.pkl")
                df.to_html(df_save_base / "classification-results.html")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    classifiers = [
        "eegnet4.2",
        "eegnet8.2",
        "EEGInceptionERP",
        "DeepConvNet",
        "ShallowConvNet",
    ]

    df_all = []

    resamples = [128, 128, 128, 250, 250]

    subjects = list(range(1, 16))

    for subject in subjects:
        for classifier, resample in zip(classifiers, resamples):
            main(timestamp, subject, classifier, resample)
