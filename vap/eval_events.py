import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from os.path import dirname
import tqdm
import matplotlib.pyplot as plt

from vap.data.dset_event import VAPClassificationDataset
from vap.utils.plot import plot_melspectrogram, plot_vap_probs, plot_vad
from vap.modules.lightning_module import VAPModule, VAP, everything_deterministic

everything_deterministic()


def get_shift_probability(out, speaker, region_start: int, region_end: int):
    """
    out['p']:        4, n_batch, n_frames
    out['p_now']:    n_batch, n_frames
    out['p_future']: n_batch, n_frames
    """
    ps = out["p"][..., region_start:region_end].mean(-1).cpu()
    pn = out["p_now"][..., region_start:region_end].mean(-1).cpu()
    pf = out["p_future"][..., region_start:region_end].mean(-1).cpu()

    batch_size = pn.shape[0]

    # if batch size == 1
    if batch_size == 1:
        speaker = [speaker]

    # Make all values 'shift probabilities'
    # The speaker is the speaker of the target IPU
    # A shift is the probability of the other speaker
    # The predictions values are always for the first speaker
    # So if the current speaker is speaker 1 then the probability of the default
    # speaker is the same as the shift-probability
    # However, if the current speaker is speaker 0 then the default probabilities
    # are HOLD probabilities, so we need to invert them
    for ii, spk in enumerate(speaker):
        if spk == 0:
            ps[:, ii] = 1 - ps[:, ii]
            pn[ii] = 1 - pn[ii]
            pf[ii] = 1 - pf[ii]

    preds = {f"p{k+1}": v.tolist() for k, v in enumerate(ps)}
    preds["p_now"] = pn.tolist()
    preds["p_fut"] = pf.tolist()
    return preds


def get_targets(labels):
    if isinstance(labels, str):
        labels = [labels]
    targets = []
    for lab in labels:
        targets.append(1 if lab == "shift" else 0)
    return targets


@torch.inference_mode()
def extract_preds_and_targets(
    model: VAP,
    dloader: DataLoader,
    region_start_time: float,
    region_end_time: float,
) -> tuple[dict[str, Tensor], Tensor]:
    model = model.eval()
    region_start = int(region_start_time * model.frame_hz)
    region_end = int(region_end_time * model.frame_hz)
    preds = {
        "p1": [],
        "p2": [],
        "p3": [],
        "p4": [],
        "p_now": [],
        "p_fut": [],
    }
    targets = []
    for batch in tqdm.tqdm(dloader, desc="Event classification"):
        # Model prediction
        out = model.probs(batch["waveform"].to(model.device))

        batch_preds = get_shift_probability(
            out, batch["speaker"], region_start, region_end
        )
        batch_targets = get_targets(batch["label"])
        targets.extend(batch_targets)
        preds["p_now"].extend(batch_preds["p_now"])
        preds["p_fut"].extend(batch_preds["p_fut"])
        for ii in range(1, 5):
            preds[f"p{ii}"].extend(batch_preds[f"p{ii}"])
    preds = {k: torch.tensor(v) for k, v in preds.items()}
    targets = torch.tensor(targets)
    return preds, targets


def extract_threshold_accuracy(preds, targets):
    acc = {
        "acc": {k: [] for k in preds.keys()},
        "bacc": {k: [] for k in preds.keys()},
        "shift": {k: [] for k in preds.keys()},
        "hold": {k: [] for k in preds.keys()},
        "thresholds": torch.arange(0, 1.05, 0.05),
    }
    for thres in acc["thresholds"]:
        for k, v in preds.items():
            pred_tmp = (v >= thres).float()
            correct = (pred_tmp == targets).float()
            shift_acc = correct[targets == 1].mean()
            hold_acc = correct[targets == 0].mean()
            acc["acc"][k].append(correct.mean())
            acc["bacc"][k].append((shift_acc + hold_acc) / 2)
            acc["shift"][k].append(shift_acc)
            acc["hold"][k].append(hold_acc)

    # Convert to tensors
    for acc_type, accs in acc.items():
        if acc_type == "thresholds":
            continue
        for pred_type, accu in accs.items():
            # print(acc_type, pred_type, torch.tensor(accu).shape)
            acc[acc_type][pred_type] = torch.tensor(accu)

    return acc


def plot_output(d, out, height_ratios=[2, 2, 1, 1, 1, 1]):
    # Create the figure and the GridSpec instance with the given height ratios
    fig, ax = plt.subplots(
        nrows=6,
        sharex=True,
        figsize=(15, 6),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.01},
    )
    plot_melspectrogram(d["waveform"], ax=ax[:2])
    # plot vad.
    x2 = torch.arange(out["vad"].shape[1]) / dset.frame_hz
    plot_vad(x2, out["vad"][0, :, 0], ax[0], ypad=3, color="w", label="VAD pred")
    plot_vad(x2, out["vad"][0, :, 1], ax[1], ypad=3, color="w", label="VAD pred")
    for i in range(4):
        plot_vap_probs(out["p"][i, 0], ax=ax[2 + i])
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].legend()
    ax[1].legend()
    ax[-1].set_xticks(list(range(0, 1 + round(x2[-1].item()))))  # list(range(0, 20)))
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig, ax


def plot_accuracy(acc):
    def get_best_acc(acc, acc_type="bacc"):
        now_t = acc["thresholds"][acc[acc_type]["p_now"].argmax()].item()
        now_p = acc[acc_type]["p_now"].max().item()
        best_t = acc["thresholds"][acc[acc_type]["p_fut"].argmax()].item()
        best_p = acc[acc_type]["p_fut"].max().item()
        best_label = f"p_fut: ({best_t:.2f}, {best_p:.2f})"
        best_color = "g"
        if now_p > best_p:
            best_p = now_p
            best_t = now_t
            best_label = f"p_now: ({best_t:.2f}, {best_p:.2f})"
            best_color = "r"
        return best_t, best_p, best_label, best_color

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        acc["thresholds"],
        acc["bacc"]["p_now"],
        label="bacc p_now",
        color="r",
        linewidth=2,
        alpha=0.6,
    )
    ax.plot(
        acc["thresholds"],
        acc["bacc"]["p_fut"],
        label="bacc p_fut",
        color="g",
        linewidth=2,
        alpha=0.6,
    )
    ax.plot(
        acc["thresholds"],
        acc["acc"]["p_now"],
        label="acc p_now",
        color="r",
        linewidth=2,
        linestyle="--",
        alpha=0.6,
    )
    ax.plot(
        acc["thresholds"],
        acc["acc"]["p_fut"],
        label="acc p_fut",
        color="g",
        linewidth=2,
        linestyle="--",
        alpha=0.6,
    )
    ax.axhline(0.5, color="k", linestyle="--")
    # Plot best BACC values
    best_bacc_t, best_bacc_p, best_bacc_label, best_bacc_color = get_best_acc(
        acc, acc_type="bacc"
    )
    ax.scatter(
        best_bacc_t,
        best_bacc_p,
        color=best_bacc_color,
        label="BAcc: " + best_bacc_label,
    )
    ax.plot(
        [0, best_bacc_t], [best_bacc_p, best_bacc_p], color=best_bacc_color, linewidth=1
    )
    ax.plot(
        [best_bacc_t, best_bacc_t], [0, best_bacc_p], color=best_bacc_color, linewidth=1
    )
    # Plot best ACC values
    best_acc_t, best_acc_p, best_acc_label, best_acc_color = get_best_acc(
        acc, acc_type="acc"
    )
    ax.scatter(
        best_acc_t,
        best_acc_p,
        color=best_acc_color,
        marker="+",
        label="Acc: " + best_acc_label,
    )
    ax.plot(
        [0, best_acc_t],
        [best_acc_p, best_acc_p],
        color=best_acc_color,
        linestyle="--",
        linewidth=1,
    )
    ax.plot(
        [best_acc_t, best_acc_t],
        [0, best_acc_p],
        color=best_acc_color,
        linestyle="--",
        linewidth=1,
    )
    ax.legend()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Threshold")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.tight_layout()
    return fig, ax


def debug():
    def debug_output(batch, out, region_start=None, region_end=None):
        w = batch["waveform"].cpu()
        vap_vad = out["vad"].squeeze().cpu()
        pn = out["p_now"].squeeze().cpu()
        pf = out["p_future"].squeeze().cpu()
        speaker = batch["speaker"]

        fig, ax = plt.subplots(
            nrows=8,
            sharex=True,
            figsize=(15, 6),
            gridspec_kw={
                "height_ratios": [1.5, 1.5, 1, 1, 1, 1, 1, 1],
                "hspace": 0.01,
            },
        )
        # Plot audio + VAD
        plot_melspectrogram(w, ax=ax[:2])
        x2 = torch.arange(vap_vad.shape[0]) / dset.frame_hz
        plot_vad(
            x2,
            vap_vad[:, 0],
            ax[0],
            ypad=3,
            color="w",
            label="VAD pred",
            linewidth=1 if speaker == 1 else 3,
        )
        plot_vad(
            x2,
            vap_vad[:, 1],
            ax[1],
            ypad=3,
            color="w",
            label="VAD pred",
            linewidth=1 if speaker == 0 else 3,
        )
        # Plot probs
        plot_vap_probs(pn, ax=ax[2])
        plot_vap_probs(pf, ax=ax[3])

        for ii, ps in enumerate(out["p"]):
            plot_vap_probs(ps.squeeze().cpu(), ax=ax[4 + ii])

        if region_start is not None and region_end is not None:
            for a in ax:
                a.axvspan(
                    region_start,
                    region_end,
                    color="r",
                    alpha=0.5,
                )
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        plt.tight_layout()
        return fig, ax

    context = 20
    region_sil_pad_time = 0.1
    region_start_time = context + region_sil_pad_time
    region_end_time = region_start_time + 0.2
    frame_start = int(region_start_time * dset.frame_hz)
    frame_end = int(region_end_time * dset.frame_hz)
    for i in range(len(dset)):
        with torch.inference_mode():
            batch = dset[i]
            out = model.probs(batch["waveform"].unsqueeze(0).to(model.device))
        tmp_preds = get_shift_probability(out, batch["speaker"], frame_start, frame_end)
        print(batch["label"].upper())
        for k, v in tmp_preds.items():
            pred_lab = "SHIFT" if v[0] > 0.5 else "HOLD"
            print(f"{k}: ", round(v[0], 2), pred_lab)
        print("#" * 40)
        fig, ax = debug_output(batch, out, region_start_time, region_end_time)
        plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser
    import pandas as pd
    import sys

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="event_results.csv")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--context", type=float, default=20)
    parser.add_argument("--region_sil_pad_time", type=float, default=0.2)
    parser.add_argument("--region_duration", type=float, default=0.2)
    parser.add_argument("--post_silence", type=float, default=1.0)
    parser.add_argument("--min_event_silence", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.result_csv:
        df = pd.read_csv(args.result_csv)

        targets = torch.from_numpy(df["targets"].values)
        preds = {
            "p_now": torch.from_numpy(df["p_now"].values),
            "p_fut": torch.from_numpy(df["p_fut"].values),
            "p1": torch.from_numpy(df["p1"].values),
            "p2": torch.from_numpy(df["p2"].values),
            "p3": torch.from_numpy(df["p3"].values),
            "p4": torch.from_numpy(df["p4"].values),
        }
        acc = extract_threshold_accuracy(preds, targets)
        print("Balanced ACCURACY")
        print("type: bacc, threshold")
        for pred_type, accuracy in acc["bacc"].items():
            t = acc["thresholds"][accuracy.argmax()].item()
            a = accuracy.max().item()
            print(f"{pred_type}: {a:.2f}, {t:.2f}")

        print("ACCURACY")
        print("type: acc, threshold")
        for pred_type, accuracy in acc["acc"].items():
            t = acc["thresholds"][accuracy.argmax()].item()
            a = accuracy.max().item()
            print(f"{pred_type}: {a:.2f}, {t:.2f}")

        if args.plot:
            fig, ax = plot_accuracy(acc)
            plt.show()
        sys.exit(0)

    assert (
        args.checkpoint is not None and args.csv is not None
    ), "Must provide --checkpoint and --csv"

    print("\nArguments")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 40)
    print()

    model: VAP = VAPModule.load_model(args.checkpoint).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    region_start_time = args.context + args.region_sil_pad_time
    region_end_time = region_start_time + args.region_duration
    dset = VAPClassificationDataset(
        df_path=args.csv,
        context=args.context,
        post_silence=args.post_silence,
        min_event_silence=0,
    )
    print("DSET: ", len(dset))
    dloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    preds, targets = extract_preds_and_targets(
        model, dloader, region_start_time, region_end_time
    )
    acc = extract_threshold_accuracy(preds, targets)

    # Combine data for save
    preds["label"] = ["SHIFT" if l == 1 else "HOLD" for l in targets]
    preds["targets"] = targets
    df = pd.DataFrame(
        columns=["p_fut", "p_now", "p1", "p2", "p3", "p4", "label", "targets"],
        data=preds,
    )
    Path(dirname(args.output)).mkdir(exist_ok=True, parents=True)
    df.to_csv(args.output, index=False)
    print("Saved results -> ", args.output)

    print("Balanced ACCURACY")
    print("type: bacc, threshold")
    for pred_type, accuracy in acc["bacc"].items():
        t = acc["thresholds"][accuracy.argmax()].item()
        a = accuracy.max().item()
        print(f"{pred_type}: {a:.2f}, {t:.2f}")

    print("ACCURACY")
    print("type: acc, threshold")
    for pred_type, accuracy in acc["acc"].items():
        t = acc["thresholds"][accuracy.argmax()].item()
        a = accuracy.max().item()
        print(f"{pred_type}: {a:.2f}, {t:.2f}")

    if args.plot:
        fig, ax = plot_accuracy(acc)
        plt.show()
