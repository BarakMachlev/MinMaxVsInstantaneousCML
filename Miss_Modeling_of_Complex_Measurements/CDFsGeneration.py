import sys
sys.path.insert(0, "/home/lucy3/BarakMachlev/PyNNcml")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pynncml as pnc


# =============================================================
# SyntheticLink definition
# =============================================================

class SyntheticLink:
    def __init__(self, tsl, meta_data):
        self.rsl = None
        self.tsl = tsl
        self.rain_rate = None
        self.attenuation = None
        self.rain_rate_15min = None
        self.meta_data = meta_data
        self.protocol_id = None
        self.link_index = None


# =============================================================
# Synthetic datasets
# =============================================================

def get_normalized_attenuation_cdf(pkl_path):

    with open(pkl_path, "rb") as f:
        synthetic_dataset = pickle.load(f)

    attenuation_list = []

    for link in synthetic_dataset:

        rsl = np.asarray(link.rsl)
        tsl = np.asarray(link.tsl)

        attenuation = tsl - rsl

        attenuation_list.append(attenuation.ravel())

    all_attenuation = np.concatenate(attenuation_list)

    all_attenuation = all_attenuation[
        np.isfinite(all_attenuation)
    ]

    # Global mean over ALL links and ALL samples
    mean_attenuation = np.mean(all_attenuation)

    if np.isclose(mean_attenuation, 0):
        raise ValueError(
            f"Mean attenuation is approximately zero for {pkl_path}"
        )

    # Normalize
    normalized_attenuation = (
        all_attenuation / mean_attenuation
    )

    # ECDF
    x = np.sort(normalized_attenuation)
    cdf = np.arange(1, len(x) + 1) / len(x)

    return x, cdf, mean_attenuation


# =============================================================
# OpenMRG
# =============================================================

def get_openmrg_normalized_attenuation_cdf():

    # ---------------------------------------------------------
    # Load the same OpenMRG dataset
    # ---------------------------------------------------------

    xy_min = [1.29e6, 0.565e6]
    xy_max = [1.34e6, 0.5875e6]

    time_slice = slice(
        "2015-06-01",
        "2015-08-31"
    )

    dataset = pnc.datasets.loader_open_mrg_dataset(
        restriction_minimum_length=0.75,
        xy_min=xy_min,
        xy_max=xy_max,
        time_slice=time_slice,
        samples_type="instantaneous",
        sampling_interval_in_sec=10
    )

    # Same protocol mapping as your training code:
    # ("instantaneous", 10) -> 11
    INSTANTANEOUS_10_SEC_PROTOCOL_ID = 11

    attenuation_list = []
    selected_link_ids = []

    # ---------------------------------------------------------
    # Go through all expanded OpenMRG items.
    #
    # IMPORTANT:
    # protocol_id comes from dataset[idx],
    # while physical link_index comes from link_list[idx].
    # ---------------------------------------------------------

    for idx, link in enumerate(dataset.link_set.link_list):

        physical_link_id = link.link_index

        # We only want the 80 physical links used in the paper
        if physical_link_id not in range(80):
            continue

        (
            rain_rate,
            rsl,
            tsl,
            metadata,
            protocol_id
        ) = dataset[idx]

        # protocol_id may be a torch tensor
        if hasattr(protocol_id, "item"):
            protocol_id_value = protocol_id.item()
        else:
            protocol_id_value = int(protocol_id)

        # Keep only full-resolution instantaneous 10-sec data
        if protocol_id_value != INSTANTANEOUS_10_SEC_PROTOCOL_ID:
            continue

        # Convert to numpy
        if hasattr(rsl, "detach"):
            rsl = rsl.detach().cpu().numpy()
        else:
            rsl = np.asarray(rsl)

        if hasattr(tsl, "detach"):
            tsl = tsl.detach().cpu().numpy()
        else:
            tsl = np.asarray(tsl)

        # -----------------------------------------------------
        # Attenuation = TSL - RSL
        # -----------------------------------------------------

        attenuation = tsl - rsl

        attenuation_list.append(
            attenuation.ravel()
        )

        selected_link_ids.append(
            physical_link_id
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    unique_link_ids = sorted(
        set(selected_link_ids)
    )

    print("\nOpenMRG validation:")
    print(f"Selected protocol items: {len(selected_link_ids)}")
    print(f"Unique physical links:   {len(unique_link_ids)}")

    assert len(unique_link_ids) == 80, (
        f"Expected 80 physical links, "
        f"but found {len(unique_link_ids)}"
    )

    assert len(selected_link_ids) == 80, (
        f"Expected one instantaneous 10-sec item "
        f"for each physical link, "
        f"but found {len(selected_link_ids)}"
    )

    assert unique_link_ids == list(range(80)), (
        "Physical link indices are not exactly 0-79"
    )

    print("✅ OpenMRG contains exactly the expected 80 physical links")
    print("✅ Using exactly one instantaneous 10-sec representation per link")

    # ---------------------------------------------------------
    # Combine attenuation from all 80 links
    # ---------------------------------------------------------

    all_attenuation = np.concatenate(
        attenuation_list
    )

    all_attenuation = all_attenuation[
        np.isfinite(all_attenuation)
    ]

    # ---------------------------------------------------------
    # Normalize by OpenMRG global mean
    # ---------------------------------------------------------

    mean_attenuation = np.mean(
        all_attenuation
    )

    if np.isclose(mean_attenuation, 0):
        raise ValueError(
            "OpenMRG mean attenuation is approximately zero."
        )

    normalized_attenuation = (
        all_attenuation
        / mean_attenuation
    )

    # ---------------------------------------------------------
    # Empirical CDF
    # ---------------------------------------------------------

    x = np.sort(
        normalized_attenuation
    )

    cdf = (
        np.arange(1, len(x) + 1)
        / len(x)
    )

    return (
        x,
        cdf,
        mean_attenuation,
        len(all_attenuation)
    )


# =============================================================
# Main
# =============================================================

dataset_dir = "."

dataset_files = [
    "synthetic_dataset_1.pkl",
    "synthetic_dataset_4.pkl",
]


plt.figure(figsize=(8, 6))


# =============================================================
# Plot synthetic datasets
# =============================================================

for filename in dataset_files:

    dataset_number = (
        filename
        .split("_")[-1]
        .split(".")[0]
    )

    pkl_path = os.path.join(
        dataset_dir,
        filename
    )

    (
        x,
        cdf,
        mean_attenuation
    ) = get_normalized_attenuation_cdf(
        pkl_path
    )

    print(
        f"Synthetic Dataset {dataset_number}: "
        f"mean attenuation = "
        f"{mean_attenuation:.6f} dB, "
        f"number of samples = {len(x):,}"
    )

    plt.plot(
        x,
        cdf,
        linewidth=4,
        color="blue" if dataset_number == "1" else "red",
        label=f"Synthetic Dataset {dataset_number}"
    )


# =============================================================
# Plot OpenMRG
# =============================================================

(
    x_openmrg,
    cdf_openmrg,
    mean_openmrg,
    n_openmrg
) = get_openmrg_normalized_attenuation_cdf()


print(
    f"OpenMRG: "
    f"mean attenuation = {mean_openmrg:.6f} dB, "
    f"number of samples = {n_openmrg:,}"
)


plt.plot(
    x_openmrg,
    cdf_openmrg,
    linewidth=4,
    color="limegreen",
    label="OpenMRG"
)


# =============================================================
# Figure
# =============================================================

plt.xlabel("Normalized Attenuation")
plt.ylabel("CDF")

plt.grid(True)

# Explicit position avoids the huge-data loc='best' warning
plt.legend(
    loc="lower right",
    fontsize=12
)

plt.tight_layout()


# =============================================================
# Save directory
# =============================================================

base_output_dir = (
    "/home/lucy3/BarakMachlev/Thesis/"
    "Article_Results/GUT/CDFs"
)

os.makedirs(
    base_output_dir,
    exist_ok=True
)


# =============================================================
# Save full-range figure - PNG and EPS
# =============================================================

figure_path_png = os.path.join(
    base_output_dir,
    "synthetic_and_openmrg_normalized_attenuation_cdf.png"
)

figure_path_eps = os.path.join(
    base_output_dir,
    "synthetic_and_openmrg_normalized_attenuation_cdf.eps"
)

plt.savefig(
    figure_path_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    figure_path_eps,
    format="eps",
    bbox_inches="tight"
)

print(
    f"\n✅ Full-range PNG saved to: {figure_path_png}"
)

print(
    f"✅ Full-range EPS saved to: {figure_path_eps}"
)


# =============================================================
# Save the SAME figure with xlim [0, 4] - PNG and EPS
# =============================================================

plt.xlim(0, 3.5)

figure_path_zoom_png = os.path.join(
    base_output_dir,
    "synthetic_and_openmrg_normalized_attenuation_cdf_xlim_0_4.png"
)

figure_path_zoom_eps = os.path.join(
    base_output_dir,
    "synthetic_and_openmrg_normalized_attenuation_cdf_xlim_0_4.eps"
)

plt.savefig(
    figure_path_zoom_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    figure_path_zoom_eps,
    format="eps",
    bbox_inches="tight"
)

print(
    f"✅ Zoomed PNG saved to: {figure_path_zoom_png}"
)

print(
    f"✅ Zoomed EPS saved to: {figure_path_zoom_eps}"
)


# =============================================================
# Show
# =============================================================

plt.show()