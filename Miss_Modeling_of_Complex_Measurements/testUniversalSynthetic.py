import numpy as np
import sys
sys.path.insert(0, "/home/lucy3/BarakMachlev/PyNNcml")
import pynncml as pnc
import torch
import os
import pickle
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader
from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric
from tqdm import tqdm
import math
import scipy

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

PROTOCOL_MAP = {
    ("instantaneous", 900): 0,
    ("instantaneous", 450): 1,
    ("instantaneous", 300): 2,
    ("instantaneous", 180): 3,
    ("instantaneous", 150): 4,
    ("instantaneous", 100): 5,
    ("instantaneous", 90): 6,
    ("instantaneous", 60): 7,
    ("instantaneous", 50): 8,
    ("instantaneous", 30): 9,
    ("instantaneous", 20): 10,
    ("instantaneous", 10): 11,
    "min_max": 12,
    "average": 13,
}

INSTANTANEOUS_INTERVALS = [900, 450, 300, 180, 150, 100, 90, 60, 50, 30, 20, 10]

def lr_schedule(epoch):
    if epoch < 3:
        return 0.5        # relative to base LR = 1e-4 → stays 1e-4
    elif epoch < 20:
        return 0.1        # 0.5 × 1e-4 = 5e-5
    else:
       return 0.1        # 0.1 × 1e-4 = 1e-5
    
def build_instantaneous_universal(link, sampling_interval_in_sec):
    k = sampling_interval_in_sec // 10
    base_token_len = 90

    rsl = link.rsl
    tsl = link.tsl
    rain = link.rain_rate_15min

    assert len(rsl) % base_token_len == 0
    T = len(rsl) // base_token_len

    rsl_tok = rsl.reshape(T, base_token_len)
    tsl_tok = tsl.reshape(T, base_token_len)

    rsl_s = rsl_tok[:, ::k].astype(np.float32)
    tsl_s = tsl_tok[:, ::k].astype(np.float32)

    rsl_u = np.repeat(rsl_s, k, axis=1)
    tsl_u = np.repeat(tsl_s, k, axis=1)

    new_link = SyntheticLink(tsl=None, meta_data=link.meta_data)
    new_link.rsl = rsl_u
    new_link.tsl = tsl_u
    new_link.rain_rate_15min = rain
    return new_link

def build_average_universal(link):
    base_token_len = 90
    rsl = link.rsl
    tsl = link.tsl
    rain = link.rain_rate_15min

    assert len(rsl) % base_token_len == 0
    T = len(rsl) // base_token_len

    rsl_tok = rsl.reshape(T, base_token_len)
    tsl_tok = tsl.reshape(T, base_token_len)

    rsl_avg = rsl_tok.mean(axis=1, keepdims=True).astype(np.float32)
    tsl_avg = tsl_tok.mean(axis=1, keepdims=True).astype(np.float32)

    rsl_u = np.repeat(rsl_avg, base_token_len, axis=1)
    tsl_u = np.repeat(tsl_avg, base_token_len, axis=1)

    new_link = SyntheticLink(tsl=None, meta_data=link.meta_data)
    new_link.rsl = rsl_u
    new_link.tsl = tsl_u
    new_link.rain_rate_15min = rain
    return new_link

def build_min_max_universal(link):
    base_token_len = 90
    half = 45

    rsl = link.rsl
    tsl = link.tsl
    rain = link.rain_rate_15min

    assert len(rsl) % base_token_len == 0
    T = len(rsl) // base_token_len

    rsl_tok = rsl.reshape(T, base_token_len)
    tsl_tok = tsl.reshape(T, base_token_len)

    max_rsl = rsl_tok.max(axis=1).astype(np.float32)
    min_rsl = rsl_tok.min(axis=1).astype(np.float32)
    max_tsl = tsl_tok.max(axis=1).astype(np.float32)
    min_tsl = tsl_tok.min(axis=1).astype(np.float32)

    rsl_u = np.concatenate([np.repeat(max_rsl[:, None], half, axis=1),
                            np.repeat(min_rsl[:, None], half, axis=1)], axis=1)

    tsl_u = np.concatenate([np.repeat(min_tsl[:, None], half, axis=1),
                            np.repeat(max_tsl[:, None], half, axis=1)], axis=1)

    new_link = SyntheticLink(tsl=None, meta_data=link.meta_data)
    new_link.rsl = rsl_u
    new_link.tsl = tsl_u
    new_link.rain_rate_15min = rain
    return new_link

with open("synthetic_dataset_4.pkl", "rb") as f:
    synthetic_dataset = pickle.load(f)

expanded_dataset = []

for physical_link_id, link in enumerate(synthetic_dataset):
    avg_link = build_average_universal(link)
    avg_link.protocol_id = PROTOCOL_MAP["average"]
    avg_link.link_index = physical_link_id
    expanded_dataset.append(avg_link)

    mm_link = build_min_max_universal(link)
    mm_link.protocol_id = PROTOCOL_MAP["min_max"]
    mm_link.link_index = physical_link_id
    expanded_dataset.append(mm_link)

    for interval in INSTANTANEOUS_INTERVALS:
        inst_link = build_instantaneous_universal(link, interval)
        inst_link.protocol_id = PROTOCOL_MAP[("instantaneous", interval)]
        inst_link.link_index = physical_link_id
        expanded_dataset.append(inst_link)

batch_size = 16
lr = 1e-4 # Originally 1e-4
weight_decay = 1e-4
n_epochs = 5
window_size = 32
metadata_n_features = 16
protocol_n_features = 16
metadata_input_size = 2
d_model = 256
dropout = 0.1
num_encoder_layers = 4
h = 8
num_protocols = 14
dynamic_input_size = 180

base_output_dir = (f"/home/lucy3/BarakMachlev/Thesis/Article_Results/GUT/Synthetic_DataSet/SL_3_FC_OriginalLoss/Final/DataSet_4_{i}")
output_dir = base_output_dir
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

class SyntheticDatasetWrapper(Dataset):
    def __init__(self, synthetic_links):
        self.synthetic_links = synthetic_links

    def __len__(self):
        return len(self.synthetic_links)

    def __getitem__(self, idx):
        link = self.synthetic_links[idx]

        rsl = torch.tensor(link.rsl, dtype=torch.float32)   # [T, 90]
        tsl = torch.tensor(link.tsl, dtype=torch.float32)   # [T, 90]
        #attenuation = tsl - rsl                             # [T, 90]

        return (
            torch.tensor(link.rain_rate_15min, dtype=torch.float32),  # [T]
            rsl,
            tsl,
            #attenuation,                                               
            torch.tensor([link.meta_data.length, link.meta_data.frequency], dtype=torch.float32),
            torch.tensor(link.protocol_id, dtype=torch.long)
        )

wrapped_dataset = SyntheticDatasetWrapper(expanded_dataset)

train_phys = list(range(64))
val_phys = list(range(64, 80))

train_indices = [i for i, lnk in enumerate(expanded_dataset) if lnk.link_index in train_phys]
val_indices = [i for i, lnk in enumerate(expanded_dataset) if lnk.link_index in val_phys]

from collections import Counter

counts = Counter([lnk.link_index for lnk in expanded_dataset])

assert all(v == 14 for v in counts.values()), "Some physical links do not have 14 protocol variants"

assert set(train_phys).isdisjoint(set(val_phys)), "Leakage between train and validation"

print(f"Total items: {len(expanded_dataset)}")
print(f"Train items: {len(train_indices)}  (expected ~ {64*14})")
print(f"Val items:   {len(val_indices)}    (expected ~ {16*14})")

training_dataset = Subset(wrapped_dataset, train_indices)
validation_dataset = Subset(wrapped_dataset, val_indices)

data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

normalization_cfg = pnc.training_helpers.compute_data_normalization(
    data_loader,
    network_dynamic_input_size=dynamic_input_size
)

norm_path = os.path.join(output_dir, "normalization_cfg.pth")
torch.save(normalization_cfg, norm_path)

print(f"✅ Normalization config saved to: {norm_path}")

model = pnc.scm.rain_estimation.two_step_network_with_attention(
    normalization_cfg=normalization_cfg,
    dynamic_input_size=dynamic_input_size,
    metadata_input_size=metadata_input_size,
    d_model=d_model,
    protocol_n_features=protocol_n_features,
    metadata_n_features=metadata_n_features,
    num_protocols=num_protocols,
    window_size=window_size,
    dropout=dropout,
    num_encoder_layers=num_encoder_layers,
    h=h
).to(device)

# Collect all rain rate values across all physical links
all_rr = np.concatenate([link.rain_rate for link in synthetic_dataset])
exp_gamma = scipy.stats.expon.fit(all_rr)[1]

print("Rain Rate Statistics")
print(f"Mean [mm/hr]: {np.mean(all_rr):.4f}")
print(f"Std  [mm/hr]: {np.std(all_rr):.4f}")
print(f"Percentage of wet samples: {100 * np.sum(all_rr > 0) / all_rr.size:.2f}%")
print(f"Percentage of dry samples: {100 * np.sum(all_rr == 0) / all_rr.size:.2f}%")
print(f"Fitted exponential scale (lambda⁻¹): {exp_gamma:.4f}")

opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule)

class RegressionLoss(torch.nn.Module):
    def __init__(self, in_gamma, gamma_s=0.9):
        super(RegressionLoss, self).__init__()
        self.in_gamma = in_gamma
        self.gamma_s = gamma_s

    def forward(self, input, target):
        delta = (target - input) ** 2
        w = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)
        return torch.sum(torch.mean(w * delta, dim=0))
    #def forward(self, input, target, protocol_id, protocol_log_precision):
    #    delta = (target - input) ** 2
    #    w_r = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)

    #    s_p = protocol_log_precision[protocol_id.long()]   # [B]
    #    w_p = torch.exp(s_p).unsqueeze(1)                  # [B,1]

    #    loss = 0.5 * w_r * w_p * delta - 0.5 * s_p.unsqueeze(1)

    #    return torch.sum(torch.mean(loss, dim=0))

ra = ResultsAccumulator()
am = AverageMetric()
ra_val = ResultsAccumulator()
am_val = AverageMetric()

model_path = os.path.join(output_dir, "trained_model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from: {model_path} — skipping training and loss plotting")

else:
    print("🟡 No saved weights found — starting training")

    loss_function_rain_est = RegressionLoss(exp_gamma)
    loss_function_wet_dry = torch.nn.BCELoss()

    model.eval()
    loss_est = 0
    loss_detection = 0
    with torch.no_grad():
        for rain_rate, rsl, tsl, metadata, protocol_id in data_loader:
        #for rain_rate, attenuation, metadata, protocol_id in data_loader:
            m_step = math.floor(rain_rate.shape[1] / window_size)
            for step in range(m_step):
                _rr  = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device) 
                #_att = attenuation[:, step * window_size:(step + 1) * window_size, :].to(device)  # [B, W, 90]

                rain_estimation_detection = model(
                    torch.cat([_rsl, _tsl], dim=-1),  # [B, W, 180]
                    #_att,  # no concat anymore
                    metadata.to(device),
                    protocol_id.to(device),
                )

                rain_hat = rain_estimation_detection[:, :, 0]
                rain_detection = rain_estimation_detection[:, :, 1]

                loss_est += loss_function_rain_est(rain_hat, _rr)
                #loss_est += loss_function_rain_est(
                #rain_hat,
                #_rr,
                #protocol_id.to(device),
                #model.protocol_log_precision
                #)
                loss_detection += loss_function_wet_dry(rain_detection, (_rr > 0.1).float())

    lambda_value = loss_detection / loss_est
    steps_counter = 0

    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")

    model.train()
    for epoch in tqdm(range(n_epochs)):
        am.clear()
        for rain_rate, rsl, tsl, metadata, protocol_id in data_loader:
        #for rain_rate, attenuation, metadata, protocol_id in data_loader:
            m_step = math.floor(rain_rate.shape[1] / window_size)
            for step in range(m_step):
                opt.zero_grad()

                _rr  = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                #_att = attenuation[:, step * window_size:(step + 1) * window_size, :].to(device)  # [B, W, 90]


                rain_estimation_detection = model(
                    torch.cat([_rsl, _tsl], dim=-1),  # [B, W, 180]
                    #_att,  # no concat anymore
                    metadata.to(device),
                    protocol_id.to(device),
                )

                rain_hat = rain_estimation_detection[:, :, 0]
                rain_detection = rain_estimation_detection[:, :, 1]

                loss_est = loss_function_rain_est(rain_hat, _rr)
                #loss_est = loss_function_rain_est(
                #    rain_hat,
                #    _rr,
                #    protocol_id.to(device),
                #    model.protocol_log_precision
                #)
                loss_detection = loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
                loss = lambda_value * loss_est + loss_detection

                loss.backward()
                opt.step()

                steps_counter += 1
                am.add_results(
                    loss=loss.item(),
                    loss_est=loss_est.item(),
                    loss_detection=loss_detection.item()
                )

        scheduler.step() 
        ra.add_results(
            loss=am.get_results("loss"),
            loss_est=am.get_results("loss_est"),
            loss_detection=am.get_results("loss_detection")
        )

        # ----- VALIDATION for this epoch -----
        model.eval()
        am_val.clear()

        with torch.no_grad():
            for rain_rate_v, rsl_, tsl_v, metadata_v, protocol_id_v in val_loader:
                m_step_v = math.floor(rain_rate_v.shape[1] / window_size)

                for step_v in range(m_step_v):
                    _rr_v = rain_rate_v[:, step_v * window_size:(step_v + 1) * window_size].float().to(device)
                    _rsl_v = rsl_[:, step_v * window_size:(step_v + 1) * window_size, :].to(device)
                    _tsl_v = tsl_v[:, step_v * window_size:(step_v + 1) * window_size, :].to(device)

                    out_v = model(
                        torch.cat([_rsl_v, _tsl_v], dim=-1),  # [B, W, 180]
                        metadata_v.to(device),
                        protocol_id_v.to(device),
                    )

                    rain_hat_v = out_v[:, :, 0]
                    rain_detection_v = out_v[:, :, 1]

                    loss_est_v = loss_function_rain_est(rain_hat_v, _rr_v)
                    #loss_est_v = loss_function_rain_est(
                    #    rain_hat_v,
                    #    _rr_v,
                    #    protocol_id_v.to(device),
                    #    model.protocol_log_precision
                    #)
                    loss_det_v = loss_function_wet_dry(rain_detection_v, (_rr_v > 0.1).float())
                    loss_v = lambda_value * loss_est_v + loss_det_v

                    am_val.add_results(
                        loss=loss_v.item(),
                        loss_est=loss_est_v.item(),
                        loss_detection=loss_det_v.item()
                    )

        ra_val.add_results(
            loss=am_val.get_results("loss"),
            loss_est=am_val.get_results("loss_est"),
            loss_detection=am_val.get_results("loss_detection")
        )

        current_val_loss = am_val.get_results("loss")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), best_model_path)

            print(
                f"✅ New best model at epoch {epoch+1} "
                f"(val loss = {best_val_loss:.6f})"
            )
        model.train()
        # ----- END VALIDATION -----

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), model_path)

    print(f"✅ Best validation model loaded and saved to: {model_path}")

    plt.plot(ra.get_results("loss"), label="Total Loss")
    plt.plot(ra.get_results("loss_est"), label="Rain Rate Loss")
    plt.plot(ra.get_results("loss_detection"), label="Wet/Dry Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    figure_name = "loss_plot_over_epochs.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # 1) Wet/Dry (classification) — train vs val
    plt.figure()
    plt.plot(ra.get_results("loss_detection"),     label="Train Wet/Dry Loss")
    plt.plot(ra_val.get_results("loss_detection"), label="Val Wet/Dry Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Wet/Dry Loss per Epoch")
    figure_name = "wet_dry_loss_train_vs_val.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # 2) Rain rate (regression) — train vs val
    plt.figure()
    plt.plot(ra.get_results("loss_est"),     label="Train Rain-Rate Loss")
    plt.plot(ra_val.get_results("loss_est"), label="Val Rain-Rate Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Rain-Rate Loss per Epoch")
    figure_name = "rain_rate_loss_train_vs_val.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # 3) Total loss — train vs val
    plt.figure()
    plt.plot(ra.get_results("loss"),     label="Train Total Loss")
    plt.plot(ra_val.get_results("loss"), label="Val Total Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss per Epoch")
    figure_name = "total_loss_train_vs_val.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    print("-----------------------------------------------")
    print(steps_counter)
    print("-----------------------------------------------")

# =========================================================
# Per-protocol validation
# =========================================================
from sklearn import metrics
from io import StringIO
from pynncml.metrics.results_accumlator import GroupAnalysis

combinations = [("instantaneous", sec) for sec in [10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900]]
combinations.append(("min_max", None))
combinations.append(("average", None))

for samples_type, sampling_interval_in_sec in combinations:

    if samples_type == "instantaneous":
        protocol_id_target = PROTOCOL_MAP[("instantaneous", sampling_interval_in_sec)]
        protocol_name = f"Instantaneous_{sampling_interval_in_sec}_sec"
    elif samples_type == "min_max":
        protocol_id_target = PROTOCOL_MAP["min_max"]
        protocol_name = "Max_Min"
    elif samples_type == "average":
        protocol_id_target = PROTOCOL_MAP["average"]
        protocol_name = "Average"

    protocol_output_dir = os.path.join(base_output_dir, protocol_name)
    os.makedirs(protocol_output_dir, exist_ok=True)

    protocol_val_indices = [
        i for i, lnk in enumerate(expanded_dataset)
        if (lnk.link_index in val_phys) and (lnk.protocol_id == protocol_id_target)
    ]

    validation_dataset_protocol = Subset(wrapped_dataset, protocol_val_indices)
    val_loader_protocol = DataLoader(validation_dataset_protocol, batch_size=batch_size, shuffle=False)

    print(f"📊 Evaluating protocol: {protocol_name}")
    print(f"Validation items: {len(protocol_val_indices)}")

    model.eval()
    ga = GroupAnalysis()

    with torch.no_grad():
        rain_ref_list = []
        rain_hat_list = []
        detection_list = []
        for rain_rate, rsl, tsl, metadata, protocol_id in val_loader_protocol:
        #for rain_rate, attenuation, metadata, protocol_id in val_loader_protocol:
            m_step = math.floor(rain_rate.shape[1] / window_size)


            for step in range(m_step):
                _rr  = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device) 
                #_att = attenuation[:, step * window_size:(step + 1) * window_size, :].to(device)  # [B, W, 90]

                rain_estimation_detection = model(
                    torch.cat([_rsl, _tsl], dim=-1),  # [B, W, 180]
                    #_att,  # no concat anymore
                    metadata.to(device),
                    protocol_id.to(device),
                )

                rain_detection = rain_estimation_detection[:, :, 1]
                rain_hat = rain_estimation_detection[:, :, 0] * torch.round(rain_detection)

                rain_hat_list.append(rain_hat.detach().cpu().numpy())
                rain_ref_list.append(_rr.detach().cpu().numpy())
                ga.append(rain_ref_list[-1], rain_hat_list[-1])
                detection_list.append(torch.round(rain_detection).detach().cpu().numpy())

    actual = np.concatenate(detection_list).flatten()
    predicted = (np.concatenate(rain_ref_list) > 0.1).astype("float").flatten()
    confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=[0, 1])
    max_rain = np.max(np.concatenate(rain_ref_list))
    g_array = np.linspace(0, max_rain, 6)

    print("Results Detection:")
    print(f"Validation Results of Universal Transformer ({protocol_name})")
    print("Accuracy[%]:", 100 * (np.sum(actual == predicted) / actual.size))
    print("F1 Score:", metrics.f1_score(actual, predicted))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title(f"Confusion Matrix ({protocol_name})")
    figure_name = f"confusion_matrix_{protocol_name}.png"
    save_path = os.path.join(protocol_output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    results_path = os.path.join(protocol_output_dir, f"Estimation_Results_{protocol_name}.txt")

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    print("Results Estimation:")
    _ = ga.run_analysis(np.stack([g_array[:-1], g_array[1:]], axis=-1))

    sys.stdout = old_stdout

    with open(results_path, "w") as f:
        f.write(mystdout.getvalue())

    print(f"✅ Results summary saved to: {results_path}")
