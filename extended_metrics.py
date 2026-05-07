
import os
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    MultiScaleStructuralSimilarityIndexMeasure,
)
import torchvision.transforms as transforms
# NeRV-specific imports — these files must be in the same directory
from model_nerv import CustomDataSet, Generator
from utils import PositionalEncoding
from edge_metrics import _parse_arch_from_ckpt_path, _load_state_dict


"""
There will be 6 new metrics added:

1) Peak Signal-to-Noise Ratio

  Compares each pixel in the reconstructed frame against the original and
  converts the average squared error into decibels, 30 dB is acceptable,
  40 dB is near-lossless. Higher = reconstruction closer to ground truth


2) Multi-Scale Structural Similarity Index Measure:

  Runs a sliding window over the frame at multiple scales and scores how
  well edges, contrast, and texture are preserved — not just raw pixel values.
  Closer to 1.0 means a human viewer would notice little to no degradation

3) Mean + P95/P99 Latency (ms)

  Times every single frame decode and sorts the results. Mean shows typical
  speed. P99 would include the outlier. Video needs every frame on
  time, one 200 ms spike at 30 FPS is a visible freeze


4) BPP + MODEL SIZE (KB)

  model size IS the storage cost, but raw KB is unfair across different resolutions and lengths,
  so BPP = (model bits) / (width × height × frames) converts it to bits-per-pixel,
  making a 480p 10-second model comparable to a 1080p 5-minute one.


5) Energy per Frame (mJ) + FPS/Watt

  Samples GPU power draw during inference, multiplies by time-per-frame to
  get joules, converts to millijoules. FPS/Watt then frames it as efficiency
  , how much video you get per unit of battery consumed.


6) Cold Start / Load Time (ms)

  Measures the wall-clock time to deserialise weights from disk into memory
  before a single forward pass can run. IoT devices sleep between triggers
  and pay this cost on every wake — inference latency alone hides it entirely.




────────────────────────────── additional edge-specific metrics that may be optional ────────────────────────────────




7) CPU-Only FPS (measure_cpu_fps)

  Moves the entire model to CPU, runs 100 timed forward passes with no CUDA,
  and reports FPS. Replicates what a Raspberry Pi 4 or ARM embedded board
  actually experiences — no tensor cores, no VRAM, just scalar arithmetic.


"""


try:

    """

    Initialization for NVIDIA energy metrics

    """

    import pynvml
    pynvml.nvmlInit()
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _pynvml_ok  = True

except Exception:

    _pynvml_ok  = False
    print("[WARN] pynvml unavailable — energy metrics will be skipped")



def setup_from_checkpoint(ckpt_path: str, dataset_name: str = "bunny", device_str: str = "cuda"):
  """ Reconstructs everything needed to run metrics from a trained checkpoint path. """

  device    = torch.device(device_str if torch.cuda.is_available() else "cpu")
  arch_args = _parse_arch_from_ckpt_path(ckpt_path)

  # create the positional encoder
  import re
  base  = os.path.basename(os.path.dirname(ckpt_path))
  embed_str = re.search(r"embed([0-9.]+_[0-9]+)", base).group(1)
  PE    = PositionalEncoding(embed_str)

  # load weights
  state_dict = _load_state_dict(ckpt_path)
  model      = Generator(**arch_args).to(device)
  model.load_state_dict(state_dict, strict=True)
  model.eval()

  # val dataloader — all frames, batch=1, no shuffle
  data_dir   = f"./data/{dataset_name.lower()}"
  img_tf     = transforms.ToTensor()
  val_ds = CustomDataSet(data_dir, img_tf, vid_list=[None], frame_gap=1)
  val_loader = torch.utils.data.DataLoader(
      val_ds, batch_size=1, shuffle=False,
      num_workers=0, pin_memory=(device.type == "cuda"),
      drop_last=False,
  )

  print(f"[SETUP] Device     : {device}")
  print(f"[SETUP] Frames     : {len(val_ds)}")
  print(f"[SETUP] Embed len  : {PE.embed_length}")
  if device.type == "cuda":
      print(f"[SETUP] GPU        : {torch.cuda.get_device_name(0)}")

  return model, val_loader, PE, arch_args, device

# PSNR and MS-SSIM

def measure_quality(model, val_dataloader, PE, device):
    psnr_m   = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_m = MultiScaleStructuralSimilarityIndexMeasure(
                   data_range=1.0, kernel_size=11).to(device)
    model.eval()
    with torch.no_grad():
        for data, norm_idx in val_dataloader:
            embed = PE(norm_idx).to(device, non_blocking=True)
            data  = data.to(device, non_blocking=True)
            pred  = model(embed)[-1].clamp(0.0, 1.0)
            gt    = F.adaptive_avg_pool2d(data, pred.shape[-2:]).clamp(0.0, 1.0)
            psnr_m.update(pred, gt)
            msssim_m.update(pred, gt)
    psnr   = psnr_m.compute().item()
    msssim = msssim_m.compute().item()
    psnr_m.reset(); msssim_m.reset()
    return psnr, msssim

# Mean + P95/P99 Latency

def measure_latency_distribution(model, PE, device, num_frames=300,
                                  n_warmup=50, n_measure=500):


  model.eval()

  # Build a single dummy embed from frame index 0.5 (middle of video)
  # Shape must match what PE produces — same as training
  mid_idx = torch.tensor([0.5])
  dummy_embed = PE(mid_idx).unsqueeze(0).to(device)
  print(f"[DEBUG] Dummy embed shape: {dummy_embed.shape} | dtype: {dummy_embed.dtype}")
  # --- Warm-up ---
  with torch.no_grad():
      for _ in range(n_warmup):
        _= model(dummy_embed)

  if device.type == "cuda":
      torch.cuda.synchronize()

  # --- Timed loop using CUDA Events ---
  latencies_ms = []
  with torch.no_grad():
      for _ in range(n_measure):
          if device.type == "cuda":
              start_evt = torch.cuda.Event(enable_timing=True)
              end_evt   = torch.cuda.Event(enable_timing=True)
              start_evt.record()
              _ = model(dummy_embed)
              end_evt.record()
              torch.cuda.synchronize()
              latencies_ms.append(start_evt.elapsed_time(end_evt))
          else:
              # CPU fallback — perf_counter is fine since CPU is synchronous
              t0 = time.perf_counter()
              _ = model(dummy_embed)
              latencies_ms.append((time.perf_counter() - t0) * 1000.0)

  arr = np.array(latencies_ms)
  return {
      "mean_ms" : float(np.mean(arr)),
      "std_ms"  : float(np.std(arr)),
      "p50_ms"  : float(np.percentile(arr, 50)),
      "p95_ms"  : float(np.percentile(arr, 95)),
      "p99_ms"  : float(np.percentile(arr, 99)),
      "min_ms"  : float(np.min(arr)),
      "max_ms"  : float(np.max(arr)),
  }

# BPP + Model Size (KB)

def measure_bpp_and_size(model, val_dataloader, PE, device, ckpt_path):


      # --- Get disk size of checkpoint ---
  assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
  size_bytes = os.path.getsize(ckpt_path)
  size_KB    = size_bytes / 1024.0
  size_MB    = size_bytes / (1024.0 ** 2)
  size_bits  = size_bytes * 8

    # --- Infer H, W from one forward pass ---
  model.eval()
  with torch.no_grad():
      data, norm_idx = next(iter(val_dataloader))
      embed_input = PE(norm_idx).to(device)
      output_list = model(embed_input)
      _, _, H, W  = output_list[-1].shape   # (B, C, H, W)

    # --- T = total frames in validation set ---
    # val_dataloader with batch_size=1 → len(val_dataloader) == T
  T = len(val_dataloader.dataset)

  bpp = size_bits / (T * H * W)

  return {
      "size_KB"  : size_KB,
      "size_MB"  : size_MB,
      "T"        : T,
      "H"        : H,
      "W"        : W,
      "bpp"      : bpp,
  }



# Cold Start / Load Time
def measure_cold_start(ckpt_path, arch_args, device, n_runs=10):
  import gc

  load_times_ms = []
  for _ in range(n_runs):
      gc.collect()
      if device.type == "cuda":
          torch.cuda.empty_cache()
          torch.cuda.synchronize()
      t0 = time.perf_counter()
      ckpt = torch.load(ckpt_path, map_location="cpu")
      state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
      m = Generator(**arch_args)
      m.load_state_dict(state_dict, strict=True)
      m.eval()
      m = m.to(device)
      if device.type == "cuda":
          torch.cuda.synchronize()
      t1 = time.perf_counter()
      load_times_ms.append((t1 - t0) * 1000.0)
      del m, state_dict, ckpt

  arr = np.array(load_times_ms)
  return {
      "load_mean_ms": round(float(np.mean(arr)), 2),
      "load_std_ms" : round(float(np.std(arr)),  2),
      "load_min_ms" : round(float(np.min(arr)),  2),
      "load_max_ms" : round(float(np.max(arr)),  2),
  }
# Energy/frame + FPS/Watt

def measure_energy(model, PE, device, num_frames=300, n_warmup=20):

  if not _pynvml_ok:
      print("[SKIP] pynvml not available — energy metrics skipped")
      return {
          "avg_power_W"        : None,
          "energy_per_frame_mJ": None,
          "fps_per_watt"       : None,
      }

  model.eval()

  # Build frame index tensor: evenly spaced normalised indices
  norm_indices = torch.linspace(0, 1, num_frames).unsqueeze(1)  # (T, 1)

  embeds = [
      PE(norm_indices[i]).unsqueeze(0).to(device) 
      for i in range(num_frames)
  ]
  
  # --- Warm-up ---
  with torch.no_grad():
      for i in range(n_warmup):
          _ = model(embeds[i % num_frames])
  if device.type == "cuda":
      torch.cuda.synchronize()

  # --- Measure idle power baseline (2 seconds) ---
  idle_readings = []
  stop_idle = threading.Event()
  def _poll_idle():
      while not stop_idle.is_set():
          try:
              idle_readings.append(
                  pynvml.nvmlDeviceGetPowerUsage(_gpu_handle) / 1000.0)
          except Exception:
              pass
          time.sleep(0.005)
  t_idle = threading.Thread(target=_poll_idle, daemon=True)
  t_idle.start()
  time.sleep(2.0)
  stop_idle.set()
  t_idle.join()
  idle_power_W = float(np.mean(idle_readings)) if idle_readings else 0.0

  # --- Decode + power polling ---
  power_readings = []
  stop_flag = threading.Event()

  def _poll_power():
      while not stop_flag.is_set():
          try:
              power_readings.append(
                  pynvml.nvmlDeviceGetPowerUsage(_gpu_handle) / 1000.0)
          except Exception:
              pass
          time.sleep(0.005)   # poll every 5 ms

  poll_thread = threading.Thread(target=_poll_power, daemon=True)
  poll_thread.start()

  t_start = time.perf_counter()
  with torch.no_grad():
      for i in range(num_frames):
          _ = model(embeds[i])
  if device.type == "cuda":
      torch.cuda.synchronize()
  elapsed_s = time.perf_counter() - t_start

  stop_flag.set()
  poll_thread.join()

  avg_power_W      = float(np.mean(power_readings)) if power_readings else 0.0
  net_power_W      = max(avg_power_W - idle_power_W, 0.0)  # subtract idle
  total_energy_J   = net_power_W * elapsed_s
  energy_per_frame_mJ = (total_energy_J / num_frames) * 1000.0
  fps              = num_frames / elapsed_s
  fps_per_watt     = fps / avg_power_W if avg_power_W > 0 else None

  return {
      "avg_power_W"        : round(avg_power_W, 3),
      "idle_power_W"       : round(idle_power_W, 3),
      "net_power_W"        : round(net_power_W, 3),
      "energy_per_frame_mJ": round(energy_per_frame_mJ, 4),
      "fps_during_energy"  : round(fps, 2),
      "fps_per_watt"       : round(fps_per_watt, 3) if fps_per_watt else None,
  }


def measure_cpu_fps(model, PE, device, n_warmup=10, n_measure=100):
    model_cpu = model.cpu().eval()
    dummy_embed = PE(torch.tensor([0.5])).unsqueeze(0)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(dummy_embed)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _ = model_cpu(dummy_embed)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    model.to(device)  # restore to GPU after measurement
    if device.type == "cuda":
        torch.cuda.synchronize()

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    return {
        "cpu_fps"    : round(1000.0 / mean_ms, 2),
        "cpu_mean_ms": round(mean_ms, 3),
        "cpu_std_ms" : round(float(np.std(arr)), 3),
        "cpu_p95_ms" : round(float(np.percentile(arr, 95)), 3),
        "cpu_p99_ms" : round(float(np.percentile(arr, 99)), 3),
    }


def run_all_metrics(ckpt_path, dataset_name="bunny", num_frames=132):
    model, val_loader, PE, arch_args, device = setup_from_checkpoint(
        ckpt_path, dataset_name)

    results = {}

    print("\n" + "="*55)
    print("  NeRV Extended Edge Metrics — Google Colab T4")
    print("="*55)

    print("\n[1/6] PSNR + MS-SSIM...")
    psnr, msssim = measure_quality(model, val_loader, PE, device)
    results["psnr_dB"] = round(psnr, 4)
    results["ms_ssim"] = round(msssim, 6)
    print(f"      PSNR={psnr:.2f} dB  MS-SSIM={msssim:.4f}")

    print("\n[2/6] Latency distribution (500 runs)...")
    lat = measure_latency_distribution(model, PE, device)
    results.update({f"lat_{k}": v for k, v in lat.items()})
    print(f"      Mean={lat['mean_ms']:.3f}ms  P95={lat['p95_ms']:.3f}ms  P99={lat['p99_ms']:.3f}ms")

    print("\n[3/6] BPP + model size...")
    bpp = measure_bpp_and_size(model, val_loader, PE, device, ckpt_path)
    results.update(bpp)
    print(f"      {bpp['size_KB']:.1f} KB  BPP={bpp['bpp']:.6f}  ({bpp['T']} frames @ {bpp['H']}x{bpp['W']})")

    print("\n[4/6] Cold start (10 runs)...")
    cs = measure_cold_start(ckpt_path, arch_args, device)
    results.update(cs)
    print(f"      Mean={cs['load_mean_ms']:.1f}ms  Max={cs['load_max_ms']:.1f}ms")

    print("\n[5/6] Energy per frame + FPS/Watt...")
    energy = measure_energy(model, PE, device, num_frames=num_frames)
    results.update(energy)
    if energy["energy_per_frame_mJ"] is not None:
        print(f"      Net={energy['net_power_W']:.2f}W  {energy['energy_per_frame_mJ']:.4f}mJ/frame  {energy['fps_per_watt']} FPS/W")
    else:
        print("      Skipped — pynvml unavailable")

    print("\n[6/6] CPU-only FPS (100 runs)...")
    cpu = measure_cpu_fps(model, PE, device)
    results.update(cpu)
    print(f"      CPU FPS={cpu['cpu_fps']:.2f}  mean={cpu['cpu_mean_ms']:.1f}ms  P99={cpu['cpu_p99_ms']:.1f}ms")

    # derived ratio
    if results.get("psnr_dB") and results.get("lat_mean_ms"):
        results["psnr_per_latency"] = round(results["psnr_dB"] / results["lat_mean_ms"], 4)

    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    for k, v in results.items():
        print(f"  {k:<30s}: {v}")
    print("="*55)
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    base   = "output/smoke_test/bunny"
    folder = os.listdir(base)[0]
    CKPT   = f"{base}/{folder}/model_val_best.pth"
    print(f"Checkpoint: {CKPT}")
    run_all_metrics(CKPT, dataset_name="bunny", num_frames=132)
