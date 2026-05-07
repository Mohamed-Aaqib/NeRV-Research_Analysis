
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

# NeRV-specific imports — these files must be in the same directory
from model_nerv import CustomDataSet, Generator
from utils import PositionalEncoding
from edge_metrics import _parse_arch_from_ckpt_path, _load_state_dict


"""
There will 9 new metrics added:

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
  PE        = PositionalEncoding(arch_args["embed_length"])

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
  val_ds     = CustomDataSet(data_dir, img_tf, vid_list=None, frame_gap=1)
  val_loader = torch.utils.data.DataLoader(
      val_ds, batch_size=1, shuffle=False,
      num_workers=2, pin_memory=(device.type == "cuda"),
      drop_last=False,
  )

  print(f"[SETUP] Device     : {device}")
  print(f"[SETUP] Frames     : {len(val_ds)}")
  print(f"[SETUP] Embed len  : {PE.embed_length}")
  if device.type == "cuda":
      print(f"[SETUP] GPU        : {torch.cuda.get_device_name(0)}")

  return model, val_loader, PE, arch_args, device

# PSNR and MS-SSIM


  model.eval()
  with torch.no_grad():
      for data, norm_idx in val_dataloader:
          embed_input = PE(norm_idx)
          data = data.to(device, non_blocking=True)
          embed_input = embed_input.to(device, non_blocking=True)

          output_list = model(embed_input)
          pred = output_list[-1]
          # resize GT to match pred resolution (matches train_nerv.py logic)
          gt   = F.adaptive_avg_pool2d(data, pred.shape[-2:])

          # clamp to [0,1]
          pred = pred.clamp(0.0, 1.0)
          gt   = gt.clamp(0.0, 1.0)

          psnr_metric.update(pred, gt)
          msssim_metric.update(pred, gt)

  mean_psnr   = psnr_metric.compute().item()
  mean_msssim = msssim_metric.compute().item()

  psnr_metric.reset()
  msssim_metric.reset()

  return mean_psnr, mean_msssim


# Mean + P95/P99 Latency

def measure_latency_distribution(model, PE, device, num_frames=300,
                                  n_warmup=50, n_measure=500):


  model.eval()

  # Build a single dummy embed from frame index 0.5 (middle of video)
  # Shape must match what PE produces — same as training
  mid_idx  = torch.tensor([[0.5]])
  dummy_embed = PE(mid_idx).to(device)

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

  import gc

  load_times_ms = []

  for _ in range(n_runs):
      gc.collect()
      if device.type == "cuda":
          torch.cuda.empty_cache()
          torch.cuda.synchronize()

      t0 = time.perf_counter()

      # Step 1 — deserialise weights from disk to CPU RAM
      ckpt = torch.load(ckpt_path, map_location="cpu")
      if isinstance(ckpt, dict):
          state_dict = ckpt.get("state_dict", ckpt)
      else:
          state_dict = ckpt

      # Step 2 — build model and load weights (CPU)
      m = model_constructor(**arch_args)
      m.load_state_dict(state_dict, strict=True)
      m.eval()

      # Step 3 — transfer weights to GPU (PCIe memcpy)
      m = m.to(device)
      if device.type == "cuda":
          torch.cuda.synchronize()   # wait for all GPU memcpy to finish

      t1 = time.perf_counter()
      load_times_ms.append((t1 - t0) * 1000.0)

      del m, state_dict, ckpt   # free memory before next run

  arr = np.array(load_times_ms)
  return {
      "load_mean_ms" : float(np.mean(arr)),
      "load_std_ms"  : float(np.std(arr)),
      "load_min_ms"  : float(np.min(arr)),
      "load_max_ms"  : float(np.max(arr)),
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
  embeds = [PE(norm_indices[i:i+1]).to(device) for i in range(num_frames)]

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

