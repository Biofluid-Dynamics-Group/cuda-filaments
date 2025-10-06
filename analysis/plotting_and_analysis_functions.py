"""
Reusable plotting & analysis utilities for cuda-filaments simulations.

Core capabilities:
 1. Kymograph plotting (handles cell-cell gaps by using true azimuths).
 2. Single-frame 3D rendering (top-down or isometric).
 3. Top-down animation export (MP4).
 4. Wave directionality analysis over the last period.

Assumptions about file naming (base_path points to common prefix):
   <base_path>_true_states.dat     (phase data)
   <base_path>_seg_states.dat      (segment position data)
   <base_path>_fil_references.dat  (single line: x1 y1 z1 x2 y2 z2 ...)

Phase file layout:
   col 0: simulation step counter
   col 1: (unused / placeholder)
   col 2 .. 2+N-1: phase values ψ for each filament

Segment file layout:
   col 0: simulation step counter (matching phase file)
   remaining: flattened (N * num_segs * 3) entries per row.

Basal references format:
   Single line with 3*N numbers (x,y,z for each filament).

Author: Auto-generated utility module.
"""

from __future__ import annotations
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

try:
    import cmcrameri.cm as cmc
    DEFAULT_CMAP = cmc.romaO
except Exception:  # fallback if cmcrameri not installed
    DEFAULT_CMAP = plt.cm.twilight

# ----------------------------- Data Structures -----------------------------

@dataclass
class SimulationData:
    base_path: str
    phases: np.ndarray          # shape (T, N)
    times: np.ndarray           # shape (T,) - in units of periods/cycles
    seg_positions: np.ndarray   # shape (T, N, S, 3)
    basal_pos: np.ndarray       # shape (N, 3)
    basal_phi: np.ndarray       # shape (N,)
    order_idx: np.ndarray       # sorting indices (ascending φ)
    num_segs: int
    num_steps: Optional[int] = None  # steps per period
    sphere_radius: Optional[float] = None

# ----------------------------- Loading Helpers -----------------------------

def _infer_num_fils_from_phase(phase_file: str) -> int:
    with open(phase_file, "r") as f:
        first = f.readline().strip().split()
    width = len(first)
    # Columns: time, ?, then N phases
    if width < 3:
        raise ValueError("Phase file appears too narrow to contain phases.")
    return (width - 2)//2

def _load_basal_positions(ref_file: str, expected_N: Optional[int]=None) -> np.ndarray:
    raw = np.loadtxt(ref_file).ravel()
    if raw.size % 3 != 0:
        raise ValueError(f"Reference file length {raw.size} not divisible by 3.")
    N = raw.size // 3
    if expected_N and N != expected_N:
        print(f"[warn] Expected {expected_N} basal entries, found {N}; proceeding with file value.")
    return raw.reshape(N, 3)

def load_simulation(base_path: str,
                    num_steps: Optional[int] = None,
                    sphere_radius: Optional[float] = None,
                    num_segs: Optional[int] = None) -> SimulationData:
    """
    Load simulation data given the common prefix base_path.
    
    Args:
        base_path: Common file prefix (without _true_states.dat etc.)
        num_steps: Steps per period for time normalization. If None, uses simple 0-1 normalization.
        sphere_radius: Sphere radius. If None, estimated from basal positions.
        num_segs: Segments per filament. If None, inferred from data.
    """
    phase_file = f"{base_path}_true_states.dat"
    seg_file   = f"{base_path}_seg_states.dat"
    ref_file   = f"{base_path}_fil_references.dat"

    if not os.path.isfile(phase_file):
        raise FileNotFoundError(phase_file)
    if not os.path.isfile(seg_file):
        raise FileNotFoundError(seg_file)
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(ref_file)

    num_fils = _infer_num_fils_from_phase(phase_file)

    phase_data = np.loadtxt(phase_file)
    times_raw = phase_data[:, 0]  # simulation step counter
    T = phase_data.shape[0]
    phases = np.mod(phase_data[:, 2:2+num_fils], 2*np.pi)  # only psi 1

    # Time normalization: convert simulation steps to periods/cycles
    if num_steps is not None and num_steps > 0:
        times = times_raw / num_steps  # time in units of periods
    else:
        # Fallback: simple 0-1 normalization
        times = times_raw / max(times_raw) if times_raw.ptp() > 0 else times_raw
        print("[warn] No num_steps provided, using simple 0-1 time normalization")

    # Load segment positions
    seg_data = np.loadtxt(seg_file)
    if seg_data.shape[0] != T:
        raise ValueError("Phase and segment files have mismatched time length.")
    flat_len = seg_data.shape[1] - 1
    if num_segs is None:
        # Solve S from flat_len = num_fils * S * 3
        if flat_len % (num_fils * 3) != 0:
            raise ValueError("Cannot infer num_segs (inconsistent flattened length).")
        num_segs = flat_len // (num_fils * 3)
    seg_positions = seg_data[:, 1:].reshape(T, num_fils, num_segs, 3)

    basal_pos = _load_basal_positions(ref_file, num_fils)
    x, y = basal_pos[:,0], basal_pos[:,1]
    basal_phi = np.mod(np.arctan2(y, x), 2*np.pi)
    order_idx = np.argsort(basal_phi)

    # If sphere_radius not supplied, estimate as mean radial norm of basal points
    if sphere_radius is None:
        sphere_radius = float(np.mean(np.linalg.norm(basal_pos, axis=1)))

    return SimulationData(
        base_path=base_path,
        phases=phases,
        times=times,
        seg_positions=seg_positions,
        basal_pos=basal_pos,
        basal_phi=basal_phi,
        order_idx=order_idx,
        num_segs=num_segs,
        num_steps=num_steps,
        sphere_radius=sphere_radius
    )

# ----------------------------- Gap Detection -----------------------------

@dataclass
class GapInfo:
    has_gap: bool
    largest_gap_width: float
    largest_gap_indices: Tuple[int,int]
    median_spacing: float
    ratio: float

def detect_gaps(basal_phi: np.ndarray,
                gap_factor: float = 1.1,
                min_gap_abs: float = 0.0) -> GapInfo:
    """Detect whether a pronounced cell-cell gap exists."""
    idx = np.argsort(basal_phi)
    phi_sorted = basal_phi[idx]
    phi_ext = np.concatenate([phi_sorted, [phi_sorted[0] + 2*np.pi]])
    dphi = np.diff(phi_ext)
    max_i = int(np.argmax(dphi))
    largest = float(dphi[max_i])
    med = float(np.median(dphi))
    has_gap = largest > max(gap_factor * med, min_gap_abs)
    next_i = (max_i + 1) % (len(phi_sorted))
    return GapInfo(has_gap, largest, (idx[max_i], idx[next_i]), med, largest / med if med>0 else np.inf)

# ----------------------------- Kymograph -----------------------------

def plot_kymograph(base_path: str,
                   sim: Optional[SimulationData]=None,
                   num_steps: Optional[int]=None,
                   use_phi_axis: bool = True,
                   cmap=DEFAULT_CMAP,
                   show: bool = True,
                   save: bool = True,
                   gap_factor: float = 1.6,
                   min_gap_abs: float = 0.15,
                   fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot kymograph of phases vs time. If use_phi_axis, y-axis = true azimuths φ
    so gaps appear as blank regions (no interpolation).
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)

    gap_info = detect_gaps(sim.basal_phi, gap_factor, min_gap_abs)

    phases_sorted = sim.phases[:, sim.order_idx]   # shape (T, N_sorted)
    phi_sorted = sim.basal_phi[sim.order_idx]
    T, N = phases_sorted.shape

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = fig_ax

    X, Y = np.meshgrid(sim.times, phi_sorted) if use_phi_axis else np.meshgrid(sim.times, np.arange(N))
    # Need same shape: transpose phases to (N, T)
    im = ax.pcolormesh(X, Y, phases_sorted.T, cmap=cmap, shading='auto', vmin=0, vmax=2*np.pi)

    if use_phi_axis:
        ax.set_ylabel("azimuth φ (rad)")
    else:
        ax.set_ylabel("cilia index (sorted)")

    # Update x-axis label based on whether we have num_steps
    if sim.num_steps is not None:
        ax.set_xlabel("t/T")
    else:
        ax.set_xlabel("normalized time")
        
    title = "Kymograph"
    if gap_info.has_gap:
        title += f" (gap Δφ={gap_info.largest_gap_width:.2f} rad)"
    ax.set_title(title)
    if use_phi_axis:
        ax.set_ylim(0, 2*np.pi)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\psi_1$")
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])

    if show:
        plt.tight_layout()
        plt.show()

    if save:
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        suffix = "_kymograph_phi.png" if use_phi_axis else "_kymograph_idx.png"
        out_path = out_dir / (Path(base_path).name + suffix)
        fig.savefig(out_path.as_posix(), dpi=180)
        print(f"[info] Saved kymograph to {out_path}")
    return fig, ax, gap_info

# ----------------------------- 3D Frame Plot -----------------------------

def plot_frame(base_path: str,
               sim: Optional[SimulationData]=None,
               num_steps: Optional[int]=None,
               frame: str | int = "first",
               view: str = "top",
               color_by_phase: bool = True,
               cmap=DEFAULT_CMAP,
               show: bool = True,
               save: bool = True,
               fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot a single frame (first, last or int index) in 3D.
    view: 'top' (elev=90, azim=-90) or 'iso' (elev=30, azim=-45).
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)

    if frame == "first":
        f = 0
    elif frame == "last":
        f = sim.phases.shape[0] - 1
    else:
        f = int(frame)
    f = max(0, min(f, sim.phases.shape[0]-1))

    if fig_ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = fig_ax

    # Axes limits
    XYZ = sim.seg_positions[f]
    x_all = sim.seg_positions[...,0]
    y_all = sim.seg_positions[...,1]
    z_all = sim.seg_positions[...,2]
    margin = 2.0
    ax.set_xlim(np.min(x_all)-margin, np.max(x_all)+margin)
    ax.set_ylim(np.min(y_all)-margin, np.max(y_all)+margin)
    ax.set_zlim(np.min(z_all)-margin, np.max(z_all)+margin)

    if view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=30, azim=-45)

    norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)
    for i in range(XYZ.shape[0]):
        col = cmap(norm(sim.phases[f, i])) if color_by_phase else "k"
        ax.plot(XYZ[i,:,0], XYZ[i,:,1], XYZ[i,:,2], '-', lw=2, color=col)

    # Light sphere surface (optional)
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = sim.sphere_radius * np.outer(np.cos(u), np.sin(v))
    ys = sim.sphere_radius * np.outer(np.sin(u), np.sin(v))
    zs = sim.sphere_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='grey', alpha=0.12, linewidth=0)

    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    
    # Update title to show actual time if we have num_steps
    if sim.num_steps is not None:
        time_val = sim.times[f]
        ax.set_title(f"Frame {f} (t={time_val:.2f} periods, {view})")
    else:
        ax.set_title(f"Frame {f} ({view})")

    if color_by_phase:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label(r"$\psi_1$")

    if show:
        plt.tight_layout()
        plt.show()

    if save:
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        suffix = f"_frame_{frame}_{view}.png"
        out_path = out_dir / (Path(base_path).name + suffix)
        fig.savefig(out_path.as_posix(), dpi=180)
        print(f"[info] Saved frame to {out_path}")
    return fig, ax

# ----------------------------- Animation -----------------------------

def make_topdown_video(base_path: str,
                       sim: Optional[SimulationData]=None,
                       num_steps: Optional[int]=None,
                       out_path: Optional[str]=None,
                       stride: int = 1,
                       fps: int = 25,
                       cmap=DEFAULT_CMAP,
                       progress: bool = True,
                       dpi: int = 180):
    """
    Generate MP4 top-down animation colored by phase.
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    if out_path is None:
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / (Path(base_path).name + "_topdown.mp4")
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)

    times = range(0, sim.phases.shape[0], stride)
    norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Bounds
    x_all = sim.seg_positions[...,0]; y_all = sim.seg_positions[...,1]; z_all = sim.seg_positions[...,2]
    margin = 2.0
    ax.set_xlim(np.min(x_all)-margin, np.max(x_all)+margin)
    ax.set_ylim(np.min(y_all)-margin, np.max(y_all)+margin)
    ax.set_zlim(np.min(z_all)-margin, np.max(z_all)+margin)
    ax.view_init(elev=90, azim=-90)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_title("Top-down animation")

    # Sphere
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = sim.sphere_radius * np.outer(np.cos(u), np.sin(v))
    ys = sim.sphere_radius * np.outer(np.sin(u), np.sin(v))
    zs = sim.sphere_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='grey', alpha=0.12, linewidth=0)

    lines = [ax.plot([], [], [], '-', lw=2)[0] for _ in range(sim.phases.shape[1])]

    def init():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return lines

    def update(f):
        XYZ = sim.seg_positions[f]
        for i, ln in enumerate(lines):
            ln.set_data(XYZ[i,:,0], XYZ[i,:,1])
            ln.set_3d_properties(XYZ[i,:,2])
            ln.set_color(cmap(norm(sim.phases[f, i])))
        return lines

    ani = FuncAnimation(fig, update, frames=times, init_func=init, blit=False)
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="cuda-filaments"))

    if progress:
        try:
            from tqdm import tqdm
            with tqdm(total=len(times), desc=f"Rendering {out_path.name}") as pbar:
                ani.save(out_path.as_posix(), writer=writer,
                         dpi=dpi,
                         progress_callback=lambda i, n: pbar.update(1))
        except ImportError:
            ani.save(out_path.as_posix(), writer=writer, dpi=dpi)
    else:
        ani.save(out_path.as_posix(), writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"[info] Saved video to {out_path}")
    return out_path

# ----------------------------- Wave Direction Analysis -----------------------------

@dataclass
class WaveDirectionResult:
    percent_positive: float
    percent_negative: float
    percent_stationary: float
    labels: np.ndarray        # shape (N,) values in {-1,0,1}
    velocity: np.ndarray      # shape (N,) average propagation indicator
    gap_info: GapInfo

def analyze_wave_direction(base_path: str,
                           sim: Optional[SimulationData]=None,
                           num_steps: Optional[int]=None,
                           period_steps: Optional[int]=None,
                           tol: float = 1e-6,
                           smooth: int = 3,
                           gap_factor: float = 1.1,
                           min_gap_abs: float = 0.01) -> WaveDirectionResult:
    """
    Estimate propagation direction sign from ψ(φ, t):
       dy/dt sign ~ - (∂ψ/∂t)/(∂ψ/∂φ).
    Uses final 'period_steps' samples; if None uses all.
    
    Uses complex exponential method to handle phase periodicity correctly.
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)

    gap_info = detect_gaps(sim.basal_phi, gap_factor, min_gap_abs)

    phases_sorted = sim.phases[:, sim.order_idx]  # (T, N)
    phi_sorted = sim.basal_phi[sim.order_idx]

    if period_steps is None or period_steps <= 0 or period_steps > phases_sorted.shape[0]:
        window = phases_sorted
    else:
        window = phases_sorted[-period_steps:]

    # Use complex exponential method to handle periodicity
    # Let f(y,t) = exp(i * ψ(y,t)). Then:
    # df/dt = i * (∂ψ/∂t) * f  =>  ∂ψ/∂t = -i * (df/dt) / f
    # df/dy = i * (∂ψ/∂y) * f  =>  ∂ψ/∂y = -i * (df/dy) / f
    
    psi = window.T  # Shape: (N, W) where W is time window length
    f_array = np.exp(1j * psi)  # Complex exponential
    
    # Compute gradients of f
    df_dy = np.gradient(f_array, phi_sorted, axis=0)  # gradient along spatial dimension
    df_dt = np.gradient(f_array, axis=1)              # gradient along time dimension
    
    # Velocity dy/dt = - (∂ψ/∂t) / (∂ψ/∂y)
    velocity = -df_dt / (df_dy + 1e-14)

    # Average direction at each position over the time window
    vel_mean = np.mean(velocity, axis=1)  # average over time (axis=1)

    # Take real part for directionality (imaginary part should be small)
    vel_real = np.real(vel_mean)

    # Optional smoothing (simple moving average in φ)
    if smooth > 1:
        kernel = np.ones(smooth)/smooth
        vel_real = np.convolve(vel_real, kernel, mode='same')

    # Classify direction
    labels = np.where(vel_real > tol, 1, np.where(vel_real < -tol, -1, 0))
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == -1)
    num_sta = np.sum(labels == 0)
    N = labels.size

    return WaveDirectionResult(
        percent_positive = 100.0 * num_pos / N,
        percent_negative = 100.0 * num_neg / N,
        percent_stationary = 100.0 * num_sta / N,
        labels = labels,
        velocity = vel_real,
        gap_info = gap_info
    )


# ----------------------------- Basal Position Plot -----------------------------

def plot_basal_positions(base_path: str,
                        sim: Optional[SimulationData]=None,
                        num_steps: Optional[int]=None,
                        color_by: str = "azimuth",  # "azimuth", "index", or "uniform"
                        show_gap: bool = True,
                        cmap=DEFAULT_CMAP,
                        show: bool = True,
                        save: bool = True,
                        gap_factor: float = 1.6,
                        min_gap_abs: float = 0.15,
                        fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot basal positions of cilia (top-down view).
    
    Args:
        color_by: "azimuth" (angle-based), "index" (sequential), or "uniform" (all same color)
        show_gap: If True and gap detected, highlight the largest gap
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)

    gap_info = detect_gaps(sim.basal_phi, gap_factor, min_gap_abs)
    
    x, y = sim.basal_pos[:, 0], sim.basal_pos[:, 1]
    
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = fig_ax

    # Color mapping
    if color_by == "azimuth":
        colors = sim.basal_phi
        cbar_label = r"azimuth $\phi$ (rad)"
        vmin, vmax = 0, 2*np.pi
    elif color_by == "index":
        colors = np.arange(len(x))
        cbar_label = "cilia index"
        vmin, vmax = 0, len(x)-1
    else:  # uniform
        colors = 'blue'
        cbar_label = None
        vmin = vmax = None

    # Main scatter plot
    if color_by == "uniform":
        sc = ax.scatter(x, y, c=colors, s=30, edgecolor='k', linewidth=0.3)
    else:
        sc = ax.scatter(x, y, c=colors, cmap=cmap, s=30, edgecolor='k', 
                       linewidth=0.3, vmin=vmin, vmax=vmax)

    # Highlight gap if requested and detected
    if show_gap and gap_info.has_gap:
        # Get the two cilia bounding the largest gap
        left_idx, right_idx = gap_info.largest_gap_indices
        ax.scatter([x[left_idx], x[right_idx]], [y[left_idx], y[right_idx]],
                  c='red', s=100, marker='*', edgecolor='black', linewidth=1,
                  label=f'Gap boundaries (Δφ={gap_info.largest_gap_width:.3f} rad)',
                  zorder=5)
        
        # Draw gap arc
        gap_start = sim.basal_phi[left_idx]
        gap_width = gap_info.largest_gap_width
        gap_end = gap_start + gap_width
        
        # Handle wraparound
        if gap_end > 2*np.pi:
            # Split into two arcs
            arc1 = np.linspace(gap_start, 2*np.pi, 50)
            arc2 = np.linspace(0, gap_end - 2*np.pi, 50)
            arc_angles = np.concatenate([arc1, arc2])
        else:
            arc_angles = np.linspace(gap_start, gap_end, 100)
        
        arc_x = sim.sphere_radius * np.cos(arc_angles)
        arc_y = sim.sphere_radius * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color='red', lw=3, alpha=0.7, 
               label='Largest gap', zorder=4)

    # Sphere outline
    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = sim.sphere_radius * np.cos(theta)
    circle_y = sim.sphere_radius * np.sin(theta)
    ax.plot(circle_x, circle_y, color='grey', lw=2, alpha=0.6, 
           label='Sphere equator')

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    ax.grid(True, alpha=0.3)
    
    title = f"Basal positions (N={len(x)})"
    if gap_info.has_gap:
        title += f"\nGap: Δφ={gap_info.largest_gap_width:.3f} rad (ratio={gap_info.ratio:.2f})"
    ax.set_title(title)

    # Colorbar for non-uniform coloring
    if color_by != "uniform" and cbar_label:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label(cbar_label)
        if color_by == "azimuth":
            cbar.set_ticks([0, np.pi, 2*np.pi])
            cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    # Legend for gap info
    if show_gap and gap_info.has_gap:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)

    if show:
        plt.tight_layout()
        plt.show()

    if save:
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        suffix = f"_basal_{color_by}.png"
        out_path = out_dir / (Path(base_path).name + suffix)
        fig.savefig(out_path.as_posix(), dpi=180, bbox_inches='tight')
        print(f"[info] Saved basal positions to {out_path}")
        
    return fig, ax, gap_info

def plot_gap_analysis(base_path: str,
                     sim: Optional[SimulationData]=None,
                     num_steps: Optional[int]=None,
                     gap_factor: float = 1.6,
                     min_gap_abs: float = 0.15,
                     zoom_factor: float = 1.5,
                     show: bool = True,
                     save: bool = True):
    """
    Create a detailed gap analysis plot with global view + zoomed gap region.
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)

    gap_info = detect_gaps(sim.basal_phi, gap_factor, min_gap_abs)
    
    if not gap_info.has_gap:
        print("[info] No significant gap detected, showing global view only")
        return plot_basal_positions(base_path, sim=sim, show_gap=False, 
                                   show=show, save=save)

    x, y = sim.basal_pos[:, 0], sim.basal_pos[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Global view with gap highlighted
    plot_basal_positions(base_path, sim=sim, show_gap=True, 
                        fig_ax=(fig, ax1), show=False, save=False,
                        gap_factor=gap_factor, min_gap_abs=min_gap_abs)
    ax1.set_title("Global view")
    
    # Right: Zoomed view around gap
    left_idx, right_idx = gap_info.largest_gap_indices
    gap_center_phi = sim.basal_phi[left_idx] + gap_info.largest_gap_width/2
    gap_center_phi = np.mod(gap_center_phi, 2*np.pi)
    
    # Find cilia within zoom window
    window_width = zoom_factor * gap_info.largest_gap_width
    def angular_distance(a, b):
        d = np.abs(a - b)
        return np.minimum(d, 2*np.pi - d)
    
    mask = angular_distance(sim.basal_phi, gap_center_phi) <= window_width/2
    
    # Plot zoomed region
    zoom_x, zoom_y = x[mask], y[mask]
    zoom_phi = sim.basal_phi[mask]
    
    sc = ax2.scatter(zoom_x, zoom_y, c=zoom_phi, cmap=DEFAULT_CMAP, 
                    s=60, edgecolor='k', linewidth=0.5, vmin=0, vmax=2*np.pi)
    
    # Highlight gap boundaries in zoom
    if left_idx in np.where(mask)[0] or right_idx in np.where(mask)[0]:
        boundary_indices = [i for i in [left_idx, right_idx] if mask[i]]
        for idx in boundary_indices:
            ax2.scatter(x[idx], y[idx], c='red', s=120, marker='*', 
                       edgecolor='black', linewidth=1, zorder=5)
    
    # Formatting for zoom plot
    pad = 0.05 * sim.sphere_radius
    ax2.set_xlim(zoom_x.min()-pad, zoom_x.max()+pad)
    ax2.set_ylim(zoom_y.min()-pad, zoom_y.max()+pad)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Gap region (Δφ={gap_info.largest_gap_width:.3f} rad)")
    
    # Shared colorbar
    cbar = fig.colorbar(sc, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label(r"azimuth $\phi$ (rad)")
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    if show:
        plt.tight_layout()
        plt.show()

    if save:
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / (Path(base_path).name + "_gap_analysis.png")
        fig.savefig(out_path.as_posix(), dpi=180, bbox_inches='tight')
        print(f"[info] Saved gap analysis to {out_path}")
        
    return fig, (ax1, ax2), gap_info

# ----------------------------- Wavelength Analysis -----------------------------

@dataclass
class WavelengthResult:
    wavelength_distances: np.ndarray    # all measured 2π accumulation distances
    mean_wavelength_rad: float          # mean wavelength in radians
    std_wavelength_rad: float           # standard deviation
    wavelength_arc: float               # mean wavelength in arc length
    wavelength_filaments: float         # mean wavelength in filament lengths
    coherence_spatial: float            # spatial coherence [0,1]
    n_measurements: int                 # number of 2π accumulation measurements
    has_gap: bool                      # whether analysis excludes gap region
    gap_info: GapInfo

def estimate_wavelength_statistical(base_path: str,
                                   sim: Optional[SimulationData]=None,
                                   num_steps: Optional[int]=None,
                                   period_steps: Optional[int]=None,
                                   filament_length: Optional[float]=None,
                                   gap_factor: float = 1.6,
                                   min_gap_abs: float = 0.15,
                                   exclude_gap: bool = True,
                                   n_starts: int = 20,
                                   show_analysis: bool = True) -> WavelengthResult:
    """
    Statistical wavelength estimation: measure multiple 2π phase accumulation distances.
    
    Uses complex exponential approach to handle phase periodicity correctly.
    
    Args:
        filament_length: Length of a single filament. If None, estimated from segment data.
        period_steps: Number of steps to analyze from the end. If None, uses last frame.
        exclude_gap: If True and gap detected, exclude gap region from analysis.
        n_starts: Number of different starting points for 2π accumulation measurements.
        show_analysis: If True, plot the analysis and histogram.
        
    Returns:
        WavelengthResult with statistical distribution of wavelength measurements.
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    gap_info = detect_gaps(sim.basal_phi, gap_factor, min_gap_abs)
    
    # Get phases in sorted azimuthal order
    phases_sorted = sim.phases[:, sim.order_idx]  # (T, N)
    phi_sorted = sim.basal_phi[sim.order_idx]
    
    # Use last frame or average over period
    if period_steps is None or period_steps <= 0:
        # Use instantaneous pattern from last frame
        phase_pattern = phases_sorted[-1, :]  # shape (N,)
    else:
        # Average over last period_steps
        window = phases_sorted[-period_steps:]
        phase_pattern = np.mean(window, axis=0)
    
    # Exclude gap region if requested and detected
    if exclude_gap and gap_info.has_gap:
        # Find indices to exclude (around the gap)
        left_idx, right_idx = gap_info.largest_gap_indices
        # Convert to sorted indices
        left_sorted = np.where(sim.order_idx == left_idx)[0][0]
        right_sorted = np.where(sim.order_idx == right_idx)[0][0]
        
        # Create mask excluding gap region
        if left_sorted < right_sorted:
            # Normal case: gap doesn't wrap around
            gap_mask = np.ones(len(phi_sorted), dtype=bool)
            gap_mask[left_sorted:right_sorted+1] = False
        else:
            # Gap wraps around 2π
            gap_mask = np.zeros(len(phi_sorted), dtype=bool)
            gap_mask[right_sorted+1:left_sorted] = True
        
        phi_analysis = phi_sorted[gap_mask]
        phase_analysis = phase_pattern[gap_mask]
        
        print(f"[info] Excluding gap region: {np.sum(~gap_mask)} cilia removed")
    else:
        phi_analysis = phi_sorted
        phase_analysis = phase_pattern
    
    N = len(phi_analysis)
    
    # Use complex exponential to handle phase periodicity
    z_pattern = np.exp(1j * phase_analysis)
    
    # Statistical approach: measure 2π accumulation distances from multiple starting points
    wavelength_distances = []
    
    # Choose starting points distributed across the available data
    start_indices = np.linspace(0, N//2, n_starts, dtype=int)  # Only use first half as starts
    
    for start_idx in start_indices:
        if start_idx >= N - 1:
            continue
            
        # Starting position and phase
        phi_start = phi_analysis[start_idx]
        z_start = z_pattern[start_idx]
        
        # Accumulate phase difference using complex exponential method
        cumulative_phase = 0.0
        
        for i in range(start_idx + 1, N):
            phi_current = phi_analysis[i]
            z_current = z_pattern[i]
            
            # Phase difference between consecutive points using complex division
            # Δψ = arg(z_current / z_prev)
            z_prev = z_pattern[i-1]
            phase_diff = np.angle(z_current / z_prev)
            
            cumulative_phase += abs(phase_diff)
            
            # Check if we've accumulated 2π
            if cumulative_phase >= 2*np.pi:
                # Interpolate to get precise crossing point
                excess = cumulative_phase - 2*np.pi
                if abs(phase_diff) > 1e-12:  # avoid division by zero
                    fraction = excess / abs(phase_diff)
                    phi_end = phi_analysis[i-1] + fraction * (phi_current - phi_analysis[i-1])
                else:
                    phi_end = phi_current
                
                # Calculate wavelength distance
                wavelength_dist = phi_end - phi_start
                
                # Handle wraparound case
                if wavelength_dist < 0:
                    wavelength_dist += 2*np.pi
                    
                wavelength_distances.append(wavelength_dist)
                break
    
    # Also try measurements going in the opposite direction (for completeness)
    for start_idx in start_indices:
        if start_idx <= 0:
            continue
            
        phi_start = phi_analysis[start_idx]
        z_start = z_pattern[start_idx]
        
        cumulative_phase = 0.0
        
        for i in range(start_idx - 1, -1, -1):
            phi_current = phi_analysis[i]
            z_current = z_pattern[i]
            
            z_prev = z_pattern[i+1]
            phase_diff = np.angle(z_current / z_prev)
            
            cumulative_phase += abs(phase_diff)
            
            if cumulative_phase >= 2*np.pi:
                excess = cumulative_phase - 2*np.pi
                if abs(phase_diff) > 1e-12:
                    fraction = excess / abs(phase_diff)
                    phi_end = phi_analysis[i+1] + fraction * (phi_current - phi_analysis[i+1])
                else:
                    phi_end = phi_current
                
                wavelength_dist = phi_start - phi_end
                if wavelength_dist < 0:
                    wavelength_dist += 2*np.pi
                    
                wavelength_distances.append(wavelength_dist)
                break
    
    wavelength_distances = np.array(wavelength_distances)
    
    if len(wavelength_distances) == 0:
        print("[warn] No wavelength measurements could be made")
        return WavelengthResult(
            wavelength_distances=np.array([]),
            mean_wavelength_rad=np.inf,
            std_wavelength_rad=0,
            wavelength_arc=np.inf,
            wavelength_filaments=np.inf,
            coherence_spatial=0,
            n_measurements=0,
            has_gap=gap_info.has_gap,
            gap_info=gap_info
        )
    
    # Statistical analysis
    mean_wavelength_rad = np.mean(wavelength_distances)
    std_wavelength_rad = np.std(wavelength_distances)
    
    # Coherence: how concentrated are the measurements?
    # High coherence = low relative standard deviation
    coherence_spatial = 1.0 / (1.0 + std_wavelength_rad / mean_wavelength_rad) if mean_wavelength_rad > 0 else 0
    
    # Convert to other units
    wavelength_arc = sim.sphere_radius * mean_wavelength_rad
    
    # Estimate filament length if not provided
    if filament_length is None:
        seg_dists = []
        for i in range(min(10, sim.phases.shape[1])):
            segs = sim.seg_positions[-1, i, :, :]
            dists = np.linalg.norm(np.diff(segs, axis=0), axis=1)
            seg_dists.extend(dists)
        mean_seg_length = np.mean(seg_dists)
        filament_length = mean_seg_length * (sim.num_segs - 1)
        print(f"[info] Estimated filament length: {filament_length:.2f}")
    
    wavelength_filaments = wavelength_arc / filament_length
    
    if show_analysis:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top: Phase vs azimuth with complex exponential representation
        ax1.plot(phi_analysis, phase_analysis, 'b.-', markersize=4, linewidth=1, label='phase')
        ax1.plot(phi_analysis, np.real(z_pattern), 'r-', alpha=0.7, label='Re(exp(iψ))')
        ax1.plot(phi_analysis, np.imag(z_pattern), 'g-', alpha=0.7, label='Im(exp(iψ))')
        ax1.set_xlabel('azimuth φ (rad)')
        ax1.set_ylabel('phase / complex components')
        title = f'Spatial phase pattern\n⟨λ⟩ = {mean_wavelength_rad:.3f} ± {std_wavelength_rad:.3f} rad = {wavelength_filaments:.2f} filament lengths'
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark mean wavelength
        n_waves = int(2*np.pi / mean_wavelength_rad) + 1
        for i in range(n_waves):
            x_mark = i * mean_wavelength_rad
            if x_mark <= phi_analysis.max():
                ax1.axvline(x_mark, color='purple', linestyle='--', alpha=0.5)
        
        # Bottom: Histogram of wavelength measurements
        bins = max(5, len(wavelength_distances) // 3)
        ax2.hist(wavelength_distances, bins=bins, alpha=0.7, density=True, 
                edgecolor='black', linewidth=0.5)
        ax2.axvline(mean_wavelength_rad, color='red', linestyle='--', 
                   label=f'mean = {mean_wavelength_rad:.3f}')
        ax2.axvline(mean_wavelength_rad - std_wavelength_rad, color='orange', linestyle=':', alpha=0.7)
        ax2.axvline(mean_wavelength_rad + std_wavelength_rad, color='orange', linestyle=':', alpha=0.7,
                   label=f'±1σ = ±{std_wavelength_rad:.3f}')
        ax2.set_xlabel('wavelength (rad)')
        ax2.set_ylabel('probability density')
        ax2.set_title(f'Wavelength distribution (N={len(wavelength_distances)}, coherence={coherence_spatial:.3f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save analysis plot
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / (Path(base_path).name + "_wavelength_statistical.png")
        fig.savefig(out_path.as_posix(), dpi=180)
        print(f"[info] Saved wavelength analysis to {out_path}")
        
        plt.show()
    
    return WavelengthResult(
        wavelength_distances=wavelength_distances,
        mean_wavelength_rad=mean_wavelength_rad,
        std_wavelength_rad=std_wavelength_rad,
        wavelength_arc=wavelength_arc,
        wavelength_filaments=wavelength_filaments,
        coherence_spatial=coherence_spatial,
        n_measurements=len(wavelength_distances),
        has_gap=gap_info.has_gap,
        gap_info=gap_info
    )

def analyze_wavelength_evolution(base_path: str,
                                sim: Optional[SimulationData]=None,
                                num_steps: Optional[int]=None,
                                stride: int = 10,
                                filament_length: Optional[float]=None,
                                exclude_gap: bool = True,
                                show_plot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze how wavelength evolves over time using statistical approach.
    
    Returns:
        times: Time points for analysis
        wavelengths: Mean wavelength in filament lengths at each time
        coherences: Spatial coherence at each time
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    # Estimate filament length if needed
    if filament_length is None:
        seg_dists = []
        for i in range(min(10, sim.phases.shape[1])):
            segs = sim.seg_positions[-1, i, :, :]
            dists = np.linalg.norm(np.diff(segs, axis=0), axis=1)
            seg_dists.extend(dists)
        mean_seg_length = np.mean(seg_dists)
        filament_length = mean_seg_length * (sim.num_segs - 1)
    
    phases_sorted = sim.phases[:, sim.order_idx]
    phi_sorted = sim.basal_phi[sim.order_idx]
    
    # Handle gap exclusion
    gap_info = detect_gaps(sim.basal_phi) if exclude_gap else None
    if exclude_gap and gap_info and gap_info.has_gap:
        left_idx, right_idx = gap_info.largest_gap_indices
        left_sorted = np.where(sim.order_idx == left_idx)[0][0]
        right_sorted = np.where(sim.order_idx == right_idx)[0][0]
        
        if left_sorted < right_sorted:
            gap_mask = np.ones(len(phi_sorted), dtype=bool)
            gap_mask[left_sorted:right_sorted+1] = False
        else:
            gap_mask = np.zeros(len(phi_sorted), dtype=bool)
            gap_mask[right_sorted+1:left_sorted] = True
        
        phi_analysis = phi_sorted[gap_mask]
    else:
        phi_analysis = phi_sorted
        gap_mask = slice(None)
    
    times = []
    wavelengths = []
    coherences = []
    
    # Analyze each timestep
    for i in range(0, len(sim.times), stride):
        phase_instant = phases_sorted[i, gap_mask]
        z_pattern = np.exp(1j * phase_instant)
        
        # Quick statistical measurement (fewer starting points for speed)
        wavelength_distances = []
        n_starts_quick = min(10, len(phi_analysis)//4)
        start_indices = np.linspace(0, len(phi_analysis)//2, n_starts_quick, dtype=int)
        
        for start_idx in start_indices:
            if start_idx >= len(phi_analysis) - 1:
                continue
                
            cumulative_phase = 0.0
            phi_start = phi_analysis[start_idx]
            
            for j in range(start_idx + 1, len(phi_analysis)):
                z_prev = z_pattern[j-1]
                z_current = z_pattern[j]
                phase_diff = np.angle(z_current / z_prev)
                cumulative_phase += abs(phase_diff)
                
                if cumulative_phase >= 2*np.pi:
                    wavelength_dist = phi_analysis[j] - phi_start
                    if wavelength_dist > 0:
                        wavelength_distances.append(wavelength_dist)
                    break
        
        if len(wavelength_distances) > 0:
            wavelength_distances = np.array(wavelength_distances)
            mean_wl = np.mean(wavelength_distances)
            std_wl = np.std(wavelength_distances)
            coherence = 1.0 / (1.0 + std_wl / mean_wl) if mean_wl > 0 else 0
            
            wavelength_fil = (sim.sphere_radius * mean_wl) / filament_length
            
            times.append(sim.times[i])
            wavelengths.append(wavelength_fil)
            coherences.append(coherence)
    
    times = np.array(times)
    wavelengths = np.array(wavelengths)
    coherences = np.array(coherences)
    
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Wavelength evolution
        ax1.plot(times, wavelengths, 'b.-', markersize=4)
        ax1.set_xlabel('time (periods)' if sim.num_steps else 'normalized time')
        ax1.set_ylabel('wavelength (filament lengths)')
        ax1.set_title('Wavelength evolution over time (statistical method)')
        ax1.grid(True, alpha=0.3)
        
        # Add final value annotation
        if len(wavelengths) > 0:
            final_val = wavelengths[-1]
            ax1.axhline(final_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'final: {final_val:.2f}')
            ax1.legend()
        
        # Coherence evolution
        ax2.plot(times, coherences, 'g.-', markersize=4)
        ax2.set_xlabel('time (periods)' if sim.num_steps else 'normalized time')
        ax2.set_ylabel('spatial coherence')
        ax2.set_title('Wave coherence evolution')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return times, wavelengths, coherences

# ----------------------------- Convenience CLI -----------------------------

def quick_all(base_path: str,
              num_steps: Optional[int] = None,
              sphere_radius: Optional[float] = None,
              num_segs: Optional[int] = None,
              make_video: bool = False):
    """
    Convenience one-call workflow:
      - load
      - kymograph
      - last frame iso + first frame top
      - wave direction analysis
      - (optional) video
    """
    sim = load_simulation(base_path, num_steps=num_steps, sphere_radius=sphere_radius, num_segs=num_segs)
    print("[info] Loaded simulation:",
          f"T={sim.phases.shape[0]}, N={sim.phases.shape[1]}, S={sim.num_segs}")
    if sim.num_steps:
        print(f"[info] Time range: {sim.times[0]:.2f} to {sim.times[-1]:.2f} periods")

    _, _, gap = plot_kymograph(base_path, sim=sim, use_phi_axis=True)
    print(f"[info] Gap detection: has_gap={gap.has_gap}, largest Δφ={gap.largest_gap_width:.3f} rad, ratio={gap.ratio:.2f}")

    plot_frame(base_path, sim=sim, frame="first", view="top")
    plot_frame(base_path, sim=sim, frame="last", view="iso")

    wd = analyze_wave_direction(base_path, sim=sim)
    print("[info] Wave directionality:")
    print(f"  + {wd.percent_positive:.2f}%  - {wd.percent_negative:.2f}%  0 {wd.percent_stationary:.2f}%")

    if make_video:
        make_topdown_video(base_path, sim=sim)

    return sim, wd

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plot & analyze a cuda-filaments simulation.")
    ap.add_argument("base_path", help="Common prefix path without file suffixes.")
    ap.add_argument("--num_steps", type=int, default=None, help="Steps per period for time normalization.")
    ap.add_argument("--video", action="store_true", help="Generate top-down MP4.")
    ap.add_argument("--period_steps", type=int, default=None, help="Steps for last-period analysis.")
    args = ap.parse_args()

    sim = load_simulation(args.base_path, num_steps=args.num_steps)
    plot_kymograph(args.base_path, sim=sim)
    plot_frame(args.base_path, sim=sim, frame="first", view="top")
    plot_frame(args.base_path, sim=sim, frame="last", view="iso")
    wd = analyze_wave_direction(args.base_path, sim=sim, period_steps=args.period_steps)
    print(f"Directionality: +{wd.percent_positive:.2f}%  -{wd.percent_negative:.2f}%  0{wd.percent_stationary:.2f}%")
    if args.video:
        make_topdown_video(args.base_path, sim=sim)