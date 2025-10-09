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
    times: np.ndarray           # shape (T,) - in units of periods
    seg_positions: np.ndarray   # shape (T, N, S, 3)
    basal_pos: np.ndarray       # shape (N, 3)
    basal_phi: np.ndarray       # shape (N,)
    order_idx: np.ndarray       # sorting indices (ascending φ)
    num_segs: int
    num_fils: int
    num_steps: Optional[int] = None  # steps per period
    sphere_radius: Optional[float] = None

# ----------------------------- Loading Helpers -----------------------------

def _infer_num_fils_from_phase(phase_file: str) -> int:
    with open(phase_file, "r") as f:
        first = f.readline().strip().split()
    width = len(first)
    # Columns: time, ?, then N ps1 and N psi2
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
    T = len(times_raw)
    phases = np.mod(phase_data[:, 2:num_fils+2], 2*np.pi)  # only psi 1

    # Time normalization: convert simulation steps to periods/cycles
    if num_steps is not None and num_steps > 0:
        times = times_raw / num_steps  # time in units of periods
        print(f"[info] Using num_steps={num_steps} for time normalization.")
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
        num_segs = int(flat_len // (num_fils * 3))
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
        num_fils=num_fils,
        sphere_radius=sphere_radius
    )

# ----------------------------- Kymograph -----------------------------
def plot_kymograph(base_path: str,
                   sim: Optional[SimulationData]=None,
                   num_steps: Optional[int]=None,
                   use_phi_axis: bool = True,
                   cmap=DEFAULT_CMAP,
                   show: bool = True,
                   save: bool = True,
                   fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot kymograph of phases vs time. If use_phi_axis, y-axis = true azimuths φ
    so gaps appear as blank regions (no interpolation).
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    phases_sorted = sim.phases[:, sim.order_idx]   # shape (T, N_sorted)
    phi_sorted = sim.basal_phi[sim.order_idx]
    T, N = phases_sorted.shape

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = fig_ax

    if use_phi_axis:
        # Use contourf like in the notebook - this works correctly
        X, Y = np.meshgrid(sim.times, phi_sorted)
        im = ax.contourf(X, Y, phases_sorted.T, levels=100, cmap=cmap, 
                         vmin=0, vmax=2*np.pi)
        ax.set_ylabel("azimuth φ (rad)")
        ax.set_yticks([0, np.pi, 2*np.pi])
        ax.set_yticklabels(['0', 'π', '2π'])
    else:
        # For index-based y-axis, use imshow
        im = ax.imshow(phases_sorted.T, aspect='auto', cmap=cmap,
                       extent=[sim.times[0], sim.times[-1], 0, N],
                       origin='lower', vmin=0, vmax=2*np.pi)
        ax.set_ylabel("cilia index (sorted)")

    # Update x-axis label based on whether we have num_steps
    if sim.num_steps is not None:
        ax.set_xlabel("time (periods)")
    else:
        ax.set_xlabel("normalized time")
        
    title = "Kymograph"
    ax.set_title(title)

    # Only add colorbar if we're not using an existing figure
    if fig_ax is None:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"phase $\psi$ (rad)")
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
    return fig, ax
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
        f = -1
    else:
        f = int(frame)

    if fig_ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = fig_ax

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = sim.sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sim.sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sim.sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='grey', alpha=0.1)

    # Axes limits
    x_coords = sim.seg_positions[0, :, :, 0]
    y_coords = sim.seg_positions[0, :, :, 1]
    z_coords = sim.seg_positions[0, :, :, 2]

    margin = 2.0
    ax.set_xlim(np.min(x_coords) - margin, np.max(x_coords) + margin)
    ax.set_ylim(np.min(y_coords) - margin, np.max(y_coords) + margin)
    ax.set_zlim(np.min(z_coords) - margin, np.max(z_coords) + margin)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title(r'')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_zticks([])

    if view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=30, azim=-45)

    # Plot cilia with phase-based coloring
    if color_by_phase:
        norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)
        for i in range(sim.num_fils):
            cilium_positions = sim.seg_positions[f, i, :, :]
            x_data = cilium_positions[:, 0]
            y_data = cilium_positions[:, 1]
            z_data = cilium_positions[:, 2]
            color = cmap(norm(sim.phases[f, i]))
            ax.plot(x_data, y_data, z_data, '-', lw=2, color=color)
    else:
        for i in range(sim.num_fils):
            cilium_positions = sim.seg_positions[f, i, :, :]
            x_data = cilium_positions[:, 0]
            y_data = cilium_positions[:, 1]
            z_data = cilium_positions[:, 2]
            ax.plot(x_data, y_data, z_data, '-', lw=2)

    if color_by_phase:
        # Place a polar inset in the top-right of the 3D axes
        fig.canvas.draw()  # ensure bbox is valid before querying
        bbox = ax.get_position()  # figure coords [0..1]
        side = 0.18                       # inset size (fraction of figure)
        pad  = 0.02                       # padding from top-right

        # Clamp so left/bottom are finite and in [0, 1-side]
        left   = np.clip(bbox.x1 - side - pad, 0.0, 1.0 - side)
        bottom = np.clip(bbox.y1 - side - pad, 0.0, 1.0 - side)
        circ = fig.add_axes([left, bottom, side, side], projection='polar')

        # Build a colored ring
        theta = np.linspace(0, 2*np.pi, 361)
        r = np.linspace(0.82, 1.00, 2)    # thin ring
        Theta, R = np.meshgrid(theta, r)
        Z = Theta                          # values -> colormap in [0, 2π]
        im_circ = circ.pcolormesh(Theta, R, Z, cmap=cmap, shading='auto', vmin=0, vmax=2*np.pi)

        # Style the circular legend
        circ.set_yticks([]); circ.set_xticks([])
        circ.spines['polar'].set_visible(False)
        circ.set_theta_zero_location('N')
        circ.set_theta_direction(-1)
        circ.set_rlim(0.78, 1.02)

        # Center label in axes coords (avoid polar-data transform issues)
        circ.text(0.5, 0.5, r'$\psi_1$', ha='center', va='center', fontsize=14, transform=circ.transAxes)

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

    # CHANGE: Match cell 6 figure size
    fig = plt.figure(figsize=(12, 12))  # was (8,8)
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
    ax.set_title("")  # Remove title to match cell 6

    # CHANGE: Match cell 6 sphere rendering
    u = np.linspace(0, 2*np.pi, 100)  # was 80
    v = np.linspace(0, np.pi, 100)    # was 40
    xs = sim.sphere_radius * np.outer(np.cos(u), np.sin(v))
    ys = sim.sphere_radius * np.outer(np.sin(u), np.sin(v))
    zs = sim.sphere_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='grey', alpha=0.15, linewidth=0, antialiased=True)

    # ADD: Circular legend matching cell 6
    fig.canvas.draw()
    bbox = ax.get_position()
    side = 0.18
    pad = 0.02
    left = np.clip(bbox.x1 - side - pad, 0.0, 1.0 - side)
    bottom = np.clip(bbox.y1 - side - pad, 0.0, 1.0 - side)
    circ = fig.add_axes([left, bottom, side, side], projection='polar')

    # Build colored ring
    theta = np.linspace(0, 2*np.pi, 361)
    r_inner, r_outer = 0.82, 1.50
    Theta, R = np.meshgrid(theta, np.linspace(r_inner, r_outer, 2))
    Z = Theta
    im_circ = circ.pcolormesh(Theta, R, Z, cmap=cmap, shading='auto', vmin=0, vmax=2*np.pi)

    # Style circular legend
    circ.set_yticks([]); circ.set_xticks([])
    circ.spines['polar'].set_visible(False)
    circ.set_theta_zero_location('N')
    circ.set_theta_direction(1)
    circ.set_rlim(0.78, 1.02)

    # Add radial guides and labels
    theta0 = 0.0
    theta_eff = 0.6*np.pi
    circ.plot([theta0, theta0], [r_inner, r_outer], color='k', lw=1.0, zorder=10)
    circ.plot([theta_eff, theta_eff], [r_inner, r_outer], color='k', lw=1.0, zorder=10)
    
    circ.text(theta0, 1.03, r'$0$', ha='center', va='bottom', fontsize=14,
              transform=circ.transData, clip_on=False, zorder=11)
    circ.text(theta_eff, 1.09, r'$2\pi f_{\mathrm{eff}}$', ha='center', va='bottom', fontsize=14,
              transform=circ.transData, clip_on=False, zorder=11)

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
    labels: np.ndarray        # shape (N) values in {-1,0,1}
    velocity: np.ndarray      # shape (N) average propagation indicator

def analyze_wave_direction(base_path: str,
                           sim: Optional[SimulationData]=None,
                           num_steps: Optional[int]=None,
                           period_steps: Optional[int]=100,
                           tol: float = 1e-6) -> WaveDirectionResult:
    """
    Estimate propagation direction sign from ψ(φ, t):
       dy/dt sign ~ - (∂ψ/∂t)/(∂ψ/∂φ).
    Uses final 'period_steps' samples; if None uses all.
    
    Uses complex exponential method to handle phase periodicity correctly.
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)

    phases_sorted = sim.phases[:, sim.order_idx]  # (T, N)
    phi_sorted = sim.basal_phi[sim.order_idx]

    if period_steps is None or period_steps <= 0 or period_steps > phases_sorted.shape[0]:
        window = phases_sorted
    else:
        window = phases_sorted[-period_steps:]

    # Use complex exponential method to handle periodicity
    # Let f(y,t) = exp(i * psi(y,t)). Then
    # df/dt = -psi(y, t) * f(y,t) * d(psi)/dt and
    # df/dy = -psi(y, t) * f(y,t) * d(psi)/dy.
    # Then, d(psi)/dt = -df/dt / (psi(y, t) * f(y,t)) and
    # d(psi)/dy = -df/dy / (psi(y, t) * f(y,t)).
    
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
        velocity = vel_real
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
                        fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot basal positions of cilia (top-down view).
    
    Args:
        color_by: "azimuth" (angle-based), "index" (sequential), or "uniform" (all same color)
        show_gap: If True and gap detected, highlight the largest gap
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
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
    
    ax.set_title(title)

    # Colorbar for non-uniform coloring
    if color_by != "uniform" and cbar_label:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label(cbar_label)
        if color_by == "azimuth":
            cbar.set_ticks([0, np.pi, 2*np.pi])
            cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

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
        
    return fig, ax

# ----------------------------- Blob Position Plot -----------------------------

def plot_blob_positions(base_path: str,
                       sim: Optional[SimulationData]=None,
                       num_steps: Optional[int]=None,
                       view: str = "top",  # "top", "iso", or "sphere"
                       color_by: str = "azimuth",  # "azimuth", "altitude", "index", or "uniform"
                       show_sphere: bool = True,
                       cmap=DEFAULT_CMAP,
                       show: bool = True,
                       save: bool = True,
                       fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot surface blob positions.
    
    Args:
        view: "top" (2D top-down), "iso" (3D isometric), or "sphere" (3D with sphere surface)
        color_by: "azimuth" (φ angle), "altitude" (θ angle), "index", or "uniform"
        show_sphere: If True and 3D view, show sphere surface
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    # Load blob references
    blob_ref_file = f"{base_path}_blob_references.dat"
    if not os.path.isfile(blob_ref_file):
        print(f"[warn] Blob reference file not found: {blob_ref_file}")
        return None, None
    
    blob_pos = np.loadtxt(blob_ref_file).reshape(-1, 3)
    x, y, z = blob_pos[:, 0], blob_pos[:, 1], blob_pos[:, 2]
    
    # Calculate spherical coordinates for coloring
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.mod(np.arctan2(y, x), 2*np.pi)  # azimuth [0, 2π]
    theta = np.arccos(np.clip(z / (r + 1e-14), -1, 1))  # polar angle [0, π]
    
    # Determine coloring
    if color_by == "azimuth":
        colors = phi
        cbar_label = r"azimuth $\phi$ (rad)"
        vmin, vmax = 0, 2*np.pi
    elif color_by == "altitude":
        colors = theta
        cbar_label = r"polar angle $\theta$ (rad)"
        vmin, vmax = 0, np.pi
    elif color_by == "index":
        colors = np.arange(len(x))
        cbar_label = "blob index"
        vmin, vmax = 0, len(x)-1
    else:  # uniform
        colors = 'grey'
        cbar_label = None
        vmin = vmax = None
    
    # Create figure
    if fig_ax is None:
        if view == "top":
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax
    
    # Plot based on view
    if view == "top":
        # 2D top-down view
        if color_by == "uniform":
            sc = ax.scatter(x, y, c=colors, s=10, alpha=0.6, edgecolor='k', linewidth=0.2)
        else:
            sc = ax.scatter(x, y, c=colors, cmap=cmap, s=10, alpha=0.6, 
                          edgecolor='k', linewidth=0.2, vmin=vmin, vmax=vmax)
        
        # Sphere outline
        theta_circle = np.linspace(0, 2*np.pi, 400)
        circle_x = sim.sphere_radius * np.cos(theta_circle)
        circle_y = sim.sphere_radius * np.sin(theta_circle)
        ax.plot(circle_x, circle_y, color='grey', lw=2, alpha=0.6)
        
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
    else:
        # 3D view
        if color_by == "uniform":
            sc = ax.scatter(x, y, z, c=colors, s=10, alpha=0.6, edgecolor='k', linewidth=0.2)
        else:
            sc = ax.scatter(x, y, z, c=colors, cmap=cmap, s=10, alpha=0.6,
                          edgecolor='k', linewidth=0.2, vmin=vmin, vmax=vmax)
        
        # Sphere surface
        if show_sphere:
            u = np.linspace(0, 2*np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            xs = sim.sphere_radius * np.outer(np.cos(u), np.sin(v))
            ys = sim.sphere_radius * np.outer(np.sin(u), np.sin(v))
            zs = sim.sphere_radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, color='grey', alpha=0.1, linewidth=0)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        
        if view == "top":
            ax.view_init(elev=90, azim=-90)
        else:  # iso
            ax.view_init(elev=30, azim=-45)
        
        # Set limits
        margin = sim.sphere_radius * 0.1
        max_range = sim.sphere_radius + margin
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
    
    title = f"Surface blobs (N={len(x)})"
    ax.set_title(title)
    
    # Colorbar for non-uniform coloring
    if color_by != "uniform" and cbar_label:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label(cbar_label)
        if color_by == "azimuth":
            cbar.set_ticks([0, np.pi, 2*np.pi])
            cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
        elif color_by == "altitude":
            cbar.set_ticks([0, np.pi/2, np.pi])
            cbar.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    
    if show:
        plt.tight_layout()
        plt.show()
    
    if save:
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True, parents=True)
        suffix = f"_blobs_{view}_{color_by}.png"
        out_path = out_dir / (Path(base_path).name + suffix)
        fig.savefig(out_path.as_posix(), dpi=180, bbox_inches='tight')
        print(f"[info] Saved blob positions to {out_path}")
    
    return fig, ax

# ----------------------------- Wavelength Analysis -----------------------------

@dataclass
class WavelengthResult:
    wavelength_distances: np.ndarray    # all measured 2π accumulation distances
    mean_wavelength_rad: float          # mean wavelength in radians
    std_wavelength_rad: float           # standard deviation
    wavelength_arc: float               # mean wavelength in arc length
    wavelength_filaments: float         # mean wavelength in filament lengths
    n_measurements: int                 # number of 2π accumulation measurements

def estimate_wavelength_statistical(base_path: str,
                                   sim: Optional[SimulationData]=None,
                                   num_steps: Optional[int]=None,
                                   filament_length: Optional[float]=None,
                                   show_analysis: bool = True) -> WavelengthResult:
    """
    Statistical wavelength estimation: measure 2π phase accumulation distances.
    
    Simple approach: start at index 0, walk around accumulating phase differences,
    record wavelength each time we cross 2π.
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    # Get phases in sorted azimuthal order (last frame only)
    phases_sorted = sim.phases[:, sim.order_idx]  # (T, N)
    phi_sorted = sim.basal_phi[sim.order_idx]
    phase_pattern = phases_sorted[-1, :]  # shape (N,) - last frame only
    
    N = len(phase_pattern)
    
    # Compute phase differences between adjacent cilia (handling 2π wraparound)
    phase_diffs = []
    for i in range(N):
        j = (i + 1) % N  # wrap around to handle the last->first connection
        diff = phase_pattern[j] - phase_pattern[i]
        # Wrap difference to [-π, π]
        while diff > np.pi:
            diff -= 2*np.pi
        while diff < -np.pi:
            diff += 2*np.pi
        phase_diffs.append(abs(diff))  # we want absolute differences
    
    phase_diffs = np.array(phase_diffs)
    
    # Simple approach: start at 0, walk around once, record each 2π crossing
    wavelength_distances = []
    cumulative_phase = 0.0
    phi_start = phi_sorted[0]
    last_wavelength_start = 0
    
    # Walk around the entire circle once (plus a bit to handle wraparound)
    for step in range(1, 2*N):  # Go one past N to handle the wrap-around case
        idx = step % N
        cumulative_phase += phase_diffs[step - 1]
        
        # Check if we've accumulated 2π (found one wavelength)
        if cumulative_phase >= 2*np.pi:
            # Calculate the azimuthal distance for this wavelength
            phi_end = phi_sorted[idx]
            phi_wavelength_start = phi_sorted[last_wavelength_start]
            
            wavelength_dist = phi_end - phi_wavelength_start
            
            # Handle wraparound case
            if wavelength_dist < 0:
                wavelength_dist += 2*np.pi
            
            wavelength_distances.append(wavelength_dist)
            
            # Reset for next wavelength measurement
            cumulative_phase = 0.0
            last_wavelength_start = idx
            if step >= N:
                break  # Stop if we've wrapped around once
    
    # Convert to array
    wavelength_distances = np.array(wavelength_distances)
    
    # Handle case where no wavelengths found
    if len(wavelength_distances) == 0:
        print("[warn] No wavelength measurements could be made")
        return WavelengthResult(
            wavelength_distances=np.array([]),
            mean_wavelength_rad=np.inf,
            std_wavelength_rad=0.0,
            wavelength_arc=np.inf,
            wavelength_filaments=np.inf,
            n_measurements=0
        )
    
    # Statistical analysis
    mean_wavelength_rad = np.mean(wavelength_distances)
    std_wavelength_rad = np.std(wavelength_distances)
    
    # # Coherence: higher when measurements are consistent
    # coherence_spatial = 1.0 / (1.0 + std_wavelength_rad / mean_wavelength_rad) if mean_wavelength_rad > 0 else 0
    
    # Convert to other units
    wavelength_arc = sim.sphere_radius * mean_wavelength_rad
    
    # Estimate filament length if needed
    if filament_length is None:
        seg_dists = []
        for i in range(min(10, sim.phases.shape[1])):
            segs = sim.seg_positions[-1, i, :, :]
            dists = np.linalg.norm(np.diff(segs, axis=0), axis=1)
            seg_dists.extend(dists)
        mean_seg_length = np.mean(seg_dists)
        filament_length = mean_seg_length * (sim.num_segs - 1)
    
    wavelength_filaments = wavelength_arc / filament_length
    
    # Visualization (same as before...)
    if show_analysis:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Phase pattern
        ax1.plot(phi_sorted, phase_pattern, 'b.-', markersize=4, label='phase pattern')
        ax1.set_xlabel('azimuth φ (rad)')
        ax1.set_ylabel('phase ψ (rad)')
        ax1.set_title(f'Phase pattern (last frame)\nMean wavelength: {mean_wavelength_rad:.3f} rad = {wavelength_filaments:.2f} fil lengths')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 2*np.pi)
        ax1.set_yticks([0, np.pi, 2*np.pi])
        ax1.set_yticklabels(['0', 'π', '2π'])
        
        # Mark measured wavelengths
        if len(wavelength_distances) > 0:
            for i, wl in enumerate(wavelength_distances):
                x_mark = i * mean_wavelength_rad
                if x_mark <= phi_sorted.max():
                    ax1.axvline(x_mark, color='red', linestyle='--', alpha=0.5, 
                              label='measured λ' if i == 0 else '')
        ax1.legend()
        
        # Bottom plot: Histogram of wavelength measurements
        if len(wavelength_distances) > 1:
            bins = max(3, len(wavelength_distances) // 2)
            ax2.hist(wavelength_distances, bins=bins, alpha=0.7, density=True, 
                    edgecolor='black', linewidth=0.5, color='skyblue')
            ax2.axvline(mean_wavelength_rad, color='red', linestyle='--', linewidth=2,
                       label=f'mean = {mean_wavelength_rad:.3f} rad')
            
            if std_wavelength_rad > 0:
                ax2.axvline(mean_wavelength_rad - std_wavelength_rad, color='orange', 
                           linestyle=':', alpha=0.7)
                ax2.axvline(mean_wavelength_rad + std_wavelength_rad, color='orange', 
                           linestyle=':', alpha=0.7, label=f'±1σ = ±{std_wavelength_rad:.3f} rad')
        else:
            # Just show single value
            ax2.axvline(wavelength_distances[0], color='red', linewidth=3,
                       label=f'single measurement = {wavelength_distances[0]:.3f} rad')
        
        ax2.set_xlabel('wavelength (rad)')
        ax2.set_ylabel('probability density')
        ax2.set_title(f'Wavelength measurements (n={len(wavelength_distances)})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return WavelengthResult(
        wavelength_distances=wavelength_distances,
        mean_wavelength_rad=mean_wavelength_rad,
        std_wavelength_rad=std_wavelength_rad,
        wavelength_arc=wavelength_arc,
        wavelength_filaments=wavelength_filaments,
        n_measurements=len(wavelength_distances)
    )