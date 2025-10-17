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
    DEFAULT_CMAP = cmc.romaO  # For periodic phase data
    SEQUENTIAL_CMAP = cmc.batlow  # For sequential data
    DIVERGING_CMAP = cmc.vik  # For diverging data
    DISCRETE_CMAP = cmc.batlowK  # For discrete categories
except Exception:  # fallback if cmcrameri not installed
    DEFAULT_CMAP = plt.cm.twilight
    SEQUENTIAL_CMAP = plt.cm.viridis
    DIVERGING_CMAP = plt.cm.RdBu_r
    DISCRETE_CMAP = plt.cm.tab10

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
    
    Note: With ablation creating a single contiguous gap at the end of the azimuthal
    ring, the gap will naturally appear as a blank region in the φ-axis plot.
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
        # Use contourf - gaps will automatically appear as blank regions
        # since we're plotting at the true φ positions
        X, Y = np.meshgrid(sim.times, phi_sorted)
        im = ax.contourf(X, Y, phases_sorted.T, levels=101, cmap=cmap, 
                         vmin=0, vmax=2*np.pi, antialiased=False)
        ax.set_ylabel("azimuth φ (rad)")
        ax.set_yticks([0, np.pi, 2*np.pi])
        ax.set_yticklabels(['0', 'π', '2π'])
        ax.set_ylim(0, 2*np.pi)  # Show full circle to visualize the gap
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
# ----------------------------- Helper Functions -----------------------------

def plot_sphere_surface(ax, radius, alpha=0.1, color='grey', resolution=100):
    """
    Add a sphere surface to a 3D axes.
    
    Args:
        ax: The 3D axes
        radius: Sphere radius
        alpha: Transparency (default 0.1)
        color: Sphere color (default 'grey')
        resolution: Number of points for sphere mesh (default 100)
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)

def setup_3d_axes(ax, seg_positions, sphere_radius, margin=2.0, view='top'):
    """
    Configure 3D axes for cilia visualization.
    
    Args:
        ax: The 3D axes
        seg_positions: Segment position data for determining bounds
        sphere_radius: Sphere radius
        margin: Extra space around data (default 2.0)
        view: 'top' or 'iso' (default 'top')
    """
    x_coords = seg_positions[:, :, 0]
    y_coords = seg_positions[:, :, 1]
    z_coords = seg_positions[:, :, 2]

    ax.set_xlim(np.min(x_coords) - margin, np.max(x_coords) + margin)
    ax.set_ylim(np.min(y_coords) - margin, np.max(y_coords) + margin)
    ax.set_zlim(np.min(z_coords) - margin, np.max(z_coords) + margin)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title('')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_zticks([])

    if view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=30, azim=-45)

def add_phase_legend(fig, ax, cmap=DEFAULT_CMAP, f_eff=0.3):
    """
    Add a circular phase legend to a 3D plot.
    
    Args:
        fig: The matplotlib figure
        ax: The 3D axes to add the legend to
        cmap: The colormap to use
        f_eff: The effective stroke fraction (default 0.3)
    
    Returns:
        The polar axes object for the legend
    """
    fig.canvas.draw()
    bbox = ax.get_position()
    side = 0.18

    left = np.clip(bbox.x1 - side + 5.0, 0.0, 1.0 - side)
    bottom = np.clip(bbox.y1 - side, 0.0, 1.0 - side)
    circ = fig.add_axes([left, bottom, side, side], projection='polar')

    # Build a colored ring
    theta = np.linspace(0, 2*np.pi, 361)
    r = np.linspace(0.82, 1.00, 2)
    Theta, R = np.meshgrid(theta, r)
    Z = Theta
    circ.pcolormesh(Theta, R, Z, cmap=cmap, shading='auto', vmin=0, vmax=2*np.pi)

    # Style the circular legend
    circ.set_yticks([])
    circ.set_xticks([])
    circ.spines['polar'].set_visible(False)
    circ.set_theta_zero_location('N')
    circ.set_theta_direction(1)
    circ.set_rlim(0.8, 1.00)

    # Add radial guide lines and labels at ψ = 0 and ψ = 2π f_eff
    theta0 = 0.0
    theta_eff = 2.0 * np.pi * f_eff
    
    circ.plot([theta0, theta0], [0.85, 1.02], color='k', lw=1.0, zorder=10)
    circ.plot([theta_eff, theta_eff], [0.85, 1.02], color='k', lw=1.0, zorder=10)
    
    circ.text(theta0, 1.0, r'$0$', ha='center', va='bottom', fontsize=12,
              transform=circ.transData, clip_on=False, zorder=11)
    circ.text(theta_eff, 1.075, r'$2\pi f_{\mathrm{eff}}$', ha='center', va='bottom', fontsize=12,
              transform=circ.transData, clip_on=False, zorder=11)

    # Center label
    circ.text(0.5, 0.5, r'$\psi_1$', ha='center', va='center', fontsize=14, transform=circ.transAxes)
    
    return circ

def plot_cilia_at_frame(ax, seg_positions, phases, frame_idx, cmap, color_by_phase=True):
    """
    Plot cilia at a specific frame.
    
    Args:
        ax: The 3D axes
        seg_positions: Segment positions array (T, N, S, 3)
        phases: Phase array (T, N)
        frame_idx: Frame index to plot
        cmap: Colormap
        color_by_phase: Whether to color by phase (default True)
    """
    num_fils = seg_positions.shape[1]
    
    if color_by_phase:
        norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)
        for i in range(num_fils):
            cilium_positions = seg_positions[frame_idx, i, :, :]
            color = cmap(norm(phases[frame_idx, i]))
            ax.plot(cilium_positions[:, 0], cilium_positions[:, 1], 
                   cilium_positions[:, 2], '-', lw=2, color=color)
    else:
        for i in range(num_fils):
            cilium_positions = seg_positions[frame_idx, i, :, :]
            ax.plot(cilium_positions[:, 0], cilium_positions[:, 1], 
                   cilium_positions[:, 2], '-', lw=2)

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

    # Plot sphere and setup axes
    plot_sphere_surface(ax, sim.sphere_radius)
    setup_3d_axes(ax, sim.seg_positions[0], sim.sphere_radius, view=view)

    # Plot cilia
    plot_cilia_at_frame(ax, sim.seg_positions, sim.phases, f, cmap, color_by_phase)

    # Add phase legend
    if color_by_phase:
        add_phase_legend(fig, ax, cmap=cmap, f_eff=0.3)

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

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Setup axes and sphere
    x_all = sim.seg_positions[...,0]
    y_all = sim.seg_positions[...,1]
    z_all = sim.seg_positions[...,2]
    margin = 2.0
    ax.set_xlim(np.min(x_all)-margin, np.max(x_all)+margin)
    ax.set_ylim(np.min(y_all)-margin, np.max(y_all)+margin)
    ax.set_zlim(np.min(z_all)-margin, np.max(z_all)+margin)
    ax.view_init(elev=90, azim=-90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title("")

    plot_sphere_surface(ax, sim.sphere_radius, alpha=0.15)
    add_phase_legend(fig, ax, cmap=cmap, f_eff=0.3)

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
                        cmap=None,
                        show: bool = True,
                        save: bool = True,
                        fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot basal positions of cilia (top-down view).
    
    Args:
        color_by: "azimuth" (angle-based), "index" (sequential), or "uniform" (all same color)
    """
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    x, y = sim.basal_pos[:, 0], sim.basal_pos[:, 1]
    
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = fig_ax

    # Color mapping with appropriate colormaps
    if color_by == "azimuth":
        colors = sim.basal_phi
        cbar_label = r"azimuth $\phi$ (rad)"
        vmin, vmax = 0, 2*np.pi
        if cmap is None:
            cmap = DEFAULT_CMAP  # romaO for periodic azimuth
    elif color_by == "index":
        colors = np.arange(len(x))
        cbar_label = "cilia index"
        vmin, vmax = 0, len(x)-1
        if cmap is None:
            cmap = SEQUENTIAL_CMAP  # batlow for sequential index
    else:  # uniform
        # Use batlow color for uniform (single color from sequential palette)
        try:
            colors = SEQUENTIAL_CMAP(0.5)  # Mid-range color from batlow
        except:
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
                       view: str = "top",
                       color_by: str = "azimuth",
                       show_sphere: bool = True,
                       split_hemispheres: bool = True,
                       cmap=None,
                       show: bool = True,
                       save: bool = True,
                       fig_ax: Optional[Tuple[Any,Any]] = None):
    """
    Plot surface blob positions.
    
    Args:
        view: "top" (2D top-down), "iso" (3D isometric), or "sphere" (3D with sphere surface)
        color_by: "azimuth" (φ angle), "altitude" (θ angle), "index", or "uniform"
        show_sphere: If True and 3D view, show sphere surface
        split_hemispheres: If True, plot top and bottom hemispheres separately (splits by z-coordinate)
        cmap: Colormap to use (if None, appropriate cmap chosen based on color_by)
        show: Whether to display the plot
        save: Whether to save the plot
        fig_ax: Optional existing figure and axes to use
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
    
    # Determine coloring with appropriate colormaps
    if color_by == "azimuth":
        colors = phi
        cbar_label = r"azimuth $\phi$ (rad)"
        vmin, vmax = 0, 2*np.pi
        if cmap is None:
            cmap = DEFAULT_CMAP  # romaO for periodic azimuth
    elif color_by == "altitude":
        colors = theta
        cbar_label = r"polar angle $\theta$ (rad)"
        vmin, vmax = 0, np.pi
        if cmap is None:
            cmap = SEQUENTIAL_CMAP  # batlow for sequential altitude
    elif color_by == "index":
        colors = np.arange(len(x))
        cbar_label = "blob index"
        vmin, vmax = 0, len(x)-1
        if cmap is None:
            cmap = SEQUENTIAL_CMAP  # batlow for sequential index
    else:  # uniform
        # Use batlow color for uniform
        try:
            colors = SEQUENTIAL_CMAP(0.3)  # Color from batlow
        except:
            colors = 'grey'
        cbar_label = None
        vmin = vmax = None
    
    # Create figure
    if fig_ax is None:
        if split_hemispheres:
            # Create side-by-side subplots for hemispheres
            if view == "top":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                                subplot_kw={'projection': '3d'})
            axes = [ax1, ax2]
        else:
            if view == "top":
                fig, ax = plt.subplots(figsize=(8, 8))
            else:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax
        split_hemispheres = False  # Don't split if using provided axes
    
    # Plot based on view
    if not split_hemispheres:
        # Original single-view behavior
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
            ax.set_title(f"Surface blobs (N={len(x)})")
            
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
            
            if view == "iso":
                ax.view_init(elev=30, azim=-45)
            
            # Set limits
            margin = sim.sphere_radius * 0.1
            max_range = sim.sphere_radius + margin
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            
            ax.set_title(f"Surface blobs (N={len(x)})")
        
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
                
    else:
        # Split hemispheres by z-coordinate: top (z >= 0) and bottom (z < 0)
        top_mask = z >= 0
        bottom_mask = z < 0
        
        for idx, (mask, hemisphere_ax, title) in enumerate([
            (top_mask, axes[0], "Top hemisphere (z ≥ 0)"),
            (bottom_mask, axes[1], "Bottom hemisphere (z < 0)")
        ]):
            x_hem = x[mask]
            y_hem = y[mask]
            z_hem = z[mask]
            colors_hem = colors[mask] if color_by != "uniform" else colors
            
            if view == "top":
                # 2D top-down view
                if color_by == "uniform":
                    sc = hemisphere_ax.scatter(x_hem, y_hem, c=colors, s=10, 
                                              alpha=0.6, edgecolor='k', linewidth=0.2)
                else:
                    sc = hemisphere_ax.scatter(x_hem, y_hem, c=colors_hem, 
                                              cmap=cmap, s=10, alpha=0.6,
                                              edgecolor='k', linewidth=0.2, 
                                              vmin=vmin, vmax=vmax)
                
                # Sphere outline
                theta_circle = np.linspace(0, 2*np.pi, 400)
                circle_x = sim.sphere_radius * np.cos(theta_circle)
                circle_y = sim.sphere_radius * np.sin(theta_circle)
                hemisphere_ax.plot(circle_x, circle_y, color='grey', lw=2, alpha=0.6)
                
                hemisphere_ax.set_aspect('equal')
                hemisphere_ax.set_xlabel('x')
                hemisphere_ax.set_ylabel('y')
                hemisphere_ax.grid(True, alpha=0.3)
                
            else:
                # 3D view
                if color_by == "uniform":
                    sc = hemisphere_ax.scatter(x_hem, y_hem, z_hem, c=colors, s=10, 
                                              alpha=0.6, edgecolor='k', linewidth=0.2)
                else:
                    sc = hemisphere_ax.scatter(x_hem, y_hem, z_hem, c=colors_hem, 
                                              cmap=cmap, s=10, alpha=0.6,
                                              edgecolor='k', linewidth=0.2, 
                                              vmin=vmin, vmax=vmax)
                
                # Sphere surface (half sphere)
                if show_sphere:
                    u = np.linspace(0, 2*np.pi, 50)
                    v = np.linspace(0, np.pi, 50)
                    xs = sim.sphere_radius * np.outer(np.cos(u), np.sin(v))
                    ys = sim.sphere_radius * np.outer(np.sin(u), np.sin(v))
                    zs = sim.sphere_radius * np.outer(np.ones_like(u), np.cos(v))
                    
                    # Only show the relevant hemisphere by masking z
                    if idx == 0:  # top
                        zs[zs < 0] = np.nan
                    else:  # bottom
                        zs[zs >= 0] = np.nan
                        
                    hemisphere_ax.plot_surface(xs, ys, zs, color='grey', 
                                              alpha=0.1, linewidth=0)
                
                hemisphere_ax.set_xlabel('x')
                hemisphere_ax.set_ylabel('y')
                hemisphere_ax.set_zlabel('z')
                hemisphere_ax.set_aspect('equal')
                
                # Adjust viewing angle for each hemisphere
                if idx == 0:  # top
                    hemisphere_ax.view_init(elev=30, azim=-45)
                else:  # bottom
                    hemisphere_ax.view_init(elev=-30, azim=-45)
                
                # Set limits
                margin = sim.sphere_radius * 0.1
                max_range = sim.sphere_radius + margin
                hemisphere_ax.set_xlim(-max_range, max_range)
                hemisphere_ax.set_ylim(-max_range, max_range)
                hemisphere_ax.set_zlim(-max_range, max_range)
            
            hemisphere_ax.set_title(title)
        
        # Add a shared colorbar
        if color_by != "uniform" and cbar_label:
            # Position colorbar between the two subplots
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sc, cax=cbar_ax)
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
        hemisphere_suffix = "_split" if split_hemispheres else ""
        suffix = f"_blobs_{view}_{color_by}{hemisphere_suffix}.png"
        out_path = out_dir / (Path(base_path).name + suffix)
        fig.savefig(out_path.as_posix(), dpi=180, bbox_inches='tight')
        print(f"[info] Saved blob positions to {out_path}")
    
    return fig, (axes if split_hemispheres else ax)

# ----------------------------- Wavelength Analysis -----------------------------

@dataclass
class WavelengthResult:
    """Results from wavelength analysis using slope-based segmentation."""
    coherent_wavelengths: np.ndarray    # wavelengths of coherent wave segments (in radians)
    coherent_wave_types: np.ndarray     # +1 for positive wave, -1 for negative wave
    coherent_fractions: np.ndarray      # fraction of equator covered by each segment
    mean_wavelength_positive: float     # mean wavelength of positive waves (rad)
    mean_wavelength_negative: float     # mean wavelength of negative waves (rad)
    std_wavelength_positive: float      # std of positive wave wavelengths
    std_wavelength_negative: float      # std of negative wave wavelengths
    percent_positive_wave: float        # percentage of equator with positive waves
    percent_negative_wave: float        # percentage of equator with negative waves
    percent_incoherent: float           # percentage with no coherent wave
    n_positive_segments: int            # number of positive wave segments
    n_negative_segments: int            # number of negative wave segments
    n_time_points: int                  # number of time points analyzed

def estimate_wavelength_slopes(base_path: str,
                               sim: Optional[SimulationData]=None,
                               num_steps: Optional[int]=None,
                               n_time_points: int = 20,
                               ruptures_penalty: float = 5.0,
                               min_slope_threshold: float = 0.1,
                               show_analysis: bool = True) -> WavelengthResult:
    """
    Estimate wavelength using slope-based segmentation of phase patterns.
    
    For each of the last n_time_points, this function:
    1. Takes phase as a function of azimuthal position
    2. Numerically differentiates to get slope
    3. Uses ruptures library to detect constant-slope segments
    4. Classifies segments as positive/negative waves
    5. Calculates wavelength for each coherent segment
    
    Args:
        base_path: Simulation file prefix
        sim: Pre-loaded simulation data (optional)
        num_steps: Steps per period for time normalization
        n_time_points: Number of final time points to analyze
        ruptures_penalty: Penalty parameter for ruptures algorithm (higher = fewer segments)
        min_slope_threshold: Minimum absolute slope to be considered coherent
        show_analysis: Whether to show diagnostic plots
    
    Returns:
        WavelengthResult with slope-based wavelength estimates
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("This analysis requires the 'ruptures' package. Install with: pip install ruptures")
    
    if sim is None:
        sim = load_simulation(base_path, num_steps=num_steps)
    
    # Get phases in sorted azimuthal order for the last n_time_points
    T = sim.phases.shape[0]
    t_start = max(0, T - n_time_points)
    phases_sorted = sim.phases[t_start:, sim.order_idx]  # shape (n_time_points, N)
    phi_sorted = sim.basal_phi[sim.order_idx]  # shape (N,)
    N = len(phi_sorted)
    
    print(f"[info] Analyzing {n_time_points} time points with {N} cilia")
    
    # Storage for all segments across all time points
    all_wavelengths = []
    all_wave_types = []
    all_fractions = []
    
    # For visualization
    example_time_idx = -1  # Last time point
    example_segments = None
    example_slopes = None
    example_breakpoints = None
    
    for t_idx in range(phases_sorted.shape[0]):
        phase_pattern = phases_sorted[t_idx, :]  # shape (N,)
        
        # Make phase continuous by unwrapping, then wrap back for periodicity
        phase_unwrapped = np.unwrap(phase_pattern)
        
        # Handle periodicity: append first point at end with 2π shift
        phi_extended = np.concatenate([phi_sorted, [2*np.pi]])
        phase_extended = np.concatenate([phase_unwrapped, [phase_unwrapped[0] + 2*np.pi]])
        
        # Compute slopes (numerical derivative)
        # Using central differences where possible
        slopes = np.gradient(phase_extended, phi_extended)
        slopes = slopes[:-1]  # Remove last point (was added for periodicity)
        
        # Detect changepoints in slope using ruptures
        algo = rpt.Pelt(model="rbf").fit(slopes)
        breakpoints = algo.predict(pen=ruptures_penalty)
        
        # breakpoints includes the final index, so we process segments
        start_idx = 0
        for break_idx in breakpoints:
            # Get segment
            segment_slopes = slopes[start_idx:break_idx]
            segment_phi = phi_sorted[start_idx:break_idx]
            
            if len(segment_slopes) == 0:
                start_idx = break_idx
                continue
            
            # Compute mean slope for this segment
            mean_slope = np.mean(segment_slopes)
            
            # Check if slope is significant (coherent wave)
            if abs(mean_slope) > min_slope_threshold:
                # This is a coherent wave segment
                wave_type = 1 if mean_slope > 0 else -1
                
                # Calculate wavelength: slope = dψ/dφ
                # For 2π phase advance: Δφ = 2π / slope
                wavelength_rad = abs(2*np.pi / mean_slope)
                
                # Calculate fraction of equator covered by this segment
                segment_length = phi_sorted[break_idx-1] - phi_sorted[start_idx]
                if start_idx == 0 and break_idx == len(phi_sorted):
                    # Wraps around
                    segment_length = 2*np.pi
                fraction = segment_length / (2*np.pi)
                
                all_wavelengths.append(wavelength_rad)
                all_wave_types.append(wave_type)
                all_fractions.append(fraction)
            
            start_idx = break_idx
        
        # Save example for visualization
        if t_idx == phases_sorted.shape[0] - 1:
            example_slopes = slopes
            example_breakpoints = breakpoints
    
    # Convert to arrays
    all_wavelengths = np.array(all_wavelengths)
    all_wave_types = np.array(all_wave_types)
    all_fractions = np.array(all_fractions)
    
    # Separate by wave type
    pos_mask = all_wave_types > 0
    neg_mask = all_wave_types < 0
    
    pos_wavelengths = all_wavelengths[pos_mask]
    neg_wavelengths = all_wavelengths[neg_mask]
    pos_fractions = all_fractions[pos_mask]
    neg_fractions = all_fractions[neg_mask]
    
    # Calculate statistics
    mean_wl_pos = np.mean(pos_wavelengths) if len(pos_wavelengths) > 0 else np.inf
    mean_wl_neg = np.mean(neg_wavelengths) if len(neg_wavelengths) > 0 else np.inf
    std_wl_pos = np.std(pos_wavelengths) if len(pos_wavelengths) > 0 else 0.0
    std_wl_neg = np.std(neg_wavelengths) if len(neg_wavelengths) > 0 else 0.0
    
    percent_pos = 100.0 * np.sum(pos_fractions)
    percent_neg = 100.0 * np.sum(neg_fractions)
    percent_incoherent = 100.0 - percent_pos - percent_neg
    
    # Visualization
    if show_analysis and example_slopes is not None:
        try:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Define colors from cmcrameri palettes
            highlight_color = SEQUENTIAL_CMAP(0.15)
            pos_color = SEQUENTIAL_CMAP(0.85)
            neg_color = SEQUENTIAL_CMAP(0.45)
            
            # Plot 1: Phase pattern (example from last time point)
            ax1 = fig.add_subplot(gs[0, :])
            phase_example = phases_sorted[-1, :]
            ax1.plot(phi_sorted, phase_example, '.-', markersize=4, 
                    label='phase pattern', alpha=0.7, color=highlight_color)
            ax1.set_xlabel('azimuth φ (rad)')
            ax1.set_ylabel('phase ψ (rad)')
            ax1.set_title('Example phase pattern (last time point)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks([0, np.pi, 2*np.pi])
            ax1.set_xticklabels(['0', 'π', '2π'])
            ax1.set_xlim(0, 2*np.pi)
            
            # Plot 2: Slopes with detected segments
            ax2 = fig.add_subplot(gs[1, :])
            rpt.display(example_slopes, example_breakpoints, figsize=(12, 3))
            ax2 = plt.gca()
            ax2.set_xlabel('cilia index')
            ax2.set_ylabel('dψ/dφ (slope)')
            ax2.set_title('Detected slope segments')
            
            # Plot 3: Wavelength distributions
            ax3 = fig.add_subplot(gs[2, 0])
            if len(pos_wavelengths) > 0:
                ax3.hist(pos_wavelengths, bins=20, alpha=0.7, 
                        color=pos_color, edgecolor='black', linewidth=0.5,
                        label=f'Positive waves (n={len(pos_wavelengths)})')
            if len(neg_wavelengths) > 0:
                ax3.hist(neg_wavelengths, bins=20, alpha=0.7,
                        color=neg_color, edgecolor='black', linewidth=0.5,
                        label=f'Negative waves (n={len(neg_wavelengths)})')
            ax3.axvline(mean_wl_pos, color=pos_color, linestyle='--', linewidth=2)
            ax3.axvline(mean_wl_neg, color=neg_color, linestyle='--', linewidth=2)
            ax3.set_xlabel('wavelength (rad)')
            ax3.set_ylabel('count')
            ax3.set_title('Wavelength distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.axis('off')
            
            summary_text = f"""
WAVELENGTH ANALYSIS SUMMARY
{'='*35}

Positive Waves:
  Mean wavelength: {mean_wl_pos:.3f} rad
  Std deviation:   {std_wl_pos:.3f} rad
  Coverage:        {percent_pos:.1f}%
  N segments:      {np.sum(pos_mask)}

Negative Waves:
  Mean wavelength: {mean_wl_neg:.3f} rad
  Std deviation:   {std_wl_neg:.3f} rad
  Coverage:        {percent_neg:.1f}%
  N segments:      {np.sum(neg_mask)}

Incoherent:        {percent_incoherent:.1f}%

Time points analyzed: {n_time_points}
            """
            
            ax4.text(0.1, 0.5, summary_text, fontfamily='monospace',
                    fontsize=10, verticalalignment='center')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"[warn] Visualization failed: {e}")
    
    print("\n" + "="*60)
    print("WAVELENGTH ANALYSIS RESULTS (Slope-Based Method)")
    print("="*60)
    print(f"Positive waves: {percent_pos:.1f}% coverage, mean λ = {mean_wl_pos:.3f} rad")
    print(f"Negative waves: {percent_neg:.1f}% coverage, mean λ = {mean_wl_neg:.3f} rad")
    print(f"Incoherent:     {percent_incoherent:.1f}%")
    print(f"Total segments analyzed: {len(all_wavelengths)}")
    print("="*60)
    
    return WavelengthResult(
        coherent_wavelengths=all_wavelengths,
        coherent_wave_types=all_wave_types,
        coherent_fractions=all_fractions,
        mean_wavelength_positive=mean_wl_pos,
        mean_wavelength_negative=mean_wl_neg,
        std_wavelength_positive=std_wl_pos,
        std_wavelength_negative=std_wl_neg,
        percent_positive_wave=percent_pos,
        percent_negative_wave=percent_neg,
        percent_incoherent=percent_incoherent,
        n_positive_segments=int(np.sum(pos_mask)),
        n_negative_segments=int(np.sum(neg_mask)),
        n_time_points=n_time_points
    )

