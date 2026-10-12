"""
Still image of one prescribed-beat cilium above a no-slip wall, with instantaneous streamlines.

The run must have saved segment positions (_seg_states.dat) and phases (_true_states.dat);
segment forces are not needed. For the chosen frame:
  1. segment velocities U come from central differences of the saved positions,
  2. segment forces F solve M F = U, with M the RPY mobility plus the Swan & Brady (2007)
     infinite-wall corrections (numpy port of rpy_interaction in src/cilia/cuda_functions.cu),
  3. the flow is u(x) = sum_j (1 + a^2/6 lap_y) G_Blake(x, y_j) F_j (Blakelet as in
     src/flow_field/flow_field_evaluator.cu), i.e. the RPY field of each segment above the wall.
Only the chosen cilium is a flow source: its neighbours are left out.

Outputs: a 3D PyVista render (streamlines_*.png) and a 2D matplotlib streamplot of the
in-plane flow on the beat plane through the cilium base (beatplane_*.png).

Usage (from the repo root, with the PyVista virtualenv):
  ~/venvs/pyvista/bin/python analysis/cilium_streamlines.py <base_path> --fil 1
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

if not os.environ.get("DISPLAY"):
    # Headless: go straight to EGL instead of failing on X first.
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from plotting_and_analysis_functions import load_simulation, SEQUENTIAL_CMAP

A = 1.0   # segment radius (RSEG); the Swan-Brady port assumes A = MU = 1
MU = 1.0


# ----------------------------- Input -----------------------------

def read_par(base_path):
    """Parse '<value> %% <NAME>' lines of the .par file into a dict of floats."""
    par = {}
    with open(base_path + ".par") as f:
        for line in f:
            if "%%" in line:
                value, name = line.split("%%")
                par[name.strip()] = float(value)
    return par


def choose_frame(phases_raw, target):
    """Frame nearest the last time the unwrapped phase passed target (mod 2 pi).

    Frames 0 and T-1 are excluded because the velocity needs a neighbour on each side.
    """
    T = len(phases_raw)
    n = np.floor((phases_raw - target)/(2*np.pi))
    crossings = np.nonzero(np.diff(n) > 0)[0]
    for k in crossings[::-1]:
        candidates = [j for j in (k, k + 1) if 1 <= j <= T - 2]
        if candidates:
            d = [abs(np.angle(np.exp(1j*(phases_raw[j] - target)))) for j in candidates]
            return candidates[int(np.argmin(d))]
    raise ValueError("Phase never reaches the target value away from the first/last frame.")


# ----------------------------- Hydrodynamics -----------------------------

def swan_brady_mobility(Y):
    """3n x 3n translational RPY mobility above a no-slip wall at z = 0 (A = MU = 1).

    Port of the force -> velocity part of rpy_interaction in src/cilia/cuda_functions.cu.
    Block (i, j) gives the velocity of segment i due to the force on segment j.
    """
    n = len(Y)
    M = np.zeros((n, 3, n, 3))
    c = 1.0/(6.0*np.pi)
    I = np.eye(3)
    zhat = np.array([0.0, 0.0, 1.0])

    for i in range(n):
        hm1 = 1.0/Y[i, 2]
        a0 = c - (9.0*hm1 - 2.0*hm1**3 + hm1**5)*c/16.0
        a1 = c - (9.0*hm1 - 4.0*hm1**3 + hm1**5)*c/8.0
        M[i, :, i, :] = np.diag([a0, a0, a1])

        for j in range(n):
            if j == i:
                continue
            # Unbounded RPY (no overlap: segment spacing exceeds 2A)
            r = Y[i] - Y[j]
            rm1 = 1.0/np.linalg.norm(r)
            rhat = r*rm1
            block = rm1/(8.0*np.pi)*((1.0 + 2.0/3.0*rm1**2)*I + (1.0 - 2.0*rm1**2)*np.outer(rhat, rhat))

            # Wall-induced corrections
            R = np.array([r[0], r[1], Y[i, 2] + Y[j, 2]])
            rm1 = 1.0/np.linalg.norm(R)
            rm3 = rm1**3
            rm5 = rm1**5
            e = R*rm1
            ez = e[2]
            ez2 = ez*ez
            h = Y[j, 2]/(Y[i, 2] + Y[j, 2])

            al0 = -(0.75*rm1*(1.0 + 2.0*h*(1.0 - h)*ez2) + 0.5*(rm3*(1.0 - 3.0*ez2) - rm5*(1.0 - 5.0*ez2)))
            b1 = -(0.75*rm1*(1.0 - 6.0*h*(1.0 - h)*ez2) - 1.5*rm3*(1.0 - 5.0*ez2) + 2.5*rm5*(1.0 - 7.0*ez2))
            b2 = ez*(1.5*h*rm1*(1.0 - 6.0*(1.0 - h)*ez2) - 3.0*rm3*(1.0 - 5.0*ez2) + 5.0*rm5*(2.0 - 7.0*ez2))
            b3 = 1.5*h*rm1 - 5.0*rm5
            b4 = -((3.0*h*h*rm1 + 3.0*rm3)*ez2 + (2.0 - 15.0*ez2)*rm5)

            # V += c*(al0*F + (b1*(e.F) + b2*Fz)*e + zhat*(b3*ez*(e.F) + b4*Fz))
            block += c*(al0*I + b1*np.outer(e, e) + b2*np.outer(e, zhat)
                        + b3*ez*np.outer(zhat, e) + b4*np.outer(zhat, zhat))
            M[i, :, j, :] = block

    return M.reshape(3*n, 3*n)


def blakelet_velocity(X, Y, F):
    """Velocity at points X (P,3) due to point forces F (S,3) at Y (S,3) above the wall z = 0.

    Port of the Blakelet in src/flow_field/flow_field_evaluator.cu, including its 1/(8 pi MU).
    """
    dx = X[:, None, 0] - Y[None, :, 0]
    dy = X[:, None, 1] - Y[None, :, 1]
    dz = X[:, None, 2] - Y[None, :, 2]
    xz = X[:, None, 2]
    yz = Y[None, :, 2]
    fx, fy, fz = F[None, :, 0], F[None, :, 1], F[None, :, 2]

    # Stokeslet
    rm2 = 1.0/(dx*dx + dy*dy + dz*dz)
    rm1 = np.sqrt(rm2)
    s = rm2*(dx*fx + dy*fy + dz*fz)
    vx = rm1*(fx + dx*s)
    vy = rm1*(fy + dy*s)
    vz = rm1*(fz + dz*s)

    # Minus the Stokeslet at the image point
    Rz = xz + yz
    R2 = dx*dx + dy*dy + Rz*Rz
    Rm1 = np.sqrt(1.0/R2)
    S = (dx*fx + dy*fy + Rz*fz)*Rm1*Rm1
    vx -= Rm1*(fx + dx*S)
    vy -= Rm1*(fy + dy*S)
    vz -= Rm1*(fz + Rz*S)

    # Higher-order image terms
    Dxx = (3.0*dx*dx - R2)*xz
    Dxy = 3.0*dx*dy*xz
    Dxz = (R2 - 3.0*Rz*xz)*dx
    Dyy = (3.0*dy*dy - R2)*xz
    Dyz = (R2 - 3.0*Rz*xz)*dy
    Dzx = (3.0*Rz*xz + R2)*dx
    Dzy = (3.0*Rz*xz + R2)*dy
    Dzz = (R2 - 3.0*Rz*Rz)*xz
    Dfac = 2.0*yz*Rm1**5
    vx += Dfac*(Dxx*fx + Dxy*fy + Dxz*fz)
    vy += Dfac*(Dxy*fx + Dyy*fy + Dyz*fz)
    vz += Dfac*(Dzx*fx + Dzy*fy + Dzz*fz)

    v = np.stack([vx.sum(axis=1), vy.sum(axis=1), vz.sum(axis=1)], axis=1)
    return v/(8.0*np.pi*MU)


def rpy_blake_velocity(X, Y, F, h=1e-2):
    """(1 + A^2/6 lap_y) of the Blakelet, with lap_y by central differences in the source position."""
    v0 = blakelet_velocity(X, Y, F)
    lap = -6.0*v0
    for k in range(3):
        shift = np.zeros(3)
        shift[k] = h
        lap += blakelet_velocity(X, Y + shift, F) + blakelet_velocity(X, Y - shift, F)
    return v0 + (A*A/6.0)*lap/(h*h)


def flow_field(X, Y, F, U, chunk=20000):
    """RPY-Blake flow at X; points inside a segment move rigidly with it."""
    out = np.empty_like(X)
    for start in range(0, len(X), chunk):
        Xc = X[start:start + chunk]
        u = rpy_blake_velocity(Xc, Y, F)
        d2 = ((Xc[:, None, :] - Y[None, :, :])**2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        inside = d2[np.arange(len(Xc)), nearest] < A*A
        u[inside] = U[nearest[inside]]
        out[start:start + chunk] = u
    return out


def sphere_quadrature(n_theta=24, n_phi=48):
    """Unit normals and weights (summing to 1) for averaging over a sphere: Gauss-Legendre x trapezoid."""
    mu, w_mu = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0*np.pi*np.arange(n_phi)/n_phi
    MU_, PHI = np.meshgrid(mu, phi, indexing="ij")
    st = np.sqrt(1.0 - MU_**2)
    normals = np.stack([st*np.cos(PHI), st*np.sin(PHI), MU_], axis=-1).reshape(-1, 3)
    weights = (np.repeat(w_mu, n_phi)/(2.0*n_phi))
    return normals, weights


def self_checks(M, Y, F, U, grid_points, grid_u):
    ok = True

    asym = np.abs(M - M.T).max()
    eig_min = np.linalg.eigvalsh(0.5*(M + M.T)).min()
    print(f"[check] mobility: max|M - M^T| = {asym:.2e}, min eigenvalue = {eig_min:.3e}")
    ok &= asym < 1e-10 and eig_min > 0

    wall = grid_points[:, 2] == grid_points[:, 2].min()
    slip = np.linalg.norm(rpy_blake_velocity(grid_points[wall], Y, F), axis=1).max()
    umax = np.linalg.norm(grid_u, axis=1).max()
    print(f"[check] no slip: max|u| on z = {grid_points[wall, 2].min():g} is {slip:.2e} ({slip/umax:.2e} of max|u|)")
    ok &= grid_points[wall, 2].min() == 0.0 and slip < 1e-10*umax

    normals, weights = sphere_quadrature()
    avg = np.array([weights @ rpy_blake_velocity(y + A*normals, Y, F) for y in Y])
    err = np.linalg.norm(avg - U, axis=1).max()/np.linalg.norm(U, axis=1).max()
    print(f"[check] Faxen: field averaged over each segment surface vs prescribed U, max rel. error = {err:.2e}")
    ok &= err < 1e-3

    if not ok:
        sys.exit("Self-checks failed: not rendering.")


# ----------------------------- Rendering -----------------------------

def direction_cones(streams, every=25.0):
    """Cones pointing along the flow, spaced every `every` units of arc length on each streamline."""
    ids = []
    for i in range(streams.n_cells):
        idx = np.asarray(streams.get_cell(i).point_ids)
        xyz = streams.points[idx]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        ids.extend(idx[np.searchsorted(arc, np.arange(0.5*every, arc[-1], every))])
    ids = np.unique(ids)
    pts = pv.PolyData(np.asarray(streams.points[ids]))
    pts["U"] = np.asarray(streams["U"][ids])
    pts["speed"] = np.asarray(streams["speed"][ids])
    return pts.glyph(orient="U", scale=False, factor=2.5, geom=pv.Cone(radius=0.3, height=1.4, resolution=24))


def render(grid, Y, base, args, out_png):
    speed_max = float(grid["speed"].max())
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    zs = np.linspace(3.0, Y[:, 2].max() + 10.0, 9)
    if args.seed == "lines":
        # Vertical lines through the cilium and 5 radii either side of it.
        c = Y.mean(axis=0)
        seed_pts = np.vstack([np.column_stack([np.full(zs.size, c[0] + dx), np.full(zs.size, c[1]), zs])
                              for dx in (-5.0, 0.0, 5.0)])
    else:
        # A grid in the beat plane through the cilium base.
        a, b = np.meshgrid(np.linspace(ymin + 5.0, ymax - 5.0, 7), zs, indexing="ij")
        seed_pts = np.column_stack([np.full(a.size, base[0]), a.ravel(), b.ravel()])

    streams = grid.streamlines_from_source(pv.PolyData(seed_pts), vectors="U", integration_direction="both",
                                           initial_step_length=0.2, max_step_length=0.5,
                                           max_steps=20000, max_length=2000.0,
                                           terminal_speed=1e-5*speed_max)
    clim = [0.0, float(np.percentile(streams["speed"], 95))]
    print(f"[render] {streams.n_lines} streamlines, colour range {clim[0]:.0f}-{clim[1]:.0f}")

    p = pv.Plotter(off_screen=True, window_size=(2000, 1500))
    p.set_background("white")
    p.add_mesh(streams.tube(radius=0.3, n_sides=12), scalars="speed", cmap=SEQUENTIAL_CMAP, clim=clim,
               smooth_shading=True, show_scalar_bar=False)
    p.add_scalar_bar(title="speed (a/T)", color="black", vertical=True, position_x=0.9, position_y=0.3,
                     height=0.45, width=0.04, title_font_size=28, label_font_size=24, fmt="%.0f")
    p.add_mesh(direction_cones(streams), scalars="speed", cmap=SEQUENTIAL_CMAP, clim=clim, show_scalar_bar=False)

    segs = pv.PolyData(Y).glyph(geom=pv.Sphere(radius=A, theta_resolution=32, phi_resolution=32),
                                scale=False, orient=False)
    p.add_mesh(segs, color="#2b2b2b", smooth_shading=True, specular=0.4)

    wall = pv.Plane(center=(0.5*(xmin + xmax), 0.5*(ymin + ymax), 0.0), direction=(0, 0, 1),
                    i_size=xmax - xmin, j_size=ymax - ymin)
    p.add_mesh(wall, color="#c8c8c8", ambient=0.3)

    focus = np.array([base[0], base[1], 0.5*Y[:, 2].max()])
    az, el = np.radians(args.azimuth), np.radians(args.elevation)
    dist = 2.2*max(ymax - ymin, zmax - zmin)
    p.camera_position = [tuple(focus + dist*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])),
                         tuple(focus), (0.0, 0.0, 1.0)]
    p.camera.view_angle = 25.0
    p.screenshot(str(out_png))
    print(f"[render] wrote {out_png}")


# ----------------------------- Beat plane (2D) -----------------------------

def beat_direction(X, base, dt_frame):
    """Unit horizontal vector along which the cilium sweeps, pointing in the effective-stroke direction.

    The axis is the dominant horizontal direction of the segment positions over the whole run;
    its sign makes the tip move along +e_b at the run's fastest tip speed (the effective stroke).
    """
    rel = (X[..., :2] - base[:2]).reshape(-1, 2)
    rel = rel - rel.mean(axis=0)
    e = np.linalg.eigh(rel.T @ rel)[1][:, -1]
    v_tip = (X[2:, -1, :2] - X[:-2, -1, :2])/(2.0*dt_frame)
    if v_tip[np.argmax(np.linalg.norm(v_tip, axis=1))] @ e < 0:
        e = -e
    return np.array([e[0], e[1], 0.0])


def beat_plane_slice(X, Y, F, U, base, e_b, spacing, margin):
    """Flow on the vertical plane through the cilium base spanned by e_b and z."""
    s_all = (X[..., :2] - base[:2]) @ e_b[:2]
    s = np.arange(np.floor((s_all.min() - margin)/spacing)*spacing, s_all.max() + margin + 0.5*spacing, spacing)
    z = np.arange(0.0, X[..., 2].max() + margin + 0.5*spacing, spacing)
    S, Z = np.meshgrid(s, z)
    pts = np.column_stack([base[0] + S.ravel()*e_b[0], base[1] + S.ravel()*e_b[1], Z.ravel()])
    return s, z, pts, flow_field(pts, Y, F, U)


def plot_beat_plane(s, z, u, Y, F, base, e_b, title, out_png, arrow_length=12.0):
    """Streamplot of the in-plane flow, the segments as black discs, and red arrows for the
    in-plane part of each segment's force on the fluid (the longest arrow is arrow_length radii)."""
    shape = (len(z), len(s))
    ub = (u @ e_b).reshape(shape)
    uz = u[:, 2].reshape(shape)
    speed = np.hypot(ub, uz)
    normal = np.cross(e_b, [0.0, 0.0, 1.0])
    off = np.abs(u @ normal).max()/np.linalg.norm(u, axis=1).max()
    vmax = float(np.percentile(speed, 99))
    print(f"[2d] out-of-plane flow on the slice: max|u_n| = {off:.1%} of max|u|; "
          f"colour map {getattr(SEQUENTIAL_CMAP, 'name', SEQUENTIAL_CMAP)}, range 0-{vmax:.0f}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    strm = ax.streamplot(s, z, ub, uz, color=speed, cmap=SEQUENTIAL_CMAP, norm=Normalize(0.0, vmax),
                         density=1.6, linewidth=0.9, arrowsize=0.8)
    ax.axhspan(-4.0, 0.0, color="0.85", zorder=0)
    ax.axhline(0.0, color="0.4", lw=1.0)
    ys = (Y[:, :2] - base[:2]) @ e_b[:2]
    for y_s, y_z in zip(ys, Y[:, 2]):
        # No edge: a stroked edge would add half its width to the radius (~0.15 a at 1 pt, 300 dpi).
        ax.add_patch(Circle((y_s, y_z), A, facecolor="black", edgecolor="none", linewidth=0, zorder=3))

    fb, fz = F @ e_b, F[:, 2]
    f_off = np.abs(F @ normal).max()/np.linalg.norm(F, axis=1).max()
    f_max = np.hypot(fb, fz).max()
    print(f"[2d] segment forces: max in-plane |F| = {f_max:.0f}, out-of-plane part = {f_off:.1%} of max|F|")
    q = ax.quiver(ys, Y[:, 2], fb, fz, color="#d01c1c", angles="xy", scale_units="xy",
                  scale=f_max/arrow_length, width=0.003, headwidth=4, zorder=4)
    f_key = float(f"{0.5*f_max:.1g}")
    ax.quiverkey(q, 0.96, 1.035, f_key, f"force on fluid: {f_key:g} $\\mu a^2/T$", labelpos="W",
                 coordinates="axes", fontproperties={"size": 9})
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(-4.0, z[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("distance from the cilium base along the beat direction (a)")
    ax.set_ylabel("height above the wall (a)")
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(strm.lines, ax=ax, shrink=0.8, pad=0.02, extend="max")
    cbar.set_label("in-plane speed (a/T)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[2d] wrote {out_png}")


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_path", help="Simulation file prefix (without _seg_states.dat etc.)")
    ap.add_argument("--fil", type=int, default=1, help="Index of the cilium to show (default 1)")
    ap.add_argument("--phase", type=float, default=0.3, help="Target phase in units of pi (default 0.3)")
    ap.add_argument("--frame", type=int, default=None, help="Saved frame to use; overrides --phase")
    ap.add_argument("--spacing", type=float, default=0.75, help="Flow grid spacing in segment radii")
    ap.add_argument("--margin", type=float, default=20.0, help="Grid margin around the cilium's swept volume")
    ap.add_argument("--seed", choices=["lines", "beat"], default="lines",
                    help="Seed on vertical lines through and beside the cilium, or on a grid in the beat plane")
    ap.add_argument("--azimuth", type=float, default=-30.0, help="Camera azimuth from +x, degrees")
    ap.add_argument("--elevation", type=float, default=12.0, help="Camera elevation above the wall, degrees")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent/"analysis_output"))
    ap.add_argument("--save-vti", action="store_true", help="Also write the 3D flow grid as .vti")
    ap.add_argument("--plots", choices=["both", "2d", "3d"], default="both",
                    help="Which figures to make: 3D PyVista render, 2D beat-plane streamplot, or both")
    args = ap.parse_args()

    par = read_par(args.base_path)
    if par["RSEG"] != A or par["MU"] != MU:
        sys.exit("The Swan-Brady port assumes RSEG = MU = 1.")
    dt_frame = par["DT"]*par["PLOT_FREQUENCY_IN_STEPS"]

    sim = load_simulation(args.base_path, num_steps=int(par["STEPS_PER_PERIOD"]))
    X = sim.seg_positions[:, args.fil].astype(np.float64)   # (T, S, 3)

    k = args.frame if args.frame is not None else choose_frame(sim.phases_raw[:, args.fil], args.phase*np.pi)
    if not 1 <= k <= len(X) - 2:
        sys.exit(f"Frame {k} has no neighbour on both sides.")
    print(f"[frame] {k}: t = {sim.times[k]:.2f} periods, phase = {sim.phases[k, args.fil]/np.pi:.3f} pi")

    Y = X[k]
    U = (X[k + 1] - X[k - 1])/(2.0*dt_frame)
    M = swan_brady_mobility(Y)
    F = np.linalg.solve(M, U.ravel()).reshape(-1, 3)
    print(f"[forces] max|U| = {np.linalg.norm(U, axis=1).max():.1f}, total force = {F.sum(axis=0).round(1)}")

    base = sim.basal_pos[args.fil]
    e_b = beat_direction(X, base, dt_frame)
    s, z, plane_pts, plane_u = beat_plane_slice(X, Y, F, U, base, e_b, args.spacing, args.margin)
    self_checks(M, Y, F, U, plane_pts, plane_u)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"wall_fil{args.fil}_frame{k}"

    if args.plots in ("both", "2d"):
        title = f"cilium {args.fil}, t = {sim.times[k]:.2f} T, phase = {sim.phases[k, args.fil]/np.pi:.2f}$\\pi$"
        plot_beat_plane(s, z, plane_u, Y, F, base, e_b, title, outdir/f"beatplane_{stem}.png")

    if args.plots in ("both", "3d"):
        lo = X.reshape(-1, 3).min(axis=0) - args.margin
        hi = X.reshape(-1, 3).max(axis=0) + args.margin
        lo[2] = 0.0
        dims = np.ceil((hi - lo)/args.spacing).astype(int) + 1
        grid = pv.ImageData(dimensions=tuple(dims), spacing=(args.spacing,)*3, origin=tuple(lo))
        t0 = time.time()
        u = flow_field(np.asarray(grid.points, dtype=np.float64), Y, F, U)
        print(f"[flow] {grid.n_points} grid points in {time.time() - t0:.1f} s")
        grid["U"] = u
        grid["speed"] = np.linalg.norm(u, axis=1)
        if args.save_vti:
            grid.save(str(outdir/f"flow_{stem}.vti"))
        render(grid, Y, base, args, outdir/f"streamlines_{stem}.png")


if __name__ == "__main__":
    main()
