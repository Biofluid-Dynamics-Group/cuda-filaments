import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wrap_to_pi(angle):
	return (angle + np.pi) % (2.0 * np.pi) - np.pi


def find_true_states_files(data_dir, tilt_factor):
	patterns = [
		f"pair_wall_tilt{tilt_factor}_spacing*",
		f"pair_wall_tilt{tilt_factor}_spacing*_0",
	]

	run_dirs = []
	for pattern in patterns:
		run_dirs.extend(sorted(data_dir.glob(pattern)))

	# Keep only unique directory paths, preserving order.
	unique_dirs = []
	seen = set()
	for d in run_dirs:
		if d.is_dir() and d not in seen:
			unique_dirs.append(d)
			seen.add(d)

	true_state_entries = []
	for run_dir in unique_dirs:
		date_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])
		if not date_dirs:
			continue

		# Use the latest date folder by lexicographic order (e.g. 20260317).
		latest_dir = date_dirs[-1]
		files = sorted(latest_dir.glob("*_true_states.dat"))
		if not files:
			continue

		spacing_match = re.search(r"spacing([0-9]*\.?[0-9]+)", run_dir.name)
		if spacing_match is None:
			continue

		spacing = float(spacing_match.group(1))
		true_state_entries.append((spacing, files[0]))

	true_state_entries.sort(key=lambda x: x[0])
	return true_state_entries


def load_phase_diff(true_states_file, steps_per_period):
	data = np.loadtxt(true_states_file)
	if data.ndim == 1:
		data = data.reshape(1, -1)

	if data.shape[1] < 4:
		raise ValueError(
			f"Expected at least 4 columns in {true_states_file}, got {data.shape[1]}"
		)

	save_step = data[:, 0]
	period = data[:, 1]
	phase_1 = data[:, 2]
	phase_2 = data[:, 3]

	# Time in units of periods: t / T = step / steps_per_period.
	t_over_T = save_step / float(steps_per_period)
	# Also provide physical time in case needed externally.
	t = t_over_T * period

	phase_diff = wrap_to_pi(phase_2 - phase_1)
	return t, t_over_T, phase_diff


def circular_mean(angle):
	return np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))


def estimate_wavelength_from_final_period(t_over_T, phase_diff, spacing_L):
	# Average phase lag over the final unit interval in t/T (the final period).
	t_end = np.max(t_over_T)
	mask = t_over_T >= (t_end - 1.0)

	if not np.any(mask):
		mask = np.ones_like(t_over_T, dtype=bool)

	delta = circular_mean(phase_diff[mask])
	abs_delta = abs(delta)

	if abs_delta < 1e-10:
		wavelength_L = np.inf
	else:
		# spacing_L is already in units of L, so wavelength is also in units of L.
		wavelength_L = 2.0 * np.pi * spacing_L / abs_delta

	return delta, wavelength_L


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Plot phase difference between two cilia from pair_wall true_states files "
			"for multiple spacings."
		)
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("data"),
		help="Path to data directory (default: data)",
	)
	parser.add_argument(
		"--tilt-factor",
		type=str,
		default="1.0",
		help="Tilt-factor suffix used in directory naming, e.g. 1.0",
	)
	parser.add_argument(
		"--steps-per-period",
		type=int,
		default=500,
		help="Simulation steps per period used to convert save_step to t/T (default: 500)",
	)
	parser.add_argument(
		"--save",
		type=Path,
		default=None,
		help="Optional output image path. If omitted, plot is shown interactively.",
	)
	args = parser.parse_args()

	entries = find_true_states_files(args.data_dir, args.tilt_factor)
	if not entries:
		raise FileNotFoundError(
			f"No true_states files found for pair_wall_tilt{args.tilt_factor}_spacing* in {args.data_dir}"
		)

	plt.figure(figsize=(9, 5))
	print("Estimated metachronal wavelength from final period:")
	print("spacing/L\tmean Δψ [rad]\tλ/L")

	for spacing, file_path in entries:
		_, t_over_T, phase_diff = load_phase_diff(file_path, args.steps_per_period)
		delta_final, wavelength_L = estimate_wavelength_from_final_period(
			t_over_T, phase_diff, spacing
		)

		if np.isinf(wavelength_L):
			lambda_text = r"$\infty$"
			print(f"{spacing:g}\t{delta_final:.6f}\tinf")
		else:
			lambda_text = f"{wavelength_L:.3f}"
			print(f"{spacing:g}\t{delta_final:.6f}\t{wavelength_L:.6f}")

		plt.plot(
			t_over_T,
			phase_diff,
			linewidth=2.0,
			label=fr"spacing={spacing:g}L, $\lambda/L={lambda_text}$",
		)

	plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
	plt.xlabel(r"$t/T$")
	plt.ylabel(r"$\Delta\psi = \mathrm{wrap}(\psi_2 - \psi_1)$")
	plt.title(f"Phase difference vs time (tilt factor = {args.tilt_factor})")
	plt.legend()
	plt.grid(alpha=0.25)
	plt.tight_layout()

	if args.save is not None:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(args.save, dpi=200)
		print(f"Saved: {args.save}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
