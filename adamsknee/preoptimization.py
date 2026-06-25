import numpy as np
import pandas as pd

class PreOptimizationMixin:
    # ── Calibration setup ──────────────────────────────────────────────
    # Each design variable (Percent_L0_*) primarily controls one ligament
    # force. The relationship is monotonic (more slack length -> less force),
    # so the Jacobian is near-diagonal and the problem is well conditioned.
    # design-variable name -> (force column in .tab file, target force [N])
    PREOPT_TARGETS = {
        'Percent_L0_FFL':       ('Force_FFL',                1.0),
        'Percent_L0_LCL':       ('Force_LCL',               20.0),
        'Percent_L0_OPL':       ('Force_OPL_PL',            10.0),
        'Percent_L0_PCL_PM':    ('TotalForce_PCL_PM',       10.0),
        'Percent_L0_PLC':       ('TotalForce_PLC',           4.0),
        'Percent_L0_PMC':       ('TotalForce_PMC',           1.0),
        'Percent_L0_POL':       ('TotalForce_POL',          18.0),
        'Percent_L0_sMCL_Dist': ('TotalForce_sMCL_WrapDist', 4.0),
        'Percent_L0_sMCL_Prox': ('TotalForce_sMCL_WrapProx', 4.0),
    }
    # Variables held fixed during calibration (PCL_AL and ALL fibers @ 1.1).
    PREOPT_BOUNDS = (0.8, 1.2)

    def preoptimization(self, verbose=False, run_adams=True):
        cmd_file = self.cmd_path("step5A_preoptimization")
        if verbose:
            print(f"Creating simulation CMD file at: {cmd_file}")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_preoptimization(fid, verbose=verbose)
        if run_adams:
            self.run_adams(cmd_file)
            self._write_forces(verbose=verbose)

    # ────────────────────────────────────────────────────────────────────
    # Automated calibration
    # ────────────────────────────────────────────────────────────────────
    def optimize_preoptimization(self, method='coordinate', bounds=None,
                                 diff_step=0.03, max_nfev=200, weights=None,
                                 x0=None, force_tol=0.5, relax=0.8,
                                 max_step=0.06, save_history=True, verbose=True):
        """Automatically calibrate the ligament Percent_L0 design variables.

        Repeatedly runs the Adams forward simulation, reads back the ligament
        forces, and adjusts the design variables to drive each force toward its
        target (``PREOPT_TARGETS``). Replaces the manual edit-rerun loop on
        ``Optimization Initial Guesses.xlsx``.

        Parameters
        ----------
        method : {'coordinate', 'least_squares', 'minimize'}
            'coordinate' (default, cheapest) applies your physical rule -- if a
            force is above target, increase that ligament's Percent_L0; if it is
            below, decrease it -- using a per-variable secant/Newton step that
            auto-tunes the magnitude from a local sensitivity estimate. Costs
            ~1 simulation per iteration. 'least_squares' solves the full
            9-residual / 9-variable problem with a bounded trust-region
            Gauss-Newton method (more robust to ligament coupling, but each
            Jacobian costs ~9 extra sims). 'minimize' is a derivative-free
            Nelder-Mead on the summed squared force error.
        bounds : (lo, hi), optional
            Per-variable bounds. Defaults to ``PREOPT_BOUNDS`` (0.8, 1.2).
        diff_step : float
            Relative finite-difference / sensitivity-probe step.
        max_nfev : int
            Maximum number of forward simulations (iterations for 'coordinate').
        weights : array-like, optional
            Per-residual weights (same order as ``PREOPT_TARGETS``). Defaults
            to all ones (raw force errors, matching OBJ_SummedForceErrors).
        x0 : dict or array-like, optional
            Starting guess. Defaults to the values in the xlsx.
        force_tol : float, optional
            Stop early ('coordinate') once every force is within this many N of
            its target. Set to None to always run to max_nfev.
        relax : float
            Damping factor (0-1) on the 'coordinate' update step; <1 prevents
            overshoot/oscillation when ligaments are coupled.
        max_step : float
            Maximum absolute change in any Percent_L0 per 'coordinate' iteration.
        save_history : bool
            Write every evaluation to ``<subject>_preopt_history.csv`` so you
            can monitor progress while the (slow) sims run.

        Returns
        -------
        (result, best_vars) : the scipy result object (or a summary dict for
        'coordinate') and a dict of the best design-variable values found.
        """
        from scipy.optimize import least_squares, minimize

        if bounds is None:
            bounds = self.PREOPT_BOUNDS
        lo, hi = float(bounds[0]), float(bounds[1])

        free_names = list(self.PREOPT_TARGETS.keys())
        force_cols = [self.PREOPT_TARGETS[n][0] for n in free_names]
        targets = np.array([self.PREOPT_TARGETS[n][1] for n in free_names], dtype=float)
        n = len(free_names)

        weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

        # Initial guess
        guess_df = pd.read_excel(self.model_inputs_dir / "Optimization Initial Guesses.xlsx")
        guess_map = dict(zip(guess_df['Name'], guess_df['Value']))
        if x0 is None:
            x0_arr = np.array([float(guess_map[name]) for name in free_names])
        elif isinstance(x0, dict):
            x0_arr = np.array([float(x0.get(name, guess_map[name])) for name in free_names])
        else:
            x0_arr = np.asarray(x0, dtype=float)
        x0_arr = np.clip(x0_arr, lo, hi)

        history = []
        hist_path = self.lig_update_dir / f"{self.subject}_preopt_history.csv"

        def evaluate(x):
            override = {name: float(val) for name, val in zip(free_names, x)}
            forces = self._run_preopt_forward(override, verbose=False)
            measured = np.array([forces.get(col, np.nan) for col in force_cols], dtype=float)
            residual = (measured - targets) * weights
            sse = float(np.nansum(residual ** 2))
            obj = float(forces.get('OBJ_SummedForceErrors', np.nan))

            row = {'eval': len(history) + 1, 'SSE': sse, 'OBJ_SummedForceErrors': obj}
            row.update({name: float(val) for name, val in zip(free_names, x)})
            row.update({col: float(m) for col, m in zip(force_cols, measured)})
            history.append(row)
            if save_history:
                pd.DataFrame(history).to_csv(hist_path, index=False)
            if verbose:
                print(f"[eval {row['eval']:3d}] SSE={sse:11.3f}   OBJ={obj:12.1f}")
            return residual, measured

        if verbose:
            print(f"\nCalibrating {n} design variables for {self.subject} "
                  f"(method={method}, bounds=[{lo}, {hi}])")
            print(f"Free variables: {free_names}\n")

        if method == 'coordinate':
            result = self._optimize_coordinate(
                evaluate, x0_arr, targets, lo, hi,
                probe=diff_step, max_iter=max_nfev, force_tol=force_tol,
                relax=relax, max_step=max_step, verbose=verbose)
            x_best = result['x']
        elif method == 'least_squares':
            def residual_fun(x):
                residual, _ = evaluate(x)
                # large penalty if a sim failed / produced NaN
                return np.nan_to_num(residual, nan=1.0e3)

            result = least_squares(
                residual_fun, x0_arr,
                bounds=([lo] * n, [hi] * n),
                diff_step=diff_step,
                max_nfev=max_nfev,
                verbose=2 if verbose else 0,
            )
            x_best = result.x
        elif method == 'minimize':
            def scalar_fun(x):
                residual, _ = evaluate(x)
                val = float(np.nansum(residual ** 2))
                return val if np.isfinite(val) else 1.0e9

            result = minimize(
                scalar_fun, x0_arr,
                method='Nelder-Mead',
                bounds=[(lo, hi)] * n,
                options={'maxfev': max_nfev, 'xatol': 1e-3, 'fatol': 1e-2},
            )
            x_best = result.x
        else:
            raise ValueError(f"Unknown method: {method!r}")

        best_vars = {name: float(val) for name, val in zip(free_names, x_best)}

        # Persist the best design variables and do a final confirming run
        self._save_best_guesses(best_vars)
        self._run_preopt_forward(best_vars, verbose=False)
        summary_df = self._write_forces(verbose=verbose)

        if verbose:
            print("\nCalibration finished.")
            print("Best design variables:")
            for name in free_names:
                print(f"  {name:24s} {best_vars[name]:.4f}")
            print(f"\nHistory written to: {hist_path}")

        return result, best_vars

    def _optimize_coordinate(self, evaluate, x0, targets, lo, hi,
                             probe=0.03, max_iter=200, force_tol=0.5,
                             relax=0.8, max_step=0.06, verbose=True):
        """Per-variable secant/Newton calibration implementing the rule:
        force above target -> increase that Percent_L0; below -> decrease it.

        ``evaluate(x)`` must return ``(residual, measured_forces)``. The step
        size per variable is derived from a local sensitivity estimate
        ``s = d(force)/d(percent)`` (physically negative), so the update
        ``x -= relax * (force - target) / s`` automatically moves in the
        direction your intuition prescribes and scales itself sensibly.
        """
        n = len(x0)
        x = np.clip(np.asarray(x0, dtype=float), lo, hi)

        _, f = evaluate(x)                       # baseline
        # One simultaneous probe to estimate the (near-diagonal) sensitivities.
        x_probe = np.clip(x + probe, lo, hi)
        _, f_probe = evaluate(x_probe)
        dx = x_probe - x
        with np.errstate(divide='ignore', invalid='ignore'):
            sens = (f_probe - f) / dx
        # Keep the better of the two evaluated points as the current iterate.
        if np.nansum((f_probe - targets) ** 2) < np.nansum((f - targets) ** 2):
            x, f = x_probe, f_probe
        x_prev, f_prev = x_probe.copy(), f_probe.copy()

        best_x = x.copy()
        best_sse = float(np.nansum((f - targets) ** 2))
        converged = False

        for it in range(max_iter):
            err = f - targets
            max_abs = float(np.nanmax(np.abs(err)))
            if verbose:
                print(f"  [coord iter {it:3d}] max|force-target|={max_abs:8.3f} N   "
                      f"SSE={float(np.nansum(err ** 2)):11.3f}")
            if force_tol is not None and max_abs < force_tol:
                converged = True
                break

            # Sensitivity must be physically negative (more slack -> less force);
            # where the estimate is unreliable, fall back to a fixed-direction
            # step that still obeys the rule (force high -> increase percent).
            good = np.isfinite(sens) & (sens < -1e-6)
            step = np.where(
                good,
                -relax * err / np.where(good, sens, -1.0),
                relax * np.sign(np.nan_to_num(err)) * probe,
            )
            step = np.clip(step, -max_step, max_step)
            x_new = np.clip(x + step, lo, hi)

            x_prev, f_prev = x.copy(), f.copy()
            _, f_new = evaluate(x_new)

            # Secant update of the sensitivities where the variable actually moved.
            dx = x_new - x_prev
            moved = np.abs(dx) > 1e-9
            with np.errstate(divide='ignore', invalid='ignore'):
                new_sens = (f_new - f_prev) / dx
            sens = np.where(moved & np.isfinite(new_sens), new_sens, sens)

            x, f = x_new, f_new
            sse = float(np.nansum((f - targets) ** 2))
            if sse < best_sse:
                best_sse, best_x = sse, x.copy()

        return {'x': best_x, 'sse': best_sse, 'converged': converged,
                'n_iter': it + 1}

    def _run_preopt_forward(self, vars_override, verbose=False):
        """Write the CMD with the given variable overrides, run Adams, and
        return a dict of final-time ligament forces."""
        cmd_file = self.cmd_path("step5A_preoptimization")
        with open(cmd_file, "w", encoding="utf-8") as fid:
            self._write_preoptimization(fid, vars_override=vars_override, verbose=verbose)
        self.run_adams(cmd_file)
        return self._read_final_forces()

    def _read_final_forces(self):
        """Read the ligament force .tab file and return {column: final value}."""
        lig_df = pd.read_csv(
            self.lig_update_dir / f"{self.subject}_Ligament_Forces.tab",
            sep='\t', skiprows=1)
        lig_df.columns = lig_df.columns.str.strip()
        cols = [c for c in lig_df.columns if c != 'Time']
        for c in cols:
            lig_df[c] = pd.to_numeric(lig_df[c], errors='coerce')
        return lig_df[cols].iloc[-1].to_dict()

    def _save_best_guesses(self, override,
                           out_name="Optimization Best Guesses.xlsx"):
        """Write a copy of the guesses spreadsheet with the optimized values,
        leaving the original ``Optimization Initial Guesses.xlsx`` untouched."""
        df = pd.read_excel(self.model_inputs_dir / "Optimization Initial Guesses.xlsx")
        df['Value'] = df.apply(
            lambda r: override.get(r['Name'], r['Value']), axis=1)
        out_path = self.model_inputs_dir / out_name
        df.to_excel(out_path, index=False)
        return out_path

    def _write_preoptimization(self, fid, vars_override=None, verbose=False):
        vars_df = pd.read_excel(self.model_inputs_dir / "Optimization Initial Guesses.xlsx")
        if vars_override:
            vars_df = vars_df.copy()
            vars_df['Value'] = vars_df.apply(
                lambda r: vars_override.get(r['Name'], r['Value']), axis=1)
        
        # ── Read Binary File  ───────────────────────────────────────
        input_bin_file = f'{self.subject}_C5'
        fid.write('! ----- Binary File ----- !\n!\n')
        fid.write(f'file bin read file="{self.bin_dir}/{input_bin_file}.bin" \n!\n')
        fid.write('interface dialog undisplay dialog=.gui.info_window\n')
        fid.write('!\n!\n')
        for _, row in vars_df.iterrows():
            var_name = row['Name']
            var_value = row['Value']
            fid.write(f'variable modify  &\n')
            fid.write(f' variable_name = .{self.subject}.{var_name}  &\n')
            fid.write(f' real_value = {var_value} \n!\n')
        
        fid.write('force modify direct single_component_force  &\n')
        fid.write(f'    single_component_force = .{self.subject}.ForceCD  &\n')
        fid.write(f'    function = "10*step(time,0,0,.5,1)"\n')
        fid.write('!\n')
        
        fid.write('! ---------- Run Simulation ----- ! \n')
        fid.write('simulation single_run transient type=dynamic initial_static=no duration=1 step_size=0.01\n')
        
        # saved ligaments
        forcevars = ['Force_FFL', 'Force_LCL', 'Force_OPL_PL', 'OBJ_SummedForceErrors', 'TotalForce_PCL_PM' , 'TotalForce_PCL_AL' ,
                     'TotalForce_PLC', 'TotalForce_PMC', 'TotalForce_POL', 'TotalForce_sMCL_WrapDist', 'TotalForce_sMCL_WrapProx' ,
                     'Force_ALL' ]
        # -----------------------------
        # Ligament Length Plots
        # -----------------------------
        fid.write("xy_plot template modify plot=.plot_1 auto_title=yes auto_subtitle=yes auto_date=yes auto_analysis_name=yes table=no\n")
        fid.write("xy_plot template clear plot=.plot_1\n")
        for idx, var in enumerate(forcevars, start=1):
            fid.write(f"xy_plot curve create curve=.plot_1.curve_{idx} create_page=no "
                        f"calculate_axis_limits=no dmeasure=.{self.subject}.{var} "
                        f"imeasure=.{self.subject}.Time run=.{self.subject}.Last_Run auto_axis=UNITS\n")
        fid.write("xy_plot template calculate_axis_limits plot_name=.plot_1\n")
        fid.write('file table write &\n')
        out_lig_file = f'{self.lig_update_dir}/{self.subject}_Ligament_Forces'
        fid.write(f'   file_name = "{out_lig_file}"  &\n')
        fid.write(f'   plot_name = .plot_1 &\n')
        fid.write(f'   format = spreadsheet \n')
        fid.write('!\n!\n')
        
        
            
        # ── Write Binary File ───────────────────────────────────────────────
        output_bin_file = f'{self.subject}_C5_preOpt'
        fid.write('! ----- Write Binary File ----- !\n!\n')
        fid.write(f'file bin write file="{self.bin_dir}/{output_bin_file}.bin" \n!\n')
        
    def _write_forces(self, verbose=False):
        lig_df = pd.read_csv(
            self.lig_update_dir / f"{self.subject}_Ligament_Forces.tab", sep='\t', skiprows=1)
        lig_df.columns = lig_df.columns.str.strip()
        
        lig_def_names = lig_df.columns[1:]  # Exclude 'Time' column
        
        for col in lig_def_names:
            lig_df[col] = pd.to_numeric(lig_df[col], errors='coerce')
        
        # Get the last row (final time point) for each ligament column
        final_forces = lig_df[lig_def_names].iloc[-1]
        
        # Build summary table
        summary_df = pd.DataFrame({
            'Ligament': final_forces.index,
            'Force_N':  final_forces.values
        })
        
        if verbose:
            print(summary_df.to_string(index=False))
        
        # Write to CSV
        out_path = self.lig_update_dir / f"{self.subject}_Ligament_PeakForces.csv"
        summary_df.to_csv(out_path, index=False)
        
        if verbose:
            print(f"Final ligament forces written to: {out_path}")
        
    
        
        display_df = summary_df.copy()
        mask = display_df['Ligament'] == 'OBJ_SummedForceErrors'
        display_df = pd.concat([display_df[~mask], display_df[mask]], ignore_index=True)
        display_df['Force_N'] = display_df['Force_N'].map('{:20.4f}'.format)

        print(f"\nLigament Peak Forces — {self.subject}")
        print("-" * 45)
        print(display_df.to_string(index=False))
        print("-" * 45)
        
        return summary_df