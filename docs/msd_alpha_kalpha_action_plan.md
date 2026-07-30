# Action Plan: Report \(\alpha\) and \(K_\alpha\) in the MSD Tab

## 1. Objective

Extend the existing MSD analysis so that one calculation reports both:

1. The current normal-diffusion result:

   \[
   MSD(\tau)=2dD\tau+b
   \]

2. An anomalous-diffusion result:

   \[
   MSD(\tau)=2dK_\alpha\tau^\alpha+b
   \]

where:

- \(d=2\) for 2D tracking and \(d=3\) for 3D tracking.
- \(D\) is the normal diffusion coefficient in \(\mathrm{\mu m^2/s}\).
- \(\alpha\) is the dimensionless anomalous exponent.
- \(K_\alpha\) is the generalized diffusion coefficient in
  \(\mathrm{\mu m^2/s^\alpha}\).
- \(b\) is a fitted MSD offset that can absorb part of the localization-error
  and motion-blur contribution.

The existing normal \(D\) calculation must remain available and unchanged in
meaning. The anomalous fit is an additional interpretation of the same
calibrated ensemble MSD curve and selected lag-time interval.

## 2. Scientific interpretation to encode in the software

Use the following interpretation:

| Result | Displayed interpretation |
| --- | --- |
| \(\alpha<1\), with the approximate confidence interval entirely below 1 | Subdiffusive over the fitted lag-time range |
| Approximate confidence interval for \(\alpha\) includes 1 | Compatible with normal diffusion over the fitted lag-time range |
| \(\alpha>1\), with the approximate confidence interval entirely above 1 | Superdiffusive/directed over the fitted lag-time range |
| Fit failed, reached a parameter bound, or has insufficient points | Indeterminate |

The GUI should always say **“over the fitted lag-time range.”** A low
\(\alpha\) does not by itself prove a specific physical mechanism. It can also
result from confinement, localization uncertainty, motion blur, tracking
errors, heterogeneous populations, or fitting too far into the MSD plateau.
The covariance-based interval proposed below is optimistic because MSD values
at different lags are correlated. The classification is therefore a
descriptive, fit-based interpretation rather than a formal hypothesis test.

### Unit requirements

- \(\alpha\) is dimensionless.
- \(K_\alpha\) in physical coordinates is
  \(\mathrm{\mu m^2/s^\alpha}\).
- \(K_\alpha\) in pixel coordinates is
  \(\mathrm{px^2/s^\alpha}\).
- When \(\alpha=1\), \(K_\alpha\) and \(D\) have the same numerical meaning and
  units.
- The conversion is:

  \[
  K_{\alpha,\mathrm{px}} =
  \frac{K_{\alpha,\mathrm{\mu m}}}{(\mathrm{\mu m/px})^2}
  \]

- For anisotropic 3D data, the existing implementation scales Z into
  XY-equivalent pixel coordinates before calling Trackpy. The pixel-space
  result should therefore be labeled **“XY-equivalent px²/s^\(\alpha\)”** in
  3D mode.
- Changing the frame interval must leave \(\alpha\) unchanged but rescale
  \(K_\alpha\) according to the time exponent. If all numerical lag times are
  multiplied by \(c\), the fitted coefficient must change by \(c^{-\alpha}\).

## 3. Required design decisions

### 3.1 Fit both models automatically

Every click on **Calculate MSD** should fit both the normal and anomalous
models over exactly the same lag points. This makes the results comparable and
avoids requiring the user to choose a model before seeing the evidence.

### 3.2 Preserve the existing fit-window control

The existing **Fit lag points** slider remains the single control for both
models.

- Normal linear fit: minimum 2 lag points.
- Anomalous fit: minimum 4 valid lag points because it estimates three
  parameters (\(K_\alpha\), \(\alpha\), and \(b\)).
- Five or more lag points should remain the recommended/default setting.
- If only 2–3 points are available, calculate and display normal \(D\), but show
  the anomalous result as unavailable instead of failing the entire MSD
  calculation.
- The slider tooltip and range message should explicitly state that the fit
  window is shared by both models.

The fit must use the first selected lag points in seconds, as it does now. Do
not silently extend the window when the anomalous fit needs more data.

### 3.3 Fit in the original MSD domain

Fit the nonlinear model against MSD in \(\mathrm{\mu m^2}\), not by performing
a straight-line regression on log-transformed values.

Reasons:

- Log transformation changes the residual weighting.
- Zero or offset-corrected MSD values cannot be logged safely.
- A fitted offset \(b\) is not represented correctly by a simple log-log line.

The existing **Log-Log Scale** checkbox must remain a display option only. It
must never change or recompute the fitted parameters.

### 3.4 Keep the normal result backward compatible

`ParticleMotion.calculate_msd()` currently returns a seven-element tuple that
may be used by notebooks or external code. Do not change that tuple in the
first implementation.

Instead:

- Keep the existing return values and existing attributes such as
  `fit_slope_um2_s`, `fit_intercept_um2`, `fit_r_squared`, and
  `D_std_err_um2_s`.
- Store the new structured fit results on the `ParticleMotion` instance.
- Let `app.py` read the structured results after `calculate_msd()` returns.

This avoids breaking current callers while providing a cleaner path for future
API modernization.

## 4. Changes in `microlive/microscopy.py`

Primary area: `ParticleMotion`, currently beginning near line 6269, especially
`calculate_msd()` near line 6424.

### 4.1 Add a structured fit-result type

Add a small dataclass or equivalent structured result named
`MSDFitResult`. It should contain:

- `model`: `"normal"` or `"anomalous"`.
- `success`: boolean.
- `status_message`: empty on success; reason on failure.
- `dimensions`: 2 or 3.
- `dimension_factor`: 4.0 or 6.0.
- `n_fit_points`.
- `fit_lag_start_s` and `fit_lag_end_s`.
- `parameter_values`.
- `parameter_standard_errors`.
- `parameter_ci95`.
- `offset_um2`.
- `fit_times_s` and `observed_fit_msd_um2`.
- `predicted_fit_msd_um2`.
- `residuals_um2`.
- `rss` and `rmse`.
- `r_squared` for the normal linear model.
- `pseudo_r_squared` for the anomalous nonlinear model.
- `aicc` when it can be calculated.
- `at_parameter_bound`: boolean.
- `weakly_identified`: boolean.
- `warnings`: list of human-readable warnings.

Use named fields rather than an additional positional tuple.
Keep this dataclass analysis-only: it must not contain plotting grids, colors,
line styles, legend strings, or other presentation state. `app.py` should
construct display coordinates by evaluating the stored model parameters over
the measured lag interval.

### 4.2 Isolate model functions

Add private, testable helpers:

- Normal model:

  \[
  f_N(\tau;D,b)=2dD\tau+b
  \]

- Anomalous model:

  \[
  f_A(\tau;K_\alpha,\alpha,b)=2dK_\alpha\tau^\alpha+b
  \]

The helpers must take lag time in seconds and return MSD in
\(\mathrm{\mu m^2}\).

The exact three-parameter function presented to `curve_fit` should be a closure
over the dimension factor:

```python
def anomalous_model_logk(tau_s, log_K_alpha, alpha, offset_um2):
    return (
        dimension_factor
        * np.exp(log_K_alpha)
        * np.power(tau_s, alpha)
        + offset_um2
    )
```

The returned physical parameter is
`K_alpha = np.exp(log_K_alpha)`. `log_K_alpha` is an internal numerical
parameter and must never be displayed or exported as the scientific result.

### 4.3 Add a reusable fitting method

Add a method such as:

`ParticleMotion.fit_msd_models(fit_times_s, fit_msd_um2)`

It should:

1. Convert inputs to finite one-dimensional floating-point arrays.
2. Reject non-positive lag times.
3. Preserve zero MSD values for the raw-domain fit.
4. Fit the normal model using the current linear regression behavior.
5. Fit the anomalous model when at least four usable lag points exist.
6. Return/store one `MSDFitResult` for each model.

This separation allows deterministic unit tests of the fitting logic without
having to construct Trackpy trajectories for every edge case.

### 4.4 Implement a stable anomalous fit

Use `scipy.optimize.curve_fit`, which is already imported by
`microscopy.py`.

Recommended parameterization:

- Optimize `log(K_alpha)` internally so \(K_\alpha\) is always positive.
- Bound \(\alpha\) to the physically interpretable interval
  \(0.05\leq\alpha\leq2.0\).
- Allow the offset \(b\) to be fitted rather than forcing the curve through
  zero.
- Permit a modest negative \(b\), because motion blur can produce a negative
  fitted intercept, but reject a solution if its predicted MSD is non-positive
  across the fitted lag range.

Make the transformed bounds explicit. For each fit, calculate:

```python
msd_scale = max(np.max(np.abs(fit_msd_um2)), np.finfo(float).eps)
time_scale = np.median(fit_times_s)
K_reference = max(
    msd_scale / (dimension_factor * time_scale),
    np.finfo(float).tiny,
)
log_K_reference = np.log(K_reference)
log_K_span = np.log(1e12)

lower_bounds = (
    log_K_reference - log_K_span,
    0.05,
    -2.0 * msd_scale,
)
upper_bounds = (
    log_K_reference + log_K_span,
    2.0,
    2.0 * msd_scale,
)
```

The resulting call should be structurally equivalent to:

```python
curve_fit(
    anomalous_model_logk,
    fit_times_s,
    fit_msd_um2,
    p0=(np.log(K0), alpha0, b0),
    bounds=(lower_bounds, upper_bounds),
    maxfev=20_000,
)
```

The \(K_\alpha\) bounds are deliberately broad numerical guards relative to
the observed MSD/time scale, not biological priors. `K0` should be estimated
from the positive scale of
\((MSD-b_0)/(2d\tau^{\alpha_0})\), clipped inside the transformed bounds, and
`b0` should be the normal-fit intercept clipped inside its bounds.

Use multiple initial values for \(\alpha\), for example 0.5, 1.0, and 1.5, and
retain the converged solution with the smallest residual sum of squares.
Initialize \(K_\alpha\) from the scale of the normal fit and the observed MSD.
This is more reliable than relying on one nonlinear starting point.

Guard against the near-degeneracy between a very small \(\alpha\) and the
offset \(b\). After fitting, calculate:

\[
\text{power span} =
\left(\frac{\tau_{\max}}{\tau_{\min}}\right)^\alpha
\]

If the power span is below 1.10, the power-law component changes by less than
10% across the fitted interval and cannot be reliably separated from the
offset. Retain the numeric result for transparency, but set
`weakly_identified=True`, classify it as indeterminate, and show a warning.
Apply the same treatment when a parameter is at/near a bound or when the
covariance matrix is singular or extremely ill-conditioned.

The method must return an unsuccessful `MSDFitResult`, not raise an exception
that discards the valid normal fit, when:

- There are fewer than four points.
- Optimization does not converge.
- The returned values are non-finite.
- \(K_\alpha\leq0\).
- The predicted MSD is invalid.

If the optimized parameter values are valid but the covariance matrix is not,
retain the fitted \(\alpha\) and \(K_\alpha\), mark their uncertainty as
unavailable, add a warning, and classify the motion as indeterminate.

### 4.5 Uncertainty and fit quality

For the initial implementation:

- Derive approximate parameter standard errors from the nonlinear covariance
  matrix.
- Transform the uncertainty from `log(K_alpha)` back to \(K_\alpha\) using the
  delta method.
- Report approximate 95% confidence intervals using the appropriate Student-t
  multiplier for the residual degrees of freedom when possible.
- Clearly label these intervals as fit-based/approximate because MSD points at
  different lags are correlated.
- Calculate a descriptive nonlinear pseudo-\(R^2\) in the original MSD
  domain:

  \[
  R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}
  \]

- Store and label this value as `pseudo_r_squared`. It can be negative and does
  not have the variance-decomposition interpretation of ordinary linear-model
  \(R^2\).
- Calculate AICc for both fits on the same points. AICc is preferable to the
  pseudo-\(R^2\) for descriptive comparison because the anomalous model has one
  extra parameter. Because lag residuals are correlated, AICc must also be
  described as a heuristic comparison rather than a formal model-selection
  test.
- If there are too few points to calculate AICc, store it as unavailable.

Do not add a synchronous particle-bootstrap calculation to every slider
change. The current GUI recalculates live, and hundreds of bootstrap fits
would make the interface unresponsive. Particle-level bootstrap confidence
intervals can be a later optional analysis or background task.

### 4.6 Classification logic

Add a helper that returns:

- `"Subdiffusive over fitted range"` if the upper approximate 95% CI is below
  1.
- `"Compatible with normal diffusion over fitted range"` if the approximate CI
  includes 1.
- `"Superdiffusive over fitted range"` if the lower approximate 95% CI is above
  1.
- `"Indeterminate"` when uncertainty is unavailable or the fit is unreliable.

If \(\alpha\) lies at or very near either bound, mark the result indeterminate
and display a fit warning. Do the same when `weakly_identified=True`.
The GUI and metadata must describe this as a fit-based classification derived
from an approximate covariance interval.

Do not classify a track as “confined” solely from \(\alpha<1\). A confinement
model has a plateau and is a separate physical model.

### 4.7 Integrate with `calculate_msd()`

After Trackpy produces `em_um2` and the selected fit arrays:

1. Call `fit_msd_models()`.
2. Continue returning the existing normal \(D\), pixel \(D\), MSD series,
   normal fit times, normal plot line, and tracking dataframe.
3. Populate:
   - `self.normal_fit`
   - `self.anomalous_fit`
   - `self.fit_results = {"normal": ..., "anomalous": ...}`
4. Retain the existing normal-fit attributes as aliases to the corresponding
   `normal_fit` values.
5. Add convenience attributes only if needed by legacy GUI code:
   - `self.alpha`
   - `self.alpha_std_err`
   - `self.alpha_ci95`
   - `self.K_alpha_um2_s_alpha`
   - `self.K_alpha_std_err_um2_s_alpha`
   - `self.K_alpha_ci95_um2_s_alpha`
   - `self.K_alpha_px2_s_alpha`
   - `self.anomalous_fit_pseudo_r_squared`
   - `self.anomalous_fit_aicc`
   - `self.motion_classification`

The pixel conversion must only affect \(K_\alpha\), not \(\alpha\).

### 4.8 Standalone `show_plot=True` behavior

Update the optional plot created directly by `ParticleMotion`:

- Plot the MSD data once.
- Plot the normal fit as a white or neutral dashed line.
- Plot the anomalous fit as a contrasting solid line.
- Include \(D\), \(K_\alpha\), \(\alpha\), and each fit's \(R^2\) in a compact
  legend.
- Skip the anomalous curve and show its warning in text/log output if the fit
  is unavailable.

## 5. Changes in `microlive/gui/app.py`

Primary areas:

- MSD state initialization near lines 1605–1623.
- `setup_msd_tab()` near line 16083.
- `reset_msd_tab()` near line 18304.
- `calculate_msd_from_gui()` near line 18372.
- `_calculate_per_trajectory_msd()` near line 18573.
- `plot_msd()` near line 18709.
- `export_msd_dataframe()` near line 18868.
- metadata export near lines 866–908.
- modern slider/channel synchronization near lines 20437–20560.

### 5.1 State additions and invalidation

Initialize and reset all new state everywhere the existing MSD state is
initialized or invalidated:

- `tracking_msd_alpha`
- `tracking_msd_alpha_std_err`
- `tracking_msd_alpha_ci95_low`
- `tracking_msd_alpha_ci95_high`
- `tracking_msd_K_alpha_um2_s_alpha`
- `tracking_msd_K_alpha_px2_s_alpha`
- corresponding \(K_\alpha\) error/CI fields
- `tracking_msd_anomalous_offset_um2`
- `tracking_msd_anomalous_pseudo_r_squared`
- `tracking_msd_normal_aicc`
- `tracking_msd_anomalous_aicc`
- `tracking_msd_motion_classification`
- `tracking_msd_anomalous_fit_equation`
- `tracking_msd_anomalous_fit_status`
- `tracking_msd_display_model`
- `msd_fit_summary`

`reset_msd_tab()` must clear these fields along with the current \(D\) fields.
All existing invalidation events—new movie, crop, retracking, channel removal,
tracking-mode change, or selected tracking-channel change—must continue to
clear both models atomically.

### 5.2 MSD parameter controls

In `setup_msd_tab()`:

1. Update the fit-slider label/tooltip to say:
   - the selected lags are shared by normal and anomalous fits;
   - at least four points are required for \(\alpha\) and \(K_\alpha\);
   - short, early-lag ranges are generally preferred for local diffusion.
2. Keep the default at five supported lag points.
3. Add a **Fit display** combo box:
   - `Both overall fits` — default.
   - `Normal fit with per-cell fits`.
   - `Anomalous fit with per-cell fits`.
4. Connect the display combo to `plot_msd()` only. It must not rerun tracking
   or refit the data.
5. Change the button caption to **Calculate MSD Fits** if the wider label fits
   the current design.

Because the Results panel will become taller, place the right-side controls in
a vertical `QScrollArea` or verify at the application's minimum supported
window height that none of the results are clipped.

### 5.3 Results panel

Split the current Results display into compact subsections.

#### Normal diffusion

- `D = value ± fit SE µm²/s`
- `D = value px²/s`
- `R²`
- `AICc`

#### Anomalous diffusion

- `α = value ± fit SE`
- `Approx. 95% CI = low–high (lag points are correlated)`
- `Kα = value µm²/s^α`
- `Kα = value px²/s^α`
- anomalous `pseudo-R²`
- anomalous `AICc`
- fitted offset \(b\) in \(\mathrm{\mu m^2}\)
- descriptive classification text
- warning/status text when unavailable or at a bound

#### Shared provenance

- 2D/3D mode and equation factor (4 or 6).
- tracking channel.
- number of particles.
- exact fitted lag range in seconds.
- requested and used lag points.
- XY calibration, frame interval, and Z calibration when relevant.

Example display:

```text
Normal: D = 1.13e-03 ± 1.6e-04 µm²/s
Anomalous: α = 0.62 ± 0.07 (approx. 95% CI 0.46–0.78)
Kα = 6.8e-03 µm²/s^0.62
Fit-based interpretation: subdiffusive over 5–25 s
CI caveat: MSD lag points are correlated
```

Use dynamic unit text containing the fitted exponent. Store the numeric value
and exponent in separate fields for export; do not make downstream code parse
the label.

### 5.4 `calculate_msd_from_gui()`

After the existing `motion.calculate_msd()` call:

1. Read `motion.normal_fit` and `motion.anomalous_fit`.
2. Keep all current flat `msd_data` keys for compatibility.
3. Add nested entries:
   - `msd_data["normal_fit"]`
   - `msd_data["anomalous_fit"]`
4. Store the full fit provenance, units, warnings, and classification.
5. Update the new GUI labels.
6. If the anomalous fit failed, keep and display the normal result.
7. Preserve the existing `tracking_msd_summary_method` value for compatibility;
   store the anomalous raw-domain fit equation, status, and metrics in the new
   dedicated metadata fields rather than changing the legacy summary string.

No part of this method should change which image source was tracked. The
existing registration, photobleaching-corrected image, projection mode,
channel filtering, pixel calibration, Z calibration, and movie frame interval
must continue to flow from the tracking data and current movie metadata.

### 5.5 Per-cell fitting

`_calculate_per_trajectory_msd()` currently creates:

- per-trajectory MSD series,
- per-track normal \(D\),
- a pair-weighted ensemble MSD fit for each cell.

Extend each per-cell `ParticleMotion` calculation to store:

- `normal_fit`
- `anomalous_fit`
- cell \(K_\alpha\)
- cell \(\alpha\)
- anomalous fit quality and classification

Use the same:

- dimensionality,
- calibrated spatial units,
- frame interval,
- selected fit lag points,
- Z scaling,
- failure rules

as the overall calculation.

### 5.6 Per-track fitting

Individual-track anomalous fits are much noisier than ensemble fits. They
can also require thousands of nonlinear optimizer calls. Do not calculate
per-track anomalous fits during the initial implementation or during live
fit-slider refreshes.

Keep the existing per-track normal \(D\) calculation unchanged. The first
implementation should report anomalous parameters for:

- the overall pair-weighted ensemble;
- each eligible per-cell ensemble.

If per-track anomalous fitting is added later, expose it only as an explicit
batch/export action. It must run outside the live refresh path, provide progress
and cancellation, require at least five usable lag points per track, and be
benchmarked on realistically large tracking tables before it is enabled by
default. Its export would then include cell/particle IDs, track length,
\(K_\alpha\), \(\alpha\), pseudo-\(R^2\), AICc, fitted lag range, validity, and
rejection reason.

Do not replace any invalid fit value with zero. Count rejected ensemble fits
and preserve a human-readable failure reason.

The main GUI result should continue to emphasize the pair-weighted ensemble
fit. Per-track means and distributions should be labeled as separate
descriptive summaries so users do not confuse them with the ensemble
estimator.

## 6. Plotting changes

### 6.1 Default view

Keep the existing MSD data and per-cell colors. Add:

- A lightly shaded vertical region covering the fitted lag-time interval.
- Overall normal fit: white dashed line.
- Overall anomalous fit: magenta/purple solid line.
- Legend text:
  - `Normal: D=... µm²/s, R²=...`
  - `Anomalous: Kα=..., α=..., pseudo-R²=...`

Do not draw the anomalous curve outside the measured lag range. Extrapolating a
power law beyond the data can be visually misleading.

### 6.2 Avoid an unreadable per-cell plot

The default **Both overall fits** mode should:

- show all eligible per-cell MSD curves/error bars;
- show both overall model fits;
- omit per-cell fit lines.

The two model-specific display modes should:

- show the selected overall fit;
- show matching per-cell fit lines;
- show compact per-cell statistics for that model.

This prevents normal and anomalous fit lines for every cell from doubling the
visual clutter.

### 6.3 Log-log view

When **Log-Log Scale** is enabled:

- reuse the parameters fitted in the original MSD domain;
- filter non-positive plotted coordinates only for rendering;
- do not recompute \(\alpha\) from a log-log line;
- keep the same fit-range shading and legend.

### 6.4 Status and warnings

Show a small plot annotation when:

- fewer than four points were available for the anomalous fit;
- \(\alpha\) is at a fit bound;
- the anomalous covariance/CI is unavailable;
- the fitted power span is too small to distinguish \(\alpha\) from the offset;
- the selected lag range extends into a visible plateau;
- too few trajectories support late lags.

The last two conditions should be warnings, not automatic deletion of data.

## 7. Data and metadata export

### 7.1 Preserve the raw MSD curve export

Keep the existing wide per-trajectory MSD CSV schema so existing analysis
scripts continue to work. Rename the GUI button to **Export MSD Curves** only
if this will not disrupt established user documentation.

### 7.2 Add a fit-summary CSV

Add an **Export Fit Summary** action. Create a long-format CSV with one row per
available scope/model. In the initial implementation this means overall and
per-cell rows for both models. The existing wide per-trajectory MSD export
continues to carry the per-track curves; do not imply that per-track anomalous
fit rows exist unless the optional batch implementation has actually been
added.

- `scope`: overall, cell, or trajectory.
- `channel`.
- `cell_id`.
- `particle_id`.
- `model`.
- `dimensions`.
- `n_particles`.
- `track_length_frames` where applicable.
- `fit_points_requested`.
- `fit_points_used`.
- `fit_lag_start_s`.
- `fit_lag_end_s`.
- `D_um2_s`, `D_se_um2_s`.
- `D_px2_s`, `D_se_px2_s`.
- `K_alpha_um2_s_alpha`, `K_alpha_se_um2_s_alpha`.
- `K_alpha_px2_s_alpha`, `K_alpha_se_px2_s_alpha`.
- `alpha`, `alpha_se`, `alpha_ci95_low`, `alpha_ci95_high`.
- `offset_um2`.
- `r_squared` for the normal linear fit.
- `pseudo_r_squared` for the anomalous nonlinear fit.
- `rss`, `rmse`, `aicc`.
- `classification`.
- `fit_valid`.
- `fit_status`.
- `xy_um_per_px`.
- `z_um_per_px`.
- `frame_interval_s`.
- explicit unit strings.

Keep numeric values numeric. Do not export combined strings such as
`"0.62 ± 0.07"`.

### 7.3 Experiment metadata export

Extend the existing **MSD Results** metadata section to include:

- both model equations;
- \(D\) and its uncertainty;
- \(\alpha\), SE, and confidence interval;
- an explicit warning that the covariance-based interval is approximate
  because lag points are correlated;
- \(K_\alpha\), uncertainty, and explicit units;
- normal and anomalous offsets;
- normal \(R^2\), anomalous pseudo-\(R^2\), and both AICc values;
- fitted lag range;
- classification;
- fit warnings/status;
- the fact that the nonlinear fit was performed in the raw MSD domain;
- selected display mode, clearly marked as a display choice rather than an
  analysis choice.

## 8. Tests

### 8.1 `tests/test_particle_motion_units.py`

Add tests for:

1. **Normal Brownian tracks**
   - Recover \(\alpha\approx1\).
   - Recover \(K_\alpha\approx D\).
   - Confirm both use the same fit window.

2. **Exact synthetic anomalous MSD**
   - Fit generated values from known \(K_\alpha\), \(\alpha\), and \(b\).
   - Test at least one subdiffusive and one superdiffusive case.

3. **2D versus 3D**
   - Confirm factors of 4 and 6.

4. **Spatial calibration**
   - Doubling pixel size multiplies physical \(K_\alpha\) by four.
   - Pixel-space \(K_\alpha\) remains unchanged.
   - \(\alpha\) remains unchanged.

5. **Time calibration**
   - Use a noiseless synthetic MSD curve generated from known
     \(K_\alpha\), \(\alpha\), and \(b\).
   - Refit the same MSD values once with \(\tau\) and once with
     \(c\tau\).
   - Assert, within tight numerical tolerance, that \(\alpha\) is unchanged and
     \(K_\alpha\) changes by \(c^{-\alpha}\).
   - Keep any stochastic Brownian-track timing test as a separate
     approximate integration test, not as the exact identity test. The scaling
     identity is mathematically valid for a fixed noisy MSD curve as well, but
     noiseless input isolates the unit conversion from optimizer variability.

6. **Insufficient lag points**
   - Normal result succeeds with two points.
   - Anomalous result is unavailable with a clear status.
   - No GUI-breaking exception is raised.

7. **Numerical edge cases**
   - NaN/inf values.
   - Constant MSD.
   - Non-convergence.
   - Parameter-bound solutions.
   - Near-degenerate low-\(\alpha\) solutions with power span below 1.10.
   - Negative fitted offset with positive predicted MSD.

### 8.2 `tests/test_gui_design_regressions.py`

Add tests for:

- New labels and fit-display combo exist.
- Default display mode is both overall fits.
- Slider changes invalidate and recompute both results.
- Display-mode changes replot without recalculating.
- Results survive ordinary tab navigation.
- Results clear after crop, retracking, mode change, channel deletion, or new
  image load.
- A 2–3-lag dataset displays normal \(D\) and an anomalous-fit warning.
- A valid calculation records \(\alpha\), \(K_\alpha\), exact fit range, and
  classification.
- The GUI marks covariance intervals as approximate and notes lag correlation.
- Weakly identified low-\(\alpha\) fits are shown as indeterminate.
- Log-log toggling does not change stored fitted parameters.
- Metadata includes the new values and units.
- Fit-summary CSV fields are numeric and complete.

### 8.3 Plot tests

Use lightweight Matplotlib assertions to verify:

- normal and anomalous overall fit artists are present;
- both use the selected lag window;
- anomalous curves are not extrapolated past measured data;
- log-log mode filters invalid display points without refitting;
- the plotted legend values match `msd_data`.
- nonlinear goodness of fit is labeled `pseudo-R²`, not ordinary `R²`.

## 9. Documentation updates after implementation

Update:

- `docs/tutorial.md`
- `docs/api_reference.md`
- `docs/user_guide.md` if it contains an MSD workflow section

The documentation should explain:

- the difference between \(D\) and \(K_\alpha\);
- why \(K_\alpha\)'s units depend on \(\alpha\);
- how to interpret \(\alpha\);
- why the fit window changes the result;
- why subdiffusive behavior is not automatically proof of confinement;
- why the log-log checkbox changes only the display;
- how to export the fit summary.

Correct the current tutorial statement that more lag points are useful for
confined/directed motion “detection.” Adding long plateau regions to a linear
fit can instead bias \(D\) downward. The tutorial should recommend inspecting
several justified early-lag windows and reporting the chosen interval.

## 10. Implementation sequence

1. Add the model functions and structured fit result in `microscopy.py`.
2. Add deterministic unit tests for known MSD curves.
3. Integrate both fits into `ParticleMotion.calculate_msd()` while preserving
   its return tuple.
4. Add physical-unit and frame-interval regression tests.
5. Extend `app.py` state initialization and reset/invalidation.
6. Add the GUI results and display controls.
7. Store the structured overall fit results in `msd_data`.
8. Extend per-cell anomalous fitting while retaining the existing per-track
   normal \(D\) calculation; defer per-track anomalous fitting.
9. Update plotting and log-log behavior.
10. Add the fit-summary and metadata exports.
11. Add GUI regression tests.
12. Update user documentation and perform a manual GUI check with:
    - a normal Brownian dataset;
    - a visibly subdiffusive dataset;
    - a short dataset with fewer than four fit lags;
    - both 2D maximum-projection and anisotropic 3D tracking data.

## 11. Acceptance criteria

The work is complete when:

- The existing normal \(D\) result remains numerically backward compatible.
- A valid MSD calculation also reports \(\alpha\) and \(K_\alpha\).
- \(K_\alpha\) has correct exponent-dependent physical and pixel units.
- Both models use the same calibrated lag times and selected fit window.
- Changing the movie frame interval produces the mathematically expected
  change in \(K_\alpha\), without changing \(\alpha\).
- 2D and 3D use factors 4 and 6 respectively.
- Failed or underdetermined anomalous fits do not remove a valid normal result.
- The GUI labels the interpretation as specific to the fitted lag range.
- The GUI labels covariance confidence intervals as approximate and anomalous
  goodness of fit as pseudo-\(R^2\).
- Low-\(\alpha\) fits that are nearly degenerate with the offset are reported
  as weakly identified/indeterminate.
- Log-log display does not change the calculation.
- Overall and per-cell anomalous results remain distinguishable from the
  existing per-track normal \(D\) summary.
- Plot, CSV, and metadata values agree.
- All current MSD regression tests and new anomalous-fit tests pass.

## 12. Explicitly out of scope for this change

- Replacing Trackpy tracking or changing trajectory linking.
- Changing registration, photobleaching correction, image-source selection, or
  channel-selection behavior.
- Automatically declaring a physical confinement mechanism from
  \(\alpha<1\).
- Adding a confined-diffusion plateau model or directed-motion velocity model.
- Running computationally expensive bootstrap confidence intervals on every
  live slider update.
- Running per-track anomalous nonlinear fits during live GUI recalculation.
- Removing the current normal \(D\) calculation.
