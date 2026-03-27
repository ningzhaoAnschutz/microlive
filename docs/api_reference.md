# API Reference — MicroLive

## Scope

Aligned to the current source tree. Only public symbols are listed (names starting with `_` are omitted).
Full constructor signatures are shown for classes; method listings are grouped by purpose.

## Primary Entry Points

| Symbol | Description |
| --- | --- |
| `microlive.gui.main.main()` | Launch the GUI |
| `microlive.microscopy` | Core analysis engine |
| `microlive.ml_spot_detection` | CNN-based spot classification |
| `microlive.pipelines.*` | End-to-end batch pipelines |
| `microlive.utils` | Resource and device utilities |

---

## `microlive` — `microlive/__init__.py`

Package root. No public top-level functions or classes.

---

## `microlive.gui.main` — `microlive/gui/main.py`

Entry point for the GUI application.

**Functions:** `main()`

---

## `microlive.gui.app` — `microlive/gui/app.py`

Main PyQt5 GUI module. Houses the `GUI` application window and supporting classes.

**Functions:** `configure_logging_and_styles()`

---

### `Plots` *(gui)*

Helper for rendering correlation/colocalization figures inside the GUI canvas.

```python
__init__(gui)
```

| Method | Description |
| --- | --- |
| `plot_autocorrelation(...)` | Render ACF curve with optional fit overlay |
| `plot_crosscorrelation(...)` | Render cross-correlation curve |
| `plot_matrix_pair_crops(...)` | Render paired crop matrix for colocalization |

---

### `Metadata`

Stores and serializes experiment metadata.

```python
__init__(**kwargs)
```

| Method | Description |
| --- | --- |
| `write_metadata()` | Serialize metadata to disk |

---

### `GUI`

Top-level application window. Manages all tabs and inter-tab state.

```python
__init__(icon_path)
```

**Properties:** `active_mask()`

#### File I/O & Display

| Method | Description |
| --- | --- |
| `open_image()` | Open file dialog and load image |
| `load_tif_image(file_path)` | Load TIFF stack |
| `load_lif_image(file_path, image_index)` | Load Leica LIF scene |
| `convert_to_standard_format(image_stack)` | Normalize axis order |
| `initUI()` | Build main window layout |
| `applyTheme(useDarkTheme)` | Switch light/dark theme |
| `ask_for_metadata_from_user(missing_fields)` | Prompt for missing metadata |
| `open_dimension_mapping_dialog(file_shape)` | Map unknown axis order |
| `setup_display_tab()` | Build display tab widgets |
| `reset_display_tab()` | Restore display tab defaults |
| `control_panel_image_properties(parent_layout)` | Add image property controls |
| `create_channel_visualization_controls(channel_index, initial_params)` | Per-channel brightness/contrast sliders |
| `create_channel_buttons()` | Channel selector button bar |
| `set_display_controls_enabled(enabled)` | Enable/disable display controls |
| `merge_color_channels()` | Merge selected channels into RGB |
| `compute_merged_image(use_brightness_slider)` | Compute composite image array |
| `plot_image()` | Render current frame to canvas |
| `update_frame(value)` | Seek to time frame |
| `update_channel(channel)` | Switch active channel |
| `update_z(value)` | Seek to z-plane (display) |
| `on_channel_tab_changed(index)` | Handle channel tab switch |
| `on_tree_item_clicked(item, column)` | Handle file-tree click |
| `on_tree_current_item_changed(current, previous)` | Handle file-tree selection change |
| `onChannelParamsChanged(channel, params)` | Handle LUT parameter change |
| `play_pause()` | Toggle global playback |
| `play_pause_display()` | Toggle display-tab playback |
| `stop_all_playback()` | Stop all running playback timers |
| `next_frame()` | Advance one frame (global) |
| `next_frame_display()` | Advance one frame (display tab) |
| `close_selected_file()` | Remove selected file from session |
| `close_all_files()` | Clear all loaded files |

#### Registration

| Method | Description |
| --- | --- |
| `setup_registration_tab()` | Build registration tab widgets |
| `reset_registration_tab()` | Restore registration tab defaults |
| `on_registration_mode_changed(mode)` | Switch rigid/affine/translation mode |
| `on_registration_time_changed(value)` | Update reference time point |
| `perform_registration()` | Run image registration |
| `remove_registration()` | Undo applied registration |
| `reset_registration_state()` | Clear registration state |
| `update_registration_channel(idx)` | Switch registration channel |
| `toggle_playback_registration()` | Toggle registration-tab playback |
| `registration_next_frame()` | Advance one frame (registration tab) |
| `plot_registration_panels()` | Render before/after panels |
| `on_reg_mouse_press/move/release(event)` | ROI rubber-band interaction |

#### Segmentation

| Method | Description |
| --- | --- |
| `setup_segmentation_tab()` | Build segmentation tab widgets |
| `reset_segmentation_tab()` | Restore segmentation tab defaults |
| `run_watershed_segmentation()` | Run watershed on current frame |
| `run_cellpose_cyto()` | Run Cellpose cytoplasm model |
| `run_cellpose_nuc()` | Run Cellpose nucleus model |
| `synchronize_and_plot_cellpose()` | Align cyto/nuc masks and display |
| `clear_cellpose_masks()` | Discard Cellpose masks |
| `clear_imported_masks()` | Discard imported masks |
| `import_mask_from_tiff(mask_type)` | Load external TIFF mask |
| `plot_cellpose_results()` | Render Cellpose overlay |
| `plot_segmentation()` | Render segmentation overlay |
| `update_watershed_threshold_factor(value)` | Adjust watershed threshold |
| `update_segmentation_source(state)` | Switch max-proj / z-slice source |
| `compute_max_proj_segmentation()` | Run segmentation on max projection |
| `on_segmentation_z_changed(value)` | Seek to z-plane (segmentation) |
| `reset_segmentation_z_slider()` | Reset z-slider to 0 |
| `create_segmentation_channel_buttons()` | Segmentation channel selector |
| `update_segmentation_channel(channel_index)` | Switch segmentation channel |
| `update_segmentation_frame(value)` | Seek to frame (segmentation) |
| `manual_segmentation()` | Enter polygon draw mode |
| `on_polygon_click(event)` / `finish_manual_polygon()` / `clear_manual_mask()` | Polygon tool interactions |
| `enter_edit_mode()` / `exit_edit_mode()` | Enter/leave mask edit mode |
| `on_edit_mask_selector_changed(index)` | Switch mask layer to edit |
| `undo_edit()` / `reset_edits()` / `apply_and_save_edits()` | Edit history controls |
| `plot_edit_mode()` | Render mask edit overlay |
| `on_remove_border_cells_changed(state)` | Toggle border-cell removal |
| `on_remove_unpaired_cells_changed(state)` | Toggle unpaired-cell removal |
| `on_keep_center_cell_changed(state)` | Toggle keep-center-cell mode |
| `get_border_touching_labels(masks)` | Return labels touching image border |
| `remove_labels_and_reindex(masks, labels_to_remove)` | Delete labels and reindex |
| `reindex_masks(masks)` | Renumber mask labels sequentially |
| `get_closest_cell_to_center(mask)` | Find label nearest image center |

#### Photobleaching

| Method | Description |
| --- | --- |
| `setup_photobleaching_tab()` | Build photobleaching tab widgets |
| `reset_photobleaching_tab()` | Restore photobleaching tab defaults |
| `compute_photobleaching()` | Fit and apply photobleaching correction |
| `plot_photobleaching()` | Render decay and correction curves |

#### Tracking & Spot Detection

| Method | Description |
| --- | --- |
| `setup_tracking_tab()` | Build tracking tab widgets |
| `reset_tracking_tab()` | Restore tracking tab defaults |
| `detect_spots(image, threshold, list_voxels, masks_complete_cells, ...)` | Detect spots in a single frame |
| `detect_spots_in_current_frame()` | Detect spots at current time point |
| `detect_spots_all_frames()` | Detect spots across all frames |
| `perform_particle_tracking()` | Link spots into trajectories |
| `track_particles(corrected_image, masks_*, parameters, use_maximum_projection)` | Core tracking call |
| `on_tracking_finished(list_dataframes_trajectories)` | Handle tracking completion |
| `on_tracking_finished_with_progress(...)` | Handle tracking completion with progress dialog |
| `clear_channel_tracking(channel)` | Clear trajectories for one channel |
| `clear_all_tracking()` | Clear all trajectory data |
| `plot_tracking()` | Render spot/trajectory overlay |
| `scale_spots()` | Rescale spot display size |
| `get_current_image_source()` | Return active image array |
| `get_tracking_image_source()` | Return image used for tracking |
| `update_threshold_histogram()` | Refresh threshold histogram |
| `update_threshold_value(value)` | Update spot detection threshold |
| `on_auto_threshold_clicked()` | Run auto-threshold |
| `on_image_source_changed()` | Handle image-source combo change |
| `on_tab_change(index)` | Handle main tab switch |
| `update_min_length_trajectory(value)` | Set minimum trajectory length |
| `update_yx_spot_size(value)` / `update_z_spot_size(value)` | Set spot detection radius |
| `update_cluster_radius(value)` / `update_max_spots_cluster(value)` | Cluster detection parameters |
| `update_use_maximum_projection(state)` | Toggle max-projection mode |
| `update_max_range_search_pixels(value)` | Set particle linking search radius |
| `update_memory(value)` | Set linking memory frames |
| `update_use_fixed_size_intensity(state)` | Toggle fixed-aperture intensity |
| `update_fast_gaussian_fit(state)` | Toggle fast Gaussian fit |
| `update_tracking_sliders()` | Sync all tracking sliders to state |
| `generate_random_spots(state)` | Toggle random-spot simulation |
| `update_use_fixed_threshold(checked)` | Toggle fixed vs. auto threshold |
| `select_tracking_vis_channel(channel_idx)` | Switch tracking-vis channel |
| `merge_tracking_visualization()` | Merge multi-channel tracking views |

#### MSD

| Method | Description |
| --- | --- |
| `setup_msd_tab()` | Build MSD tab widgets |
| `reset_msd_tab()` | Restore MSD tab defaults |
| `calculate_msd_from_gui()` | Compute MSD from loaded trajectories |
| `plot_msd()` | Render MSD curve |
| `export_msd_dataframe()` | Save MSD results to CSV |
| `export_msd_plot()` | Save MSD plot to file |

#### Distributions & Time Course

| Method | Description |
| --- | --- |
| `setup_distributions_tab()` | Build distributions tab |
| `reset_distribution_tab()` | Restore distribution tab defaults |
| `plot_distribution()` | Render intensity distribution |
| `plot_intensity_histogram()` | Plot per-channel intensity histogram |
| `on_data_type_changed(new_data_type)` | Switch distribution data type |
| `setup_time_course_tab()` | Build time-course tab |
| `reset_time_course_tab()` | Restore time-course tab defaults |
| `plot_intensity_time_course()` | Render mean-intensity time course |

#### Correlation

| Method | Description |
| --- | --- |
| `setup_correlation_tab()` | Build correlation tab |
| `reset_correlation_tab()` | Restore correlation tab defaults |
| `compute_correlations()` | Run ACF/CCF computation |
| `display_correlation_plot()` | Render correlation result |
| `create_correlation_channel_checkboxes()` | Build channel selection checkboxes |
| `on_channel_selection_changed()` | Handle channel checkbox toggle |
| `update_fit_type()` | Switch exponential/linear fit |
| `on_correlation_percentile_changed()` | Update trajectory percentile filter |
| `update_snr_threshold_for_acf(value)` | Set SNR filter for ACF |
| `update_correct_baseline(state)` | Toggle baseline correction |
| `update_remove_outliers(state)` | Toggle outlier removal |
| `update_field_name(text)` | Set correlation data field |
| `update_min_percentage_data_in_trajectory(value)` | Set minimum data density |
| `update_de_correlation_threshold(value)` | Set de-correlation threshold |
| `update_max_lag(value)` | Set maximum lag time |
| `update_multi_tau(state)` | Toggle multi-tau mode |

#### Colocalization

| Method | Description |
| --- | --- |
| `setup_colocalization_tab()` | Build colocalization tab |
| `setup_coloc_visual_subtab()` / `setup_coloc_distance_subtab()` | Visual / distance sub-tabs |
| `setup_coloc_verify_visual_subtab()` / `setup_coloc_verify_distance_subtab()` | Verification sub-tabs |
| `reset_colocalization_tab()` / `reset_manual_colocalization()` | Restore colocalization state |
| `compute_colocalization()` | Run colocalization analysis |
| `display_colocalization_results(...)` | Render crop matrix result |
| `display_colocalization_plot()` | Render summary plot |
| `extract_colocalization_data(save_df)` | Export colocalization CSV |
| `extract_manual_colocalization_data(save_df)` | Export manual colocalization CSV |
| `on_colocalization_hover/leave(event)` | Hover tooltip interaction |
| `update_colocalization_method()` | Switch colocalization algorithm |
| `populate_colocalization_channels()` | Refresh channel combo boxes |
| `on_colocalization_tracking_channel_changed(index)` | Handle channel combo change |
| `populate_distance_channel_combos()` | Refresh distance-mode combos |
| `run_distance_colocalization()` | Run distance-based colocalization |
| `export_distance_colocalization_data/image()` | Export distance results |
| `display_distance_colocalization()` | Render distance overlay |
| `toggle_distance_playback()` / `advance_distance_frame()` / `on_distance_frame_changed(value)` | Playback controls |
| `populate_verify_visual/distance()` | Load verification data |
| `cleanup_verify_visual/distance()` | Clear verification state |
| `export_verify_visual/distance_data()` | Export verification results |
| `update_z_dist_coloc(value)` / `reset_dist_coloc_z_slider()` | Z-slider controls |
| `update_distance_nm_label()` | Refresh distance label |

#### Tracking Visualization

| Method | Description |
| --- | --- |
| `setup_tracking_visualization_tab()` | Build tracking-vis tab |
| `reset_tracking_visualization_tab()` | Restore tracking-vis state |
| `display_tracking_visualization(selected_channelIndex, spot_coord)` | Render cell-zoom + time course |
| `play_pause_tracking()` / `play_pause_tracking_vis()` | Playback toggles |
| `next_frame_tracking()` / `next_frame_tracking_vis()` | Frame advance |
| `update_z_tracking(value)` | Seek to z-plane (tracking) |

#### Export

| Method | Description |
| --- | --- |
| `setup_export_tab()` | Build export tab |
| `export_selected_items()` | Export all checked items |
| `select_all_exports()` / `deselect_all_exports()` | Bulk selection |
| `get_default_export_filename(prefix, extension)` | Generate timestamped filename |
| `on_comments_combo_changed(index)` | Handle comment combo change |
| `reset_export_comment()` | Clear export comment |
| `export_displayed_image_as_png()` | Save current canvas frame |
| `export_displayed_video()` / `export_tracking_video()` | Save video MP4 |
| `export_time_course_image()` | Save time-course plot |
| `export_tracking_image()` / `export_tracking_data()` | Save tracking overlay / CSV |
| `export_segmentation_image()` / `export_mask_as_tiff()` | Save segmentation outputs |
| `export_intensity_image()` | Save intensity map |
| `export_correlation_image()` | Save correlation plot |
| `export_colocalization_image()` | Save colocalization plot |
| `export_tracking_visualization_image()` / `export_tracking_visualization_video()` | Save tracking-vis outputs |

#### Global Reset

| Method | Description |
| --- | --- |
| `reset_all_state()` | Full application state reset |
| `reset_cellpose_tab()` | Reset Cellpose parameters |

---

## `microlive.microscopy` — `microlive/microscopy.py`

Core analysis engine. Import via `from microlive import microscopy as mi`.

---

### `PatchMPSFloat64`

Context manager that monkeypatches `torch.zeros` for MPS float64 compatibility.

```python
__init__()
```

---

### `Banner`

Prints a formatted banner to stdout.

```python
__init__(text=None, image=None, show=True, padding=5)
```

**Methods:** `print_banner()`

---

### `Photobleaching`

Fit and apply exponential photobleaching correction to time-lapse images.

```python
__init__(image_TZYXC, mask_YX=None, show_plot=True, mode='inside_cell',
         precalulated_list_decay_rates=None, plot_name=None, radius=50,
         time_interval_seconds=None, min_intensity_threshold=10, verbose=False)
```

**Methods:** `calculate_photobleaching()`, `apply_photobleaching_correction()`

---

### `AutoThreshold`

Elbow-method automatic threshold selection for spot detection.

```python
__init__(image, voxel_size_yx=130, voxel_size_z=300,
         yx_spot_size_in_px=5, z_spot_size_in_px=2, use_3d=None)
```

**Methods:** `calculate()`, `get_elbow_data()`

---

### `ReadLif`

Read Leica `.lif` files and extract images with physical metadata.

```python
__init__(path, show_metadata=True, save_tif=False, save_png=False,
         format='TZYXC', lazy=False)
```

**Methods:** `read()`, `read_scene(image_index)`, `get_laser_info(image_index)`

---

### `ReadCzi`

Read Zeiss `.czi` files and extract images with physical metadata. Automatically detects and removes Apotome structured-illumination grid artifacts by averaging H-phases when raw (unprocessed) Apotome data is present.

```python
__init__(path, show_metadata=True, format='TZYXC')
```

| Method | Description |
| --- | --- |
| `read()` | Read all scenes and return 11-element tuple matching `ReadLif` format |
| `read_scene(scene_index)` | Read a single scene as a 5-D array (TZYXC) |

**Apotome Handling:** When `IsOnlineProcessing=false` and an H dimension is present, the reader automatically averages all phase-shifted images to produce a clean widefield-equivalent result. Properly processed CZI files pass through unchanged.

---

### `ConvertFormat`

Transpose image arrays between axis orderings.

```python
__init__(image, original_order, desired_order)
```

**Methods:** `convert()`

---

### `GaussianFilter`

Apply Gaussian smoothing to a time-lapse video.

```python
__init__(video, sigma=1)
```

**Methods:** `apply_filter()`

---

### `Intensity`

Calculate spot intensities via disk-doughnut background subtraction or Gaussian fit.

```python
__init__(original_image, spot_size=5, array_spot_location_z_y_x=None,
         use_max_projection=False, optimize_spot_size=False,
         allow_subpixel_repositioning=False, fast_gaussian_fit=True,
         snr_method='peak')
```

**Methods:** `calculate_intensity()`, `fit_2D_gaussian(data)`,
`two_dimensional_gaussian(xy, ...)`, `optimize_spot_size_method(...)`,
`search_best_center(...)`

---

### `RemoveExtrema`

Clip image intensity at percentile thresholds.

```python
__init__(image, min_percentile=1, max_percentile=99, selected_channels=None)
```

**Methods:** `remove_outliers()`

---

### `Cellpose`

Single-frame cell segmentation using Cellpose deep-learning models.

```python
__init__(image, num_iterations=3, channels=None, diameter=120,
         model_type='cyto3', selection_method='max_cells_and_area',
         NUMBER_OF_CORES=1, pretrained_model=None, selection_metric='max_cells')
```

**Methods:** `calculate_masks()`

---

### `CellposeTimeSeries`

Cellpose segmentation across multiple time points with consistent cell-ID tracking.

```python
__init__(image, channels_cytosol=None, channels_nucleus=None,
         diameter_cytosol=120, diameter_nucleus=60, max_timepoints=10,
         linking_memory=5, min_iou_threshold=0.3, model_type_cyto='cyto3',
         model_type_nuc='nuclei', use_memmap=False, progress_callback=None,
         selection_metric_cyto=None, selection_metric_nuc=None)
```

**Methods:** `calculate_tyx_masks()`, `filter_short_lived_masks(masks_tyx, min_frames=2)`

---

### `CellSegmentationWatershed`

Gradient-based watershed segmentation (production variant).

```python
__init__(image, footprint_size=3, expected_radius=200, threshold_method='li',
         threshold_factor=1.0, markers_method='local', canny_sigma=2.0,
         min_object_size=500, separation_size=1)
```

**Methods:** `apply_watershed()`

---

### `CellSegmentationWatershed_standard`

Standard watershed segmentation variant.

```python
__init__(image, footprint_size=5, threshold_method='li',
         markers_method='distance', separation_size=5, threshold_factor=1.0)
```

**Methods:** `apply_watershed()`

---

### `CellSegmentation`

Dual-compartment (cytosol + nucleus) segmentation using Cellpose.

```python
__init__(image, channels_cytosol=None, channels_nucleus=None,
         diameter_cytosol=150, diameter_nucleus=100,
         optimization_segmentation_method='default',
         remove_fragmented_cells=False, show_plot=True, image_name=None,
         NUMBER_OF_CORES=1, running_in_pipeline=False,
         model_nuc_segmentation='nuclei', model_cyto_segmentation='cyto3',
         pretrained_model_nuc_segmentation=None,
         pretrained_model_cyto_segmentation=None,
         selection_metric='max_cells_and_area', num_iterations=5)
```

**Methods:** `calculate_masks()`, `synchronize_masks(masks_cyto, masks_nuclei)`,
`synchronize_masks_tyx(...)`, `is_nucleus_in_cytosol(mask_n, mask_c)`

---

### `ManualSegmentation`

Interactive polygon-drawing segmentation tool (Jupyter/inline).

```python
__init__(image, cmap='Spectral', polygon_color=(255, 0, 0))
```

**Methods:** `polygon(new_image, points_in_polygon)`, `switch_to_inline()`,
`onclick(event)`, `close_and_save(filename, save_mask)`

---

### `MultiManualSegmentation`

Multi-class interactive segmentation.

```python
__init__(image, cmap='Greys_r')
```

**Methods:** `get_mask()`, `get_class_map()`, `get_mask_for_class(class_name)`

---

### `LineProfile`

Interactive two-point line-profile tool.

```python
__init__(image, ax=None, cmap='gray', max_pixels=None)
```

**Methods:** `get_profile()`

---

### `TrackPyDetection`

Spot detection using TrackPy.

```python
__init__(image, channels_spots, voxel_size_yx=150, yx_spot_size_in_px=5,
         show_plot=False, image_name=None, save_all_images=False,
         spot_diameter=5, display_spots_on_multiple_z_planes=False,
         use_max_projection=True, threshold_for_spot_detection=None,
         save_files=False, reference_threshold=None)
```

**Methods:** `detect()`

---

### `BigFISH`

Spot detection using the Big-FISH library.

```python
__init__(image, channels_spots, voxel_size_z=300, voxel_size_yx=103,
         cluster_radius_nm=350, yx_spot_size_in_px=5, z_spot_size_in_px=2,
         show_plot=False, image_name=None, save_all_images=False,
         display_spots_on_multiple_z_planes=False,
         use_log_filter_for_spot_detection=True,
         threshold_for_spot_detection=None, save_files=False,
         decompose_alpha=0.3, decompose_beta=2, decompose_gamma=5,
         decompose_dense_regions=False, reference_threshold=None)
```

**Methods:** `detect()`

---

### `SpotDetection`

Multi-channel spot detection orchestrator (BigFISH or TrackPy backend).

```python
__init__(image, channels_spots, channels_cytosol, channels_nucleus,
         cluster_radius_nm=500, masks_complete_cells=None, masks_nuclei=None,
         masks_cytosol_no_nuclei=None, dataframe=None, image_counter=0,
         list_voxels=None, show_plot=True, image_name=None,
         save_all_images=True, display_spots_on_multiple_z_planes=False,
         use_log_filter_for_spot_detection=True,
         threshold_for_spot_detection=None, save_files=True,
         yx_spot_size_in_px=None, z_spot_size_in_px=None,
         use_trackpy=False, use_maximum_projection=False,
         calculate_intensity=True, use_fixed_size_for_intensity_calculation=True,
         fast_gaussian_fit=True, reference_threshold=None)
```

**Methods:** `get_dataframe()`

---

### `ParticleTracking`

Multi-frame spot detection and particle linking for time-lapse imaging.

```python
__init__(image, channels_spots, list_voxels, channels_cytosol,
         channels_nucleus, remove_clusters=False, maximum_spots_cluster=None,
         min_length_trajectory=10, threshold_for_spot_detection=100,
         masks=None, masks_nuclei=None, masks_cytosol_no_nuclei=None,
         memory=0, yx_spot_size_in_px=5, z_spot_size_in_px=2,
         cluster_radius_nm=None, link_particles=True, use_trackpy=False,
         use_fixed_size_for_intensity_calculation=True, number_cores=None,
         use_maximum_projection=False, separate_clusters_and_spots=False,
         maximum_range_search_pixels=10, link_using_3d_coordinates=False,
         neighbor_strategy='KDTree', generate_random_particles=False,
         number_of_random_particles_trajectories=None, step_size_in_sec=1.0,
         fast_gaussian_fit=True, verbose=False, use_fixed_threshold=False)
```

**Methods:** `run()`

---

### `Registration`

Image registration using StackReg.

```python
__init__(image, roi_bounds, reference_channel=0, mode='RIGID_BODY',
         padding=10, progress_callback=None, verbose=False)
```

**Methods:** `register()`, `get_registered_image()`

---

### `DataProcessing`

Convert spot detection results to a per-cell DataFrame.

```python
__init__(clusters_and_spots, image, masks_complete_cells, masks_nuclei,
         masks_cytosol_no_nuclei, channels_cytosol, channels_nucleus,
         yx_spot_size_in_px, spot_type=0, dataframe=None,
         reset_cell_counter=False, image_counter=0,
         number_color_channels=None, use_maximum_projection=False,
         use_fixed_size_for_intensity_calculation=True, fast_gaussian_fit=True)
```

**Methods:** `get_dataframe()`

---

### `ParticleMotion`

Compute MSD and diffusion coefficients from trajectories.

```python
__init__(trackpy_dataframe, microns_per_pixel=1, step_size_in_sec=1,
         max_lagtime=100, show_plot=True, remove_drift=False, spot_type=0,
         plot_name=None, max_fit_points=20, is_3d=False,
         microns_per_pixel_z=None)
```

**Methods:** `calculate_msd()`

---

### `CropArray`

Extract and normalize per-spot image crops.

```python
__init__(image, df_crops, crop_size, remove_outliers=True,
         max_percentile=99.5, selected_time_point=None,
         normalize_each_particle=False)
```

**Methods:** `run()`

---

### `ColocalizationDistance`

Euclidean-distance colocalization between two spot channels.

```python
__init__(df, list_spot_type_to_compare=[0, 1], time_point=0,
         threshold_intensity_0=0, threshold_intensity_1=0,
         threshold_distance=2, show_plot=False, voxel_size_z=None,
         psf_z=None, voxel_size_yx=None, psf_yx=None,
         report_codetected_spots_in_both_channels=False)
```

**Methods:** `extract_spot_classification_from_df()`

---

### `PointSpreadFunction`

Build a mean PSF crop and fit 3-D Gaussian.

```python
__init__(image, df_crops, crop_size_xy=5, crop_size_z=3,
         remove_outliers=True, selected_color_channel=None,
         min_percentile=0.5, max_percentile=99, show_plot=False,
         plot_name='temp.png', save_plots=False)
```

**Methods:** `calculate()`, `fit_3D_gaussian(data)`, `gaussian_3d(coords, ...)`

---

### `Correlation`

Auto- and cross-correlation with baseline correction and multi-tau support.

```python
__init__(primary_data, secondary_data=None, max_lag=None,
         nan_handling='zeros', return_full=True, use_bootstrap=True,
         shift_data=False, show_plot=False, save_plots=False,
         plot_name='temp_AC.png', time_interval_between_frames_in_seconds=1,
         index_max_lag_for_fit=None, color_channel=0, start_lag=0,
         line_color='blue', line_color_fit='red', correct_baseline=False,
         baseline_offset=None, use_global_mean=False, plot_title=None,
         fit_type='linear', de_correlation_threshold=0.01,
         use_linear_projection_for_lag_0=True, normalize_plot_with_g0=False,
         remove_outliers=True, MAD_THRESHOLD_FACTOR=6.0,
         plot_individual_trajectories=False, y_axes_min_max_list_values=None,
         x_axes_min_max_list_values=None, multi_tau=False,
         multi_tau_raw_points=20, multi_tau_bins_per_stage=8,
         baseline_method='auto_plateau', baseline_manual_range=None,
         baseline_plateau_fraction=0.25, baseline_percentile=10.0,
         baseline_smooth_window=7, baseline_min_points=5,
         baseline_weight_by_pairs=True, figsize=(8, 6))
```

**Methods:** `run()`

---

### `Utilities`

Static utility methods for image processing and data manipulation.

```python
__init__()
```

#### Array & Trajectory Manipulation

| Method | Description |
| --- | --- |
| `forward_fill_nan(data)` | Forward-fill NaN in 1-D array |
| `forward_fill_nan_2d(data)` | Forward-fill NaN in 2-D array |
| `detrend_trajectories(intensity_array)` | Remove linear trend from trajectories |
| `downsample_array(arr, factor, method)` | Downsample array by integer factor |
| `simulate_missing_data(matrix1, matrix2, ...)` | Randomly blank data entries |
| `shift_initial_nans(data)` | Move leading NaNs to end |
| `find_last_valid_column(data)` | Find last non-NaN column index |
| `remove_nan_rows(array, nan_percentage)` | Drop rows exceeding NaN fraction |
| `shift_trajectories(array_ch0, array_ch1, ...)` | Align trajectory arrays |
| `df_fields_to_arrays_aligned(dataframe, ...)` | Extract two DataFrame fields as aligned arrays |
| `df_trajectories_to_array(dataframe, ...)` | Convert trajectory DataFrame to array |
| `df_extract_data(dataframe, spot_type, ...)` | Extract spot-type subset from DataFrame |
| `summary_df_by_spot_type(df)` | Summarize DataFrame by spot type |
| `remove_outliers(array, min_percentile, max_percentile)` | Percentile clip 1-D array |
| `parse_bool_or_int(value)` | Parse string to bool or int |

#### Image Processing

| Method | Description |
| --- | --- |
| `log_filter(image_TZYXC, spot_radius_px)` | Laplacian-of-Gaussian filter |
| `gaussian_laplace_filter_image(image_TZYXC, ...)` | Apply LoG across channels |
| `pad_image(image, pixels_to_pad)` | Zero-pad image |
| `erode_mask(img, px_to_remove)` | Binary erosion of mask |
| `convert_to_int8(image, rescale, ...)` | Convert array to uint8 |
| `calculate_projection(image, axis, projection_method)` | Mean/max projection |
| `calculate_sharpness(list_images, ...)` | Local-variance sharpness metric |
| `remove_images_not_processed(images_metadata, list_images)` | Filter unprocessed images |

#### Spot & Gaussian Tools

| Method | Description |
| --- | --- |
| `two_dimensional_gaussian(x_y, ...)` | Evaluate 2-D Gaussian model |
| `generate_gaussian_data(amplitude, ...)` | Synthesize 2-D Gaussian array |
| `fit_2D_gaussian(data)` | Fit 2-D Gaussian to crop |
| `optimize_spot_size(frame_data, x_pos, y_pos, ...)` | Find best-fit spot radius |
| `calculate_SNR(mean_array, spot_size)` | Compute signal-to-noise ratio |
| `calculate_threshold_for_spot_detection(image_TZYXC, ...)` | Compute detection threshold |
| `calculate_threshold_from_percentage(tested_image, ...)` | Threshold from target spot % |
| `is_spot_in_crop(selected_crop_id, ...)` | Check if crop contains a spot |
| `normalize_crop_return_list(array_crops_YXC, ...)` | Normalize crops to [0, 1] |
| `test_particle_presence_all_frames_with_ML(...)` | ML-based spot presence test |

#### Mask Operations

| Method | Description |
| --- | --- |
| `masks_to_contours(masks, downsample_factor)` | Convert label masks to contour lists |
| `contours_to_maks(contours, image_shape)` | Reconstruct mask from contours |
| `merge_masks(list_masks)` | Combine multiple mask arrays |
| `separate_masks(masks)` | Split merged mask into components |
| `spots_in_mask(df, mask, edge_exclusion_px)` | Filter spots inside mask |
| `reorder_mask_image(mask_image_tested)` | Sort mask labels by area |
| `return_n_masks(mask_image_tested, number_of_selected_masks)` | Keep N largest masks |
| `remove_artifacts_from_mask_image(mask_image_tested, minimal_mask_area_size)` | Remove small mask artifacts |
| `metric_max_cells_and_area(masks, mode)` | Score mask quality |

#### DataFrame & Pipeline Helpers

| Method | Description |
| --- | --- |
| `spots_in_mask(df, mask, edge_exclusion_px)` | Filter spots by mask |
| `remove_cells_below_spots_threshold(df, ...)` | Filter cells with too few spots |
| `image_cell_selection(cell_id, list_images, ...)` | Crop image around a cell |
| `extract_spot_location_from_cell(df, ...)` | Get spot coordinates for one cell |
| `spot_crops(image, df, number_crops_to_show, spot_size)` | Extract spot crop array |
| `generate_random_colocalized_trajectories(...)` | Simulate paired trajectories |
| `merge_trajectories(df_trajectories_0, df_trajectories_1, ...)` | Link trajectories across channels |
| `extracting_data_for_each_df_in_directory(...)` | Batch-load per-folder DataFrames |
| `extract_data_interpretation(...)` | Aggregate multi-condition results |
| `function_get_df_columns_as_array(df, ...)` | Extract DataFrame column as array |
| `convert_list_to_df(list_number_cells, ...)` | Convert result lists to DataFrame |
| `export_data_to_CSV(...)` | Write spot counts to CSV |

#### File & Folder Utilities

| Method | Description |
| --- | --- |
| `find_src_directory(current_directory)` | Walk up to repo `src/` root |
| `find_folders_by_keywords(base_path, keywords)` | Glob folders by keyword list |
| `clear_folder_except_substring(directory, substring)` | Delete files not matching substring |
| `convert_str_to_path(file_path)` | Coerce string to `Path` |
| `unzip_local_folders(list_local_files, local_folder_path)` | Extract ZIP archives |
| `combine_images_vertically(image_paths, save_path, ...)` | Stack images into single PNG |
| `show_metadta_and_plot_imeges(data_folder_path, ...)` | Print metadata and preview images |
| `get_one_drive_dir()` | Return OneDrive root path |
| `convert_to_standard_format(data_folder_path, ...)` | Standardize raw folder layout |
| `create_output_folders(data_folder_path, ...)` | Create pipeline output directories |

#### Decorators & Miscellaneous

| Method | Description |
| --- | --- |
| `metadata_decorator(...)` | Decorator to auto-save metadata |
| `get_metadata_folder(*args, **kwargs)` | Resolve metadata output folder |
| `is_None(variable_to_test)` | Check if value is None-like |
| `make_it_a_list(variable_to_test)` | Wrap scalar in list if needed |

---

### `Plots` *(microscopy)*

Visualization utilities for microscopy data. No constructor arguments required.

```python
__init__()
```

#### Image & Cell Views

| Method | Description |
| --- | --- |
| `plot_images(image, df, masks, ...)` | Multi-channel image overview |
| `plot_cell_zoom_selected_crop(image_TZYXC, df, ...)` | Zoomed cell + spots |
| `plot_cell_zoom_with_timecourse(image_TZYXC, df, ...)` | Zoomed cell + intensity time course |
| `plot_cell_zoom_with_timecourse_horizontal(...)` | Horizontal layout variant |
| `plot_single_cell(image, df, selected_channel, ...)` | Single cell render |
| `plot_single_cell_all_channels(image, df, ...)` | Single cell across all channels |
| `plot_selected_cell_colors(image, df, ...)` | Overlay spots on false-color cell |
| `plot_cell_all_z_planes(image, ...)` | All z-planes montage |
| `plot_complete_fov(list_images, df, ...)` | Full field-of-view overview |
| `plot_all_cells(list_images, complete_dataframe, ...)` | All cells tiled view |
| `plot_all_cells_and_spots(list_images, ...)` | All cells with spot overlay |
| `plotting_masks_and_original_image(image, ...)` | Mask + raw image overlay |
| `plotting_all_original_images(list_images, ...)` | Raw image montage |
| `plotting_segmentation_images(directory, ...)` | Segmentation results montage |

#### Spot & Trajectory Views

| Method | Description |
| --- | --- |
| `plot_trajectories_and_mask(df, masks, ...)` | 2-D/3-D trajectory map |
| `plot_3d_video_detected_spots(original_image, ...)` | 3-D spot detection summary |
| `plot_colocalized_spots(filtered_images, ...)` | Colocalized vs. exclusive spots |
| `plot_croparray(croparray, crop_size, ...)` | Grid of spot crops |
| `plot_average_crops(mean_crop, crop_size, ...)` | Mean spot crop panel |
| `plot_matrix_crops(mean_crop, crop_size, ...)` | Matrix layout of crops |
| `plot_matrix_pair_crops(mean_crop, crop_size, ...)` | Paired-channel crop matrix |
| `plot_single_particle(croparray, crop_size, ...)` | Single spot crop detail |
| `plot_crops_properties(list_particles_arrays, ...)` | Amplitude/sigma distributions |
| `plot_pixel_properties(list_amplitude, ...)` | Per-pixel Gaussian fit properties |
| `plot_merged_trajectories(df_trajectories_0, ...)` | Merged trajectory overview |

#### Correlation & Distribution Views

| Method | Description |
| --- | --- |
| `plot_autocorrelation(mean_correlation, ...)` | ACF with fit |
| `plot_crosscorrelation(intensity_array_ch0, ...)` | CCF plot |
| `plot_histograms_from_df(df_tracking, ...)` | Per-field histogram |
| `plot_image_pixel_intensity_distribution(image, ...)` | Pixel intensity histogram |
| `dist_plots(df, plot_title, ...)` | Distribution panel |
| `plot_comparing_df(df_all, df_cyto, df_nuc, ...)` | Cytosol vs. nucleus comparison |
| `plot_TS(df_original, plot_title, ...)` | Transcription-site spot plot |
| `plot_TS_bar_stacked(df_original, ...)` | Stacked-bar TS variant |
| `plotting_results_as_distributions(...)` | Full distribution panel |
| `plot_scatter_and_distributions(x, y, ...)` | Scatter + marginal histograms |
| `plot_scatter_bleed_thru(dataframe, ...)` | Bleed-through scatter |
| `plot_all_distributions(dataframe, ...)` | Complete distribution summary |
| `plot_spot_intensity_distributions(dataframe, ...)` | Per-spot intensity distributions |
| `plot_nuc_cyto_int_ratio_distributions(dataframe, ...)` | Nuc/cyto intensity ratio |
| `plot_cell_size_spots(...)` | Cell-size vs. spot-count scatter |
| `plot_cell_intensity_spots(dataframe, ...)` | Cell intensity vs. spot count |
| `plot_interpretation_distributions(...)` | Multi-condition interpretation |
| `compare_intensities_spots_interpretation(...)` | Cross-condition intensity comparison |
| `plot_matrix_sample_time(array1, array2, ...)` | Raster sample-time matrix |

#### Interactive Widgets

| Method | Description |
| --- | --- |
| `visualize_image_widget(image_TZYXC)` | Jupyter slider viewer |
| `display_visualization_plot(ax, frame_idx, ...)` | Render one frame into axes |
| `Napari_Visualizer(tested_image_TZYXC, ...)` | Export Napari-compatible GIF |

---

### `SliderWidgetTracking`

Interactive Jupyter widget for threshold-tuning with spot overlay.

```python
__init__(image_TZYXC, masks=None, list_voxels=None, time_point=None,
         list_spot_size_px=None, channels_spots=None, channels_cytosol=None,
         channels_nucleus=None, min_length_trajectory=3,
         yx_spot_size_in_px=2, starting_threshold=500,
         channel_for_tracking=0)
```

**Methods:** `display()`, `get_threshold()`, `get_cached_dataframes()`,
`get_cached_images()`, `plot_filtered_image(selected_time, max_percentile, threshold)`

---

### `SingleTimePointSpotDetection`

Single time-point spot detection widget.

```python
__init__(image_TZYXC, masks=None, list_voxels=None, list_spot_size_px=None,
         channels_spots=None, channels_cytosol=None, channels_nucleus=None,
         yx_spot_size_in_px=2, starting_threshold=500, channel_for_tracking=0)
```

**Methods:** `display()`, `detect_spots(time_point, threshold)`,
`plot_spots(df, list_filtered_images, max_percentile)`,
`get_threshold_and_time()`, `get_cached_dataframes()`, `get_cached_images()`

---

### `SliderPlotting`

Simple Jupyter slider for browsing image stacks.

```python
__init__(image_TZYXC, masks=None, cmap='custom', df_tracking=None,
         use_gaussian_filter=False, sigma=1.5)
```

**Methods:** `display()`, `plot_filtered_image(selected_time, max_percentiles, sigma)`,
`convert_to_uint8(image, rescale, min_percentile, max_percentile)`

---

### `VideoTracking`

Generate MP4-ready frames for a tracked particle video.

```python
__init__(image_TZYXC, df_tracking, voxel_xy_um, list_channel_order_to_plot,
         list_max_percentile, min_percentile, zoom_size, selected_spot,
         figsize=(10, 10), dpi=150)
```

**Methods:** `generate_video_frames(max_percentile)`, `display_video(video_frames)`

---

## `microlive.ml_spot_detection` — `microlive/ml_spot_detection.py`

CNN-based spot classification for colocalization verification.

### Functions

| Function | Description |
| --- | --- |
| `random_rotate_image(image)` | Random 90° rotation augmentation |
| `load_model(model, path)` | Load model weights from file |
| `save_model(model, path)` | Save model weights to file |
| `normalize_crop_return_list(array_crops_YXC, ...)` | Normalize crops to [0, 255] |
| `standardize_spot_return_list(array_crops_YXC, ...)` | Z-score standardize crops |
| `standarize_crop(crop)` | Z-score single crop |
| `normalize_crop(crop)` | Min-max normalize single crop |
| `predict_crops(model, list_crops, threshold)` | Run inference on crop batch |
| `validate(model, loader, criterion, device)` | Validation loop |
| `run_network(image_dir, num_epochs, ...)` | Train the CNN |

### Classes

#### `ParticleDetectionCNN`

Lightweight CNN binary classifier for spot/no-spot.

```python
__init__()
```

**Methods:** `forward(x)`

#### `ParticleDataset`

PyTorch Dataset for labeled spot crops.

```python
__init__(images_dir, subset='train', use_transform=False)
```

---

## `microlive.pipelines.pipeline_particle_tracking`

### Functions (microlive.pipelines.pipeline_particle_tracking)

#### `pipeline_particle_tracking(...)`

Top-level batch runner. Discovers images in `data_folder_path`, calls
`process_single_image` for each, aggregates results.

Key parameters: `channels_spots`, `channels_cytosol`, `channels_nucleus`,
`min_length_trajectory`, `yx_spot_size_in_px`, `z_spot_size_in_px`,
`apply_photobleaching_correction`, `calculate_MSD`, `calculate_correlations`,
`results_folder_path`.

#### `process_single_image(...)`

Process one image through the full pipeline:
segmentation → photobleaching correction → spot detection →
particle tracking → MSD → correlation.

---

## `microlive.pipelines.pipeline_FRAP`

FRAP analysis pipeline. See source for full function list.

### Key Functions

| Function | Description |
| --- | --- |
| `read_lif_files_in_folder(folder_path)` | Load all LIFs in a directory |
| `segment_image(image_TXY, ...)` | Segment FRAP field |
| `find_frap_roi(image_TZXYC_masked, ...)` | Detect bleached ROI |
| `detect_roi_by_difference(...)` | Detect ROI by pre/post difference |
| `detect_roi_by_tracking(...)` | Detect ROI by particle tracking |
| `process_selected_df(df_roi, ...)` | Compute FRAP curves from ROI df |
| `fit_model_to_frap(time, intensity, ...)` | Single-exponential FRAP fit |
| `fit_model_to_frap_immobile_fraction(...)` | FRAP fit with immobile fraction |
| `plot_frap_quantification(...)` | FRAP intensity curves |
| `plot_frap_quantification_all_images(...)` | Per-image FRAP summary |
| `plot_images_frap(...)` | Selected-frame image montage |
| `plot_t_half_values(df_fit, ...)` | t½ distribution plot |
| `create_pdf(list_combined_image_paths, ...)` | Bundle images into PDF |
| `concatenate_images(list_images, ...)` | Merge image lists |
| `create_image_arrays(...)` | Build image arrays from LIF list |
| `remove_cell_without_roi_detection(df, ...)` | Filter cells with no ROI |

---

## `microlive.pipelines.pipeline_folding_efficiency`

### Functions (microlive.pipelines.pipeline_folding_efficiency)

| Function | Description |
| --- | --- |
| `pipeline_folding_efficiency(...)` | Full folding-efficiency pipeline |
| `metadata_folding_efficiency(...)` | Write pipeline metadata record |

---

## `microlive.utils`

### `microlive.utils.resources`

| Function | Description |
| --- | --- |
| `get_package_data_dir()` | Path to `microlive/data/` |
| `get_icon_path()` | Path to application icon |
| `get_model_path()` | Path to bundled model file |

### `microlive.utils.model_downloader`

| Function | Description |
| --- | --- |
| `get_model_path(model_name)` | Resolve cached model path |
| `is_model_cached(model_name)` | Check if model is in cache |
| `cache_model(model_name, force_download)` | Download and cache model |
| `verify_model_integrity(model_name)` | Checksum verification |
| `list_cached_models()` | Dictionary of all cached models |
| `clear_model_cache(model_name)` | Remove model(s) from cache |
| `get_frap_nuclei_model_path()` | Path to FRAP nuclei model |
| `download_url_to_file(url, dst, progress)` | Low-level URL downloader |

### `microlive.utils.device`

| Function | Description |
| --- | --- |
| `get_device()` | Return best available torch device |
| `is_gpu_available()` | Return True if GPU is detected |
| `get_device_info()` | Dict of device properties |
| `check_gpu_status()` | Print GPU status summary |
