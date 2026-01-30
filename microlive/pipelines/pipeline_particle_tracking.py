"""Particle tracking pipeline for live-cell microscopy analysis.

This module provides a complete pipeline for automated particle detection,
tracking, and quantification in live-cell microscopy images. It integrates
segmentation, spot detection, trajectory linking, MSD analysis, and
correlation analysis.

The pipeline supports:
    - Leica LIF file format with automatic metadata extraction
    - Multi-channel spot detection and tracking
    - Cell segmentation via Cellpose or watershed
    - Photobleaching correction
    - Mean Squared Displacement (MSD) analysis
    - Auto- and cross-correlation analysis

Example:
    >>> import microlive.microscopy as mi
    >>> from microlive.pipelines import pipeline_particle_tracking
    >>> 
    >>> results = pipeline_particle_tracking(
    ...     data_folder_path=pathlib.Path("experiment.lif"),
    ...     selected_image=0,
    ...     channels_spots=[0],
    ...     channels_cytosol=[1],
    ... )

Authors:
    Luis U. Aguilera, Ning Zhao

License:
    GPL v3
"""
from microlive.imports import *

# =============================================================================
# Constants
# =============================================================================

# Default percentile values for image visualization
DEFAULT_MIN_PERCENTILE = 0.5
DEFAULT_MAX_PERCENTILE = 99.9
DEFAULT_MIN_PERCENTILE_TRACKING = 0.05
DEFAULT_MAX_PERCENTILE_TRACKING = 99.95

# Default quality thresholds
DEFAULT_MINIMAL_SNR = 0.5
DEFAULT_MAX_CROP_PERCENTILE = 99


def pipeline_particle_tracking(
    data_folder_path,
    selected_image,
    channels_spots,
    max_spots_for_threshold=100000,
    show_plot=True,
    channels_cytosol=None,
    channels_nucleus=None,
    memory=1,
    min_length_trajectory=5,
    yx_spot_size_in_px=5,
    z_spot_size_in_px=2,
    maximum_spots_cluster=4,
    cluster_radius_nm=500,
    MINIMAL_SNR=0.5,
    diameter_cytosol=300,
    diameter_nucleus=200,
    segmentation_selection_metric='max_area',
    recalculate_mask=False,
    optimization_segmentation_method='diameter_segmentation',
    pretrained_model_cyto_segmentation=None,
    use_watershed=False,
    list_images_to_process=None,
    save_3d_visualization=False,
    max_percentage_empty_data_in_trajectory=0.1,
    particle_detection_threshold=None,
    save_croparray=False,
    apply_photobleaching_correction=False,
    photobleaching_mode='inside_cell',
    use_maximum_projection=False,
    max_lag_for_MSD=30,
    step_size_in_sec=1,
    separate_clusters_and_spots=False,
    maximum_range_search_pixels=10,
    results_folder_path=None,
    calculate_MSD=True,
    calculate_correlations=True,
    link_particles=True
):
    """Execute the complete particle tracking pipeline on LIF microscopy data.

    This function orchestrates the full analysis workflow: loading images,
    segmentation, spot detection, tracking, and downstream analysis (MSD,
    correlations). It can process a single image or batch process multiple
    images from a LIF file.

    Args:
        data_folder_path: Path to the LIF file or data folder.
        selected_image: Index of the image to process, or None for batch mode.
        channels_spots: List of channel indices for spot detection.
        max_spots_for_threshold: Maximum spots to consider for threshold
            calculation. Defaults to 100000.
        show_plot: Whether to display plots during processing. Defaults to True.
        channels_cytosol: List of channel indices for cytosol segmentation.
            Defaults to None.
        channels_nucleus: List of channel indices for nucleus segmentation.
            Defaults to None.
        memory: Number of frames to remember for trajectory linking.
            Defaults to 1.
        min_length_trajectory: Minimum trajectory length in frames.
            Defaults to 5.
        yx_spot_size_in_px: Expected spot size in XY (pixels). Defaults to 5.
        z_spot_size_in_px: Expected spot size in Z (pixels). Defaults to 2.
        maximum_spots_cluster: Maximum spots per cluster. Defaults to 4.
        cluster_radius_nm: Cluster radius in nanometers. Defaults to 500.
        MINIMAL_SNR: Minimum signal-to-noise ratio for trajectories.
            Defaults to 0.5.
        diameter_cytosol: Expected cytosol diameter for Cellpose (pixels).
            Defaults to 300.
        diameter_nucleus: Expected nucleus diameter for Cellpose (pixels).
            Defaults to 200.
        segmentation_selection_metric: Metric for mask selection
            ('max_area', 'center', etc.). Defaults to 'max_area'.
        recalculate_mask: Force mask recalculation even if cached.
            Defaults to False.
        optimization_segmentation_method: Segmentation optimization method.
            Defaults to 'diameter_segmentation'.
        pretrained_model_cyto_segmentation: Path to pretrained Cellpose model.
            Defaults to None.
        use_watershed: Use watershed instead of Cellpose. Defaults to False.
        list_images_to_process: List of image names to process in batch mode.
            Defaults to None (all images).
        save_3d_visualization: Save 3D Napari visualization. Defaults to False.
        max_percentage_empty_data_in_trajectory: Maximum fraction of NaN values
            allowed in trajectories. Defaults to 0.1.
        particle_detection_threshold: Manual threshold for spot detection.
            Defaults to None (auto-calculate).
        save_croparray: Save crop array visualization. Defaults to False.
        apply_photobleaching_correction: Apply photobleaching correction.
            Defaults to False.
        photobleaching_mode: Mode for photobleaching correction
            ('inside_cell', 'whole_image'). Defaults to 'inside_cell'.
        use_maximum_projection: Use Z maximum projection. Defaults to False.
        max_lag_for_MSD: Maximum lag time for MSD calculation.
            Defaults to 30.
        step_size_in_sec: Time step between frames in seconds.
            Defaults to 1.
        separate_clusters_and_spots: Separate clustered spots during detection.
            Defaults to False.
        maximum_range_search_pixels: Maximum search range for linking (pixels).
            Defaults to 10.
        results_folder_path: Custom path for results. Defaults to None.
        calculate_MSD: Whether to calculate MSD. Defaults to True.
        calculate_correlations: Whether to calculate correlations.
            Defaults to True.
        link_particles: Whether to link detected spots into trajectories.
            If False, performs detection-only. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - final_df (pd.DataFrame): Combined tracking results for all images.
            - list_df (list): List of DataFrames, one per processed image.
            - list_masks (list): List of segmentation masks.
            - list_images_tested (list): List of processed images.
            - list_diffusion_coefficient (list): List of diffusion coefficients.

    Raises:
        FileNotFoundError: If the data_folder_path does not exist.
        ValueError: If channels_spots is empty or invalid.

    Example:
        >>> # Process a single image
        >>> df, dfs, masks, images, D = pipeline_particle_tracking(
        ...     data_folder_path=pathlib.Path("data.lif"),
        ...     selected_image=0,
        ...     channels_spots=[0],
        ...     channels_cytosol=[1],
        ...     min_length_trajectory=10,
        ...     calculate_MSD=True
        ... )
        >>> print(f"Found {len(df)} spots, D = {D[0]:.4f} um²/s")
    """
    # Read images and metadata from LIF file
    (
        list_images,
        list_names,
        pixel_xy_um,
        voxel_z_um,
        channel_names,
        number_color_channels,
        list_time_intervals,
        bit_depth
    ) = mi.ReadLif(
        data_folder_path,
        show_metadata=False,
        save_tif=False,
        save_png=False,
        format='TZYXC'
    ).read()

    # Keep a complete copy for reference
    list_images_complete = list_images.copy()

    # Filter images if a subset is specified
    if list_images_to_process is not None:
        selected_indices = [
            i for i in range(len(list_names))
            if list_names[i] in list_images_to_process
        ]
        if use_maximum_projection:
            list_images = [
                np.max(list_images[i], axis=1, keepdims=True)
                for i in selected_indices
            ]
        else:
            list_images = [list_images[i] for i in selected_indices]
    else:
        selected_indices = range(len(list_names))
        if use_maximum_projection:
            list_images = [
                np.max(img, axis=1, keepdims=True) for img in list_images
            ]

    # Batch processing: process all images if selected_image is None
    if selected_image is None:
        list_df = []
        list_masks = []
        list_images_tested = []
        list_diffusion_coefficient = []

        for idx in range(len(list_images_complete)):
            if idx not in selected_indices:
                continue

            df, masks, image, diffusion_coefficient = process_single_image(
                data_folder_path=data_folder_path,
                selected_image=idx,
                channels_spots=channels_spots,
                max_spots_for_threshold=max_spots_for_threshold,
                show_plot=show_plot,
                channels_cytosol=channels_cytosol,
                channels_nucleus=channels_nucleus,
                min_length_trajectory=min_length_trajectory,
                yx_spot_size_in_px=yx_spot_size_in_px,
                z_spot_size_in_px=z_spot_size_in_px,
                maximum_spots_cluster=maximum_spots_cluster,
                cluster_radius_nm=cluster_radius_nm,
                MINIMAL_SNR=MINIMAL_SNR,
                diameter_cytosol=diameter_cytosol,
                diameter_nucleus=diameter_nucleus,
                segmentation_selection_metric=segmentation_selection_metric,
                list_images=list_images_complete,
                list_names=list_names,
                pixel_xy_um=pixel_xy_um,
                voxel_z_um=voxel_z_um,
                channel_names=channel_names,
                image_time_interval=list_time_intervals[idx],
                recalculate_mask=recalculate_mask,
                optimization_segmentation_method=optimization_segmentation_method,
                pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation,
                use_watershed=use_watershed,
                save_3d_visualization=save_3d_visualization,
                apply_photobleaching_correction=apply_photobleaching_correction,
                photobleaching_mode=photobleaching_mode,
                use_maximum_projection=use_maximum_projection,
                max_lag_for_MSD=max_lag_for_MSD,
                step_size_in_sec=step_size_in_sec,
                separate_clusters_and_spots=separate_clusters_and_spots,
                maximum_range_search_pixels=maximum_range_search_pixels,
                max_percentage_empty_data_in_trajectory=max_percentage_empty_data_in_trajectory,
                memory=memory,
                particle_detection_threshold=particle_detection_threshold,
                results_folder_path=results_folder_path,
                calculate_MSD=calculate_MSD,
                calculate_correlations=calculate_correlations,
                save_croparray=save_croparray,
                link_particles=link_particles,
            )

            # Skip empty results
            if df.empty:
                continue

            # Update image_id to reflect the original index
            df['image_id'] = idx
            list_df.append(df)
            list_masks.append(masks)
            list_images_tested.append(image)
            list_diffusion_coefficient.append(diffusion_coefficient)

        # Combine all DataFrames
        if len(list_df) > 1:
            final_df = pd.concat(list_df, ignore_index=True)
        elif len(list_df) == 1:
            final_df = list_df[0]
        else:
            final_df = pd.DataFrame()

        return final_df, list_df, list_masks, list_images_tested, list_diffusion_coefficient

    else:
        # Single image processing
        df, masks, image, diffusion_coefficient = process_single_image(
            data_folder_path=data_folder_path,
            selected_image=selected_image,
            channels_spots=channels_spots,
            max_spots_for_threshold=max_spots_for_threshold,
            show_plot=show_plot,
            channels_cytosol=channels_cytosol,
            channels_nucleus=channels_nucleus,
            min_length_trajectory=min_length_trajectory,
            yx_spot_size_in_px=yx_spot_size_in_px,
            z_spot_size_in_px=z_spot_size_in_px,
            maximum_spots_cluster=maximum_spots_cluster,
            cluster_radius_nm=cluster_radius_nm,
            MINIMAL_SNR=MINIMAL_SNR,
            diameter_cytosol=diameter_cytosol,
            diameter_nucleus=diameter_nucleus,
            segmentation_selection_metric=segmentation_selection_metric,
            list_images=list_images,
            list_names=list_names,
            pixel_xy_um=pixel_xy_um,
            voxel_z_um=voxel_z_um,
            channel_names=channel_names,
            image_time_interval=list_time_intervals[selected_image],
            recalculate_mask=recalculate_mask,
            optimization_segmentation_method=optimization_segmentation_method,
            pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation,
            use_watershed=use_watershed,
            save_3d_visualization=save_3d_visualization,
            apply_photobleaching_correction=apply_photobleaching_correction,
            photobleaching_mode=photobleaching_mode,
            use_maximum_projection=use_maximum_projection,
            max_lag_for_MSD=max_lag_for_MSD,
            step_size_in_sec=step_size_in_sec,
            separate_clusters_and_spots=separate_clusters_and_spots,
            maximum_range_search_pixels=maximum_range_search_pixels,
            max_percentage_empty_data_in_trajectory=max_percentage_empty_data_in_trajectory,
            memory=memory,
            particle_detection_threshold=particle_detection_threshold,
            results_folder_path=results_folder_path,
            calculate_MSD=calculate_MSD,
            calculate_correlations=calculate_correlations,
            save_croparray=save_croparray,
            link_particles=link_particles,
        )

        return df, [df], [masks], [image], [diffusion_coefficient]


@mi.Utilities().metadata_decorator(
    metadata_folder_func=mi.Utilities().get_metadata_folder,
    exclude_args=['list_images']
)
def process_single_image(
    data_folder_path,
    selected_image,
    channels_spots,
    max_spots_for_threshold=100000,
    show_plot=True,
    channels_cytosol=None,
    channels_nucleus=None,
    memory=1,
    min_length_trajectory=5,
    yx_spot_size_in_px=5,
    z_spot_size_in_px=2,
    maximum_spots_cluster=4,
    cluster_radius_nm=500,
    MINIMAL_SNR=0.5,
    diameter_cytosol=300,
    diameter_nucleus=200,
    segmentation_selection_metric='area',
    list_images=None,
    list_names=None,
    pixel_xy_um=None,
    voxel_z_um=None,
    channel_names=None,
    optimization_segmentation_method='diameter_segmentation',
    recalculate_mask=False,
    use_watershed=False,
    pretrained_model_cyto_segmentation=None,
    particle_detection_threshold=None,
    save_croparray=False,
    image_time_interval=None,
    save_3d_visualization=False,
    apply_photobleaching_correction=False,
    photobleaching_mode='inside_cell',
    max_percentage_empty_data_in_trajectory=0.1,
    use_maximum_projection=False,
    max_lag_for_MSD=30,
    step_size_in_sec=1,
    separate_clusters_and_spots=False,
    maximum_range_search_pixels=10,
    results_folder_path=None,
    calculate_MSD=True,
    calculate_correlations=True,
    link_particles=True
):
    """Process a single image through the complete particle tracking workflow.

    This function handles the core analysis for one image: segmentation,
    spot detection, trajectory linking, quality filtering, and optional
    MSD and correlation analysis.

    Args:
        data_folder_path: Path to the data folder or LIF file.
        selected_image: Index of the image to process.
        channels_spots: List of channel indices for spot detection.
        max_spots_for_threshold: Maximum spots for threshold calculation.
        show_plot: Whether to display plots.
        channels_cytosol: List of channels for cytosol segmentation.
        channels_nucleus: List of channels for nucleus segmentation.
        memory: Frames to remember for trajectory linking.
        min_length_trajectory: Minimum trajectory length.
        yx_spot_size_in_px: Expected spot size in XY (pixels).
        z_spot_size_in_px: Expected spot size in Z (pixels).
        maximum_spots_cluster: Maximum spots per cluster.
        cluster_radius_nm: Cluster radius in nanometers.
        MINIMAL_SNR: Minimum SNR threshold for quality filtering.
        diameter_cytosol: Expected cytosol diameter (pixels).
        diameter_nucleus: Expected nucleus diameter (pixels).
        segmentation_selection_metric: Metric for mask selection.
        list_images: Pre-loaded list of images.
        list_names: List of image names.
        pixel_xy_um: Pixel size in XY (micrometers).
        voxel_z_um: Voxel size in Z (micrometers).
        channel_names: List of channel names.
        optimization_segmentation_method: Segmentation optimization method.
        recalculate_mask: Force mask recalculation.
        use_watershed: Use watershed segmentation.
        pretrained_model_cyto_segmentation: Path to pretrained model.
        particle_detection_threshold: Manual detection threshold.
        save_croparray: Save crop array visualization.
        image_time_interval: Time interval between frames (seconds).
        save_3d_visualization: Save 3D visualization.
        apply_photobleaching_correction: Apply photobleaching correction.
        photobleaching_mode: Photobleaching correction mode.
        max_percentage_empty_data_in_trajectory: Maximum NaN fraction allowed.
        use_maximum_projection: Use Z maximum projection.
        max_lag_for_MSD: Maximum lag for MSD calculation.
        step_size_in_sec: Time step for MSD (seconds).
        separate_clusters_and_spots: Separate clusters during detection.
        maximum_range_search_pixels: Maximum search range for linking.
        results_folder_path: Custom results folder path.
        calculate_MSD: Whether to calculate MSD.
        calculate_correlations: Whether to calculate correlations.
        link_particles: Whether to link detected spots into trajectories.
            If False, performs detection-only. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - df_tracking (pd.DataFrame): Tracking results with spot positions,
                intensities, and trajectory IDs.
            - masks (np.ndarray): Cell segmentation mask.
            - original_tested_image (np.ndarray): Original image data.
            - diffusion_coefficient (float or None): Calculated D value.

    Raises:
        ValueError: If list_images or list_names is None.
    """
    # Ensure lists are properly formatted
    channels_spots = [channels_spots] if not isinstance(channels_spots, list) else channels_spots
    channels_cytosol = [channels_cytosol] if not isinstance(channels_cytosol, list) else channels_cytosol
    channels_nucleus = [channels_nucleus] if not isinstance(channels_nucleus, list) else channels_nucleus

    # Convert pixel and voxel sizes to nm
    pixel_xy_nm = int(pixel_xy_um * 1000)
    voxel_z_nm = int(voxel_z_um * 1000)
    list_voxels = [voxel_z_nm, pixel_xy_nm]
    list_spot_size_px = [z_spot_size_in_px, yx_spot_size_in_px]

    # Log progress
    print('--------------------------------------------------')
    print(f'Processing image: {list_names[selected_image]}')

    tested_image = list_images[selected_image]  # TZYXC format
    original_tested_image = tested_image.copy()

    # Create results folder
    results_name = f'results_{data_folder_path.stem}_cell_id_{selected_image}'
    current_dir = pathlib.Path().absolute()

    if results_folder_path is not None:
        if not isinstance(results_folder_path, pathlib.Path):
            results_folder_path = pathlib.Path(results_folder_path)
        results_folder = results_folder_path.joinpath(results_name)
    else:
        results_folder = current_dir.joinpath('results_live_cell', results_name)

    results_folder.mkdir(parents=True, exist_ok=True)
    mi.Utilities().clear_folder_except_substring(results_folder, 'mask')

    # Plot the original image
    plot_name_original = results_folder.joinpath('original_image.png')
    suptitle = (
        f'Image: {data_folder_path.stem[:16]} - '
        f'{list_names[selected_image]} - Cell_ID: {selected_image}'
    )

    mi.Plots().plot_images(
        image_ZYXC=tested_image[0],
        figsize=(12, 5),
        show_plot=show_plot,
        use_maximum_projection=True,
        use_gaussian_filter=True,
        cmap='binary',
        min_max_percentile=[DEFAULT_MIN_PERCENTILE, DEFAULT_MAX_PERCENTILE],
        show_gird=False,
        save_plots=True,
        plot_name=plot_name_original,
        suptitle=suptitle
    )

    # Read or create masks
    mask_file_name = f'mask_{data_folder_path.stem}_image_{selected_image}.tif'
    mask_file_path = results_folder.joinpath(mask_file_name)
    path_mask_exist = os.path.exists(str(mask_file_path))

    if path_mask_exist and not recalculate_mask:
        masks = imread(str(mask_file_path)).astype(bool)
    else:
        # Use Cellpose or watershed to create masks
        if use_watershed:
            masks_complete_cells = mi.CellSegmentationWatershed(
                np.max(tested_image[:, :, :, :, channels_cytosol[0]], axis=(0, 1)),
                footprint_size=2
            ).apply_watershed()
        else:
            masks_complete_cells, _, _ = mi.CellSegmentation(
                tested_image[0],
                channels_cytosol=channels_cytosol,
                channels_nucleus=channels_nucleus,
                diameter_cytosol=diameter_cytosol,
                diameter_nucleus=diameter_nucleus,
                optimization_segmentation_method=optimization_segmentation_method,
                remove_fragmented_cells=False,
                show_plot=show_plot,
                image_name=None,
                NUMBER_OF_CORES=1,
                selection_metric=segmentation_selection_metric,
                pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation
            ).calculate_masks()

        # Select the mask at the center of the image
        center_y = masks_complete_cells.shape[0] // 2
        center_x = masks_complete_cells.shape[1] // 2
        selected_mask_id = masks_complete_cells[center_y, center_x]

        if selected_mask_id > 0:
            masks = masks_complete_cells == selected_mask_id
        else:
            # Fall back to the largest mask
            mask_labels = np.unique(masks_complete_cells)
            mask_sizes = [
                (label, np.sum(masks_complete_cells == label))
                for label in mask_labels if label != 0
            ]
            if mask_sizes:
                selected_mask_id = max(mask_sizes, key=lambda x: x[1])[0]
                masks = masks_complete_cells == selected_mask_id
            else:
                masks = np.zeros_like(masks_complete_cells, dtype=bool)

        # Save the mask
        masks = masks.astype(np.uint8)
        tifffile.imwrite(str(mask_file_path), masks, dtype='uint8')

    # Apply photobleaching correction if requested
    if apply_photobleaching_correction:
        file_path_photobleaching = results_folder.joinpath('photobleaching.png')
        corrected_image = mi.Photobleaching(
            image_TZYXC=tested_image,
            mask_YX=masks,
            show_plot=False,
            mode=photobleaching_mode,
            plot_name=file_path_photobleaching
        ).apply_photobleaching_correction()
    else:
        corrected_image = tested_image

    # Calculate the threshold for spot detection
    plot_name_threshold = results_folder.joinpath('threshold_spot_detection.png')

    if particle_detection_threshold is None:
        starting_threshold = mi.Utilities().calculate_threshold_for_spot_detection(
            corrected_image,
            list_spot_size_px,
            list_voxels,
            channels_spots,
            max_spots_for_threshold=max_spots_for_threshold,
            show_plot=True,
            plot_name=plot_name_threshold
        )
    else:
        starting_threshold = [particle_detection_threshold] * len(channels_spots)

    # Run particle tracking
    try:
        list_dataframes_trajectories, _ = mi.ParticleTracking(
            image=corrected_image,
            channels_spots=channels_spots,
            masks=masks,
            list_voxels=list_voxels,
            memory=memory,
            channels_cytosol=channels_cytosol,
            channels_nucleus=channels_nucleus,
            min_length_trajectory=min_length_trajectory,
            threshold_for_spot_detection=starting_threshold,
            yx_spot_size_in_px=yx_spot_size_in_px,
            z_spot_size_in_px=z_spot_size_in_px,
            maximum_spots_cluster=maximum_spots_cluster,
            cluster_radius_nm=cluster_radius_nm,
            separate_clusters_and_spots=separate_clusters_and_spots,
            maximum_range_search_pixels=maximum_range_search_pixels,
            link_particles=link_particles,
        ).run()
    except Exception as e:
        print(f'Error during particle tracking: {e}')
        return pd.DataFrame(), masks, original_tested_image, None

    df_tracking = list_dataframes_trajectories[0]

    if len(df_tracking) == 0:
        return pd.DataFrame(), masks, original_tested_image, None

    # Combine trajectories from multiple channels if present
    if len(list_dataframes_trajectories) > 1:
        for i in range(1, len(list_dataframes_trajectories)):
            df_tracking = pd.concat(
                [df_tracking, list_dataframes_trajectories[i]],
                ignore_index=True
            )
    df_tracking = df_tracking.reset_index(drop=True)

    # Plot histograms for SNR
    selected_field = 'snr'
    plot_name_snr = results_folder.joinpath(f'spots_{selected_field}.png')
    mean_snr = mi.Plots().plot_histograms_from_df(
        df_tracking,
        selected_field=selected_field,
        figsize=(8, 2),
        plot_name=plot_name_snr,
        bin_count=60,
        save_plot=True,
        list_colors=channel_names,
        remove_outliers=True
    )

    # Plot histograms for spot intensity
    selected_field = 'spot_int'
    plot_name_int = results_folder.joinpath(f'spots_{selected_field}.png')
    mean_int = mi.Plots().plot_histograms_from_df(
        df_tracking,
        selected_field=selected_field,
        figsize=(8, 2),
        plot_name=plot_name_int,
        bin_count=60,
        save_plot=True,
        list_colors=channel_names,
        remove_outliers=True
    )

    # Remove tracks with low SNR in the tracking channel
    if MINIMAL_SNR is not None:
        array_selected_field = mi.Utilities().df_trajectories_to_array(
            dataframe=df_tracking,
            selected_field=f'{selected_field}_ch_{channels_spots[0]}',
            fill_value='nans'
        )
        mean_snr = np.nanmean(array_selected_field, axis=1)
        indices_low_quality_tracks = np.where(mean_snr < MINIMAL_SNR)[0]
        df_tracking = df_tracking[~df_tracking['particle'].isin(indices_low_quality_tracks)]
        df_tracking = df_tracking.reset_index(drop=True)
        df_tracking['particle'] = df_tracking.groupby('particle').ngroup()

    # Plot image intensity histogram
    masked_data = corrected_image * masks[np.newaxis, np.newaxis, :, :, np.newaxis].astype(float)
    for i in range(len(channels_spots)):
        plot_name_histogram = results_folder.joinpath(
            f'pixel_histogram_in_cell_{channels_spots[i]}.png'
        )
        mi.Plots().plot_image_pixel_intensity_distribution(
            image=np.mean(masked_data, axis=(0, 1)),
            figsize=(8, 2),
            bins=100,
            remove_outliers=True,
            remove_zeros=True,
            save_plots=True,
            plot_name=plot_name_histogram,
            single_color=None,
            list_colors=channel_names,
            tracking_channel=channels_spots[0],
            threshold_tracking=starting_threshold[i]
        )

    # Plot original image and tracks
    suptitle = (
        f'Image: {data_folder_path.stem[:16]} - '
        f'{list_names[selected_image]} - Cell_ID: {selected_image}'
    )
    plot_name_original_image_and_tracks = results_folder.joinpath('original_image_tracking.png')
    mi.Plots().plot_images(
        image_ZYXC=corrected_image[0],
        df=df_tracking,
        masks=masks,
        show_trajectories=True,
        suptitle=suptitle,
        figsize=(12, 3),
        show_plot=True,
        selected_time=0,
        use_maximum_projection=True,
        use_gaussian_filter=True,
        cmap='binary',
        min_max_percentile=[DEFAULT_MIN_PERCENTILE_TRACKING, DEFAULT_MAX_PERCENTILE_TRACKING],
        show_gird=False,
        save_plots=True,
        plot_name=plot_name_original_image_and_tracks
    )

    # Combine the original image and the image with tracks
    plot_name_complete_image = results_folder.joinpath('complete_image_tracking.png')
    mi.Utilities().combine_images_vertically(
        [plot_name_original, plot_name_original_image_and_tracks],
        plot_name_complete_image,
        delete_originals=True
    )

    # Save the DataFrame
    df_tracking.to_csv(results_folder.joinpath('tracking_results.csv'), index=False)

    # Prepare crop arrays for visualization
    normalize_each_particle = True
    crop_size = yx_spot_size_in_px + 5
    if crop_size % 2 == 0:
        crop_size += 1
    selected_time_point = None

    filtered_image = mi.Utilities().gaussian_laplace_filter_image(
        corrected_image, list_spot_size_px, list_voxels
    )
    croparray_filtered, mean_crop_filtered, first_appearance, crop_size = mi.CropArray(
        image=filtered_image,
        df_crops=df_tracking,
        crop_size=crop_size,
        remove_outliers=False,
        max_percentile=DEFAULT_MAX_PERCENTILE_TRACKING,
        selected_time_point=selected_time_point,
        normalize_each_particle=normalize_each_particle
    ).run()

    # Save crop array if requested
    if save_croparray:
        path_crop_array = results_folder.joinpath('crop_array.png')
        mi.Plots().plot_croparray(
            croparray_filtered,
            crop_size,
            save_plots=True,
            plot_name=path_crop_array,
            suptitle=None,
            show_particle_labels=True,
            cmap='binary_r',
            max_percentile=DEFAULT_MAX_CROP_PERCENTILE
        )

    # Plot pair of crops
    plot_name_crops_filter = results_folder.joinpath('crops.png')
    mi.Plots().plot_matrix_pair_crops(
        mean_crop_filtered,
        crop_size,
        save_plots=True,
        plot_name=plot_name_crops_filter
    )

    # Calculate the Mean Squared Displacement
    plot_name_MSD = results_folder.joinpath('MSD_plot.png')

    if image_time_interval is None:
        image_time_interval = step_size_in_sec
        print(
            f'Warning: image_time_interval not provided. '
            f'Using step_size_in_sec: {step_size_in_sec} seconds.'
        )
    else:
        image_time_interval = float(image_time_interval)

    # MSD and correlations require linked trajectories
    if not link_particles:
        if calculate_MSD or calculate_correlations:
            print(
                'Warning: MSD and correlation calculations require linked trajectories. '
                'Skipping because link_particles=False.'
            )
        diffusion_coefficient = None
    elif calculate_MSD:
        # calculate_msd() returns: D_um2_s, D_px2_s, em_um2, em_px2, fit_times, fit_line_msd, trackpy_df
        diffusion_coefficient, _, _, _, _, _, _ = mi.ParticleMotion(
            df_tracking,
            microns_per_pixel=pixel_xy_um,
            step_size_in_sec=image_time_interval,
            max_lagtime=max_lag_for_MSD,
            show_plot=True,
            remove_drift=False,
            plot_name=plot_name_MSD
        ).calculate_msd()
    else:
        diffusion_coefficient = None

    # Calculate correlations (requires linked trajectories)
    if calculate_correlations and link_particles:
        array_ch0 = mi.Utilities().df_trajectories_to_array(
            dataframe=df_tracking,
            selected_field='spot_int_ch_0',
            fill_value='nans'
        )

        if 'spot_int_ch_1' in df_tracking.columns:
            array_ch1 = mi.Utilities().df_trajectories_to_array(
                dataframe=df_tracking,
                selected_field='spot_int_ch_1',
                fill_value='nans'
            )
            intensity_array_ch0_short, intensity_array_ch1_short = mi.Utilities().shift_trajectories(
                array_ch0,
                array_ch1,
                max_percentage_empty_data_in_trajectory=max_percentage_empty_data_in_trajectory
            )
        else:
            array_ch1 = None
            intensity_array_ch0_short = mi.Utilities().shift_trajectories(
                array_ch0,
                max_percentage_empty_data_in_trajectory=max_percentage_empty_data_in_trajectory
            )
            intensity_array_ch1_short = None

        plot_name_intensity_matrix = results_folder.joinpath('intensity_matrix.png')
        mi.Plots().plot_matrix_sample_time(
            intensity_array_ch0_short,
            intensity_array_ch1_short,
            plot_name=plot_name_intensity_matrix
        )

        plot_name_AC_ch0 = results_folder.joinpath('AC_plot_ch0.png')
        (
            mean_correlation_ch0,
            std_correlation_ch0,
            lags_ch0,
            correlations_array_ch0,
            dwell_time_ch0
        ) = mi.Correlation(
            primary_data=intensity_array_ch0_short,
            max_lag=None,
            nan_handling='ignore',
            shift_data=True,
            return_full=False,
            time_interval_between_frames_in_seconds=image_time_interval,
            show_plot=True,
            start_lag=1,
            fit_type='exponential',
            use_linear_projection_for_lag_0=True,
            save_plots=True,
            plot_name=plot_name_AC_ch0
        ).run()

        if array_ch1 is not None:
            plot_name_AC_ch1 = results_folder.joinpath('AC_plot_ch1.png')
            (
                mean_correlation_ch1,
                std_correlation_ch1,
                lags_ch1,
                correlations_array_ch1,
                dwell_time_ch1
            ) = mi.Correlation(
                primary_data=intensity_array_ch1_short,
                max_lag=None,
                nan_handling='ignore',
                shift_data=True,
                return_full=False,
                time_interval_between_frames_in_seconds=image_time_interval,
                show_plot=True,
                start_lag=1,
                fit_type='exponential',
                use_linear_projection_for_lag_0=True,
                save_plots=True,
                plot_name=plot_name_AC_ch1
            ).run()

            # Plot cross-correlation
            plot_name_cross_correlation = results_folder.joinpath('cross_correlation.png')
            (
                mean_cross_correlation,
                std_cross_correlation,
                lags_cross_correlation,
                cross_correlations_array,
                max_lag
            ) = mi.Correlation(
                primary_data=intensity_array_ch0_short,
                secondary_data=intensity_array_ch1_short,
                max_lag=None,
                nan_handling='ignore',
                shift_data=False,
                return_full=True,
                time_interval_between_frames_in_seconds=image_time_interval,
                show_plot=True,
                save_plots=True,
                plot_name=plot_name_cross_correlation
            ).run()

    # Plot 3D visualization if requested
    if save_3d_visualization:
        mask_expanded = masks[np.newaxis, np.newaxis, :, :, np.newaxis]
        masked_image_TZYXC = filtered_image * mask_expanded
        masked_image_TZYXC = gaussian_filter(masked_image_TZYXC, sigma=1)
        masked_image_TZYXC = mi.RemoveExtrema(
            masked_image_TZYXC,
            min_percentile=0.001,
            max_percentile=99.995
        ).remove_outliers()
        plot_name_3d_visualizer = str(results_folder.joinpath('image_3d.gif'))
        mi.Plots().Napari_Visualizer(
            masked_image_TZYXC,
            df_tracking,
            z_correction=7,
            channels_spots=0,
            plot_name=plot_name_3d_visualizer
        )

    # Log completion
    print(f'Image {list_names[selected_image]} has been processed.')

    return df_tracking, masks, original_tested_image, diffusion_coefficient
