# work on a code to count spots but not link trajectories.
import sys; from pathlib import Path
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))
from imports import *

def pipeline_particle_tracking(data_folder_path, selected_image, channels_spots, max_spots_for_threshold=100000,
                               show_plot=True, channels_cytosol=None, channels_nucleus=None,memory=1,
                               min_length_trajectory=5, yx_spot_size_in_px=5, z_spot_size_in_px=2, maximum_spots_cluster=4,cluster_radius_nm =500,
                               MINIMAL_SNR=0.5, diameter_cytosol=300, diameter_nucleus=200,
                               segmentation_selection_metric='max_area',recalculate_mask=False,optimization_segmentation_method='diameter_segmentation',
                               pretrained_model_cyto_segmentation=None,use_watershed=False,list_images_to_process=None,save_3d_visualization=False,
                               max_percentage_empty_data_in_trajectory=0.1,particle_detection_threshold=None,save_croparray=False,
                               apply_photobleaching_correction=False,photobleaching_mode='inside_cell',use_maximum_projection=False,
                               max_lag_for_MSD=30,step_size_in_sec=1,separate_clusters_and_spots=False,maximum_range_search_pixels=10,results_folder_path=None,
                               calculate_MSD=True,calculate_correlations=True):
    # Read images and metadata
    # detect if the data is a lif file
    list_images, list_names, pixel_xy_um, voxel_z_um, channel_names, number_color_channels, list_time_intervals, bit_depth = \
        mi.ReadLif(data_folder_path, show_metadata=False, save_tif=False, save_png=False, format='TZYXC').read()
    # Prepare full list of images and set up indices to process
    list_images_complete = list_images.copy()
    if list_images_to_process is not None:
        selected_indices = [i for i in range(len(list_names)) if list_names[i] in list_images_to_process]
        if use_maximum_projection:
            # use the maximum projection in Z. but keep the image shape with only one Z slice
            list_images = [np.max(list_images[i], axis=1, keepdims=True) for i in selected_indices]
        else:
            list_images = [list_images[i] for i in selected_indices]
    else:
        selected_indices = range(len(list_names))
        if use_maximum_projection:
            list_images = [np.max(img, axis=1, keepdims=True) for img in list_images]



    # If selected_image is None, process all images
    if selected_image is None:
        list_df, list_masks, list_images_tested, list_diffusion_coefficient = [], [], [], []
        
        #for idx in range(len(list_images)):
        for idx in range(len(list_images_complete)):
            if idx not in selected_indices:
                continue
            df, masks,image,diffusion_coefficient = process_single_image(
                data_folder_path=data_folder_path,
                selected_image=idx,
                channels_spots=channels_spots,
                max_spots_for_threshold=max_spots_for_threshold,
                show_plot=show_plot,
                channels_cytosol=channels_cytosol,
                channels_nucleus=channels_nucleus,
                min_length_trajectory=min_length_trajectory,
                yx_spot_size_in_px=yx_spot_size_in_px,
                z_spot_size_in_px = z_spot_size_in_px,
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
                max_lag_for_MSD = max_lag_for_MSD,
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
            )
            #if df is None:
            # if the df is None or empty, continue to the next image
            if df.empty:
                continue
            # rename the field image_id to idx
            df['image_id'] = idx
            list_df.append(df)
            list_masks.append(masks)
            list_images_tested.append(image)
            list_diffusion_coefficient.append(diffusion_coefficient)

        if len(list_df) >1 :
            final_df = pd.concat(list_df, ignore_index=True)
        else:
            final_df = df
        return final_df, list_df, list_masks, list_images_tested, list_diffusion_coefficient
    else:
        # Process single image
        df,masks,image, diffusion_coefficient= process_single_image(
            data_folder_path=data_folder_path,
            selected_image=selected_image,
            channels_spots=channels_spots,
            max_spots_for_threshold=max_spots_for_threshold,
            show_plot=show_plot,
            channels_cytosol=channels_cytosol,
            channels_nucleus=channels_nucleus,
            min_length_trajectory=min_length_trajectory,
            yx_spot_size_in_px=yx_spot_size_in_px,
            z_spot_size_in_px = z_spot_size_in_px,
            maximum_spots_cluster=maximum_spots_cluster,
            cluster_radius_nm=cluster_radius_nm,
            MINIMAL_SNR=MINIMAL_SNR,
            diameter_cytosol=diameter_cytosol,
            diameter_nucleus=diameter_nucleus,
            segmentation_selection_metric=segmentation_selection_metric,
            list_images=list_images,
            list_names=list_names[idx],
            pixel_xy_um=pixel_xy_um,
            voxel_z_um=voxel_z_um,
            channel_names=channel_names,
            image_time_interval = list_time_intervals[selected_image],
            recalculate_mask=recalculate_mask,
            optimization_segmentation_method=optimization_segmentation_method,
            pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation,
            use_watershed=use_watershed,
            save_3d_visualization=save_3d_visualization,
            apply_photobleaching_correction=apply_photobleaching_correction,
            photobleaching_mode=photobleaching_mode,
            use_maximum_projection=use_maximum_projection,
            max_lag_for_MSD = max_lag_for_MSD,
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
        )
        return df, [df], [masks], [image], [diffusion_coefficient]






@mi.Utilities().metadata_decorator(metadata_folder_func=mi.Utilities().get_metadata_folder,exclude_args=['list_images',]) # exclude_args=['list_images', 'list_names' ]
def process_single_image(data_folder_path, selected_image, channels_spots, max_spots_for_threshold=100000,
                         show_plot=True, channels_cytosol=None, channels_nucleus=None,memory=1,
                         min_length_trajectory=5, yx_spot_size_in_px=5, z_spot_size_in_px=2 , maximum_spots_cluster=4,cluster_radius_nm=500,
                         MINIMAL_SNR=0.5, diameter_cytosol=300, diameter_nucleus=200, segmentation_selection_metric='area',
                         list_images=None, list_names=None, pixel_xy_um=None, voxel_z_um=None,
                         channel_names=None, optimization_segmentation_method='diameter_segmentation',
                         recalculate_mask=False,use_watershed=False, pretrained_model_cyto_segmentation=None,particle_detection_threshold=None,save_croparray=False,
                         image_time_interval=None,save_3d_visualization=False,apply_photobleaching_correction=False,photobleaching_mode='inside_cell',max_percentage_empty_data_in_trajectory=0.1,
                         use_maximum_projection=False,max_lag_for_MSD=30,step_size_in_sec=1,separate_clusters_and_spots=False,maximum_range_search_pixels=10,results_folder_path=None,
                         calculate_MSD=True,calculate_correlations=True):
    # Ensure lists are properly formatted
    channels_spots = [channels_spots] if not isinstance(channels_spots, list) else channels_spots
    channels_cytosol = [channels_cytosol] if not isinstance(channels_cytosol, list) else channels_cytosol
    channels_nucleus = [channels_nucleus] if not isinstance(channels_nucleus, list) else channels_nucleus

    # Convert pixel and voxel sizes to nm
    pixel_xy_nm = int(pixel_xy_um * 1000)
    voxel_z_nm = int(voxel_z_um * 1000)
    list_voxels = [voxel_z_nm, pixel_xy_nm]
    list_spot_size_px = [z_spot_size_in_px, yx_spot_size_in_px]
    # print a line
    print('--------------------------------------------------')
    print(f'Processing image: {list_names[selected_image]}')
    tested_image = list_images[selected_image]  # TZYXC
    original_tested_image = tested_image.copy()
    # Creating the results folder
    results_name = 'results_' + data_folder_path.stem + '_cell_id_' + str(selected_image)
    current_dir = pathlib.Path().absolute()
    
    
    if results_folder_path is not None:
        # ensure that results_folder_path is a Path object
        if not isinstance(results_folder_path, pathlib.Path):
            results_folder_path = pathlib.Path(results_folder_path)
        results_folder = results_folder_path.joinpath(results_name)
    else:
        results_folder = current_dir.joinpath('results_live_cell', results_name)
    results_folder.mkdir(parents=True, exist_ok=True)
    mi.Utilities().clear_folder_except_substring(results_folder, 'mask')
    # Plot the original image
    plot_name_original = results_folder.joinpath('original_image.png')
    suptitle=f'Image: {data_folder_path.stem[:16]} - {list_names[selected_image]} - Cell_ID: {selected_image}'
        
    
    mi.Plots().plot_images(
        image_ZYXC=tested_image[0],
        figsize=(12, 5),
        show_plot=show_plot,
        use_maximum_projection=True,
        use_gaussian_filter=True,
        cmap='binary',
        min_max_percentile=[0.5, 99.9],
        show_gird=False,
        save_plots=True,
        plot_name=plot_name_original,
        suptitle=suptitle
    )
    # Read or create masks
    mask_file_name = 'mask_' + data_folder_path.stem + '_image_' + str(selected_image) + '.tif'
    mask_file_path = results_folder.joinpath(mask_file_name)
    path_mask_exist = os.path.exists(str(mask_file_path))
    if path_mask_exist and recalculate_mask is False:
        masks = imread(str(mask_file_path)).astype(bool)
    else:
        # Use Cellpose to create masks
        if use_watershed:
            masks_complete_cells = mi.CellSegmentationWatershed(np.max(tested_image[:,:,:,:,channels_cytosol[0]],
                                                                       axis=(0,1)), footprint_size=2, ).apply_watershed()
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
                    pretrained_model_cyto_segmentation = pretrained_model_cyto_segmentation
                    ).calculate_masks()
        # Selecting the mask that is in the center of the image
        center_y = masks_complete_cells.shape[0] // 2
        center_x = masks_complete_cells.shape[1] // 2
        selected_mask_id = masks_complete_cells[center_y, center_x]
        if selected_mask_id > 0:
            masks = masks_complete_cells == selected_mask_id
        else:
            # Select the largest mask that is not the background mask (0)
            mask_labels = np.unique(masks_complete_cells)
            mask_sizes = [(label, np.sum(masks_complete_cells == label)) for label in mask_labels if label != 0]
            if mask_sizes:
                selected_mask_id = max(mask_sizes, key=lambda x: x[1])[0]
                masks = masks_complete_cells == selected_mask_id
            else:
                masks = np.zeros_like(masks_complete_cells, dtype=bool)
        # Save the mask
        masks = masks.astype(np.uint8)
        tifffile.imwrite(str(mask_file_path), masks, dtype='uint8')

    if apply_photobleaching_correction:
        file_path_photobleacing = results_folder.joinpath('photobleaching.png')
        corrected_image =  mi.Photobleaching(image_TZYXC=tested_image,mask_YX=masks, show_plot=False, mode= photobleaching_mode,plot_name=file_path_photobleacing).apply_photobleaching_correction() #mi.PhotobleachingCorrection(tested_image).apply_correction()

    else:
        corrected_image = tested_image
    # Calculate the threshold for spot detection
    plot_name_threshold = results_folder.joinpath('threshold_spot_detection.png')
    
    if particle_detection_threshold is None:
        starting_threshold = mi.Utilities().calculate_threshold_for_spot_detection(
            corrected_image, list_spot_size_px, list_voxels, channels_spots,
            max_spots_for_threshold=max_spots_for_threshold,
            show_plot=True,plot_name=plot_name_threshold
        )
    else:
        starting_threshold =  [particle_detection_threshold]*len(channels_spots)
    
    # Run the particle tracking
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
            cluster_radius_nm = cluster_radius_nm,
            separate_clusters_and_spots=separate_clusters_and_spots,
            maximum_range_search_pixels=maximum_range_search_pixels,
        ).run()
    except Exception as e:
        print(f'Error: {e}')
        return pd.DataFrame(), masks, original_tested_image, None
    #df_tracking = list_dataframes_trajectories[0]
    df_tracking = list_dataframes_trajectories[0]

    if len(df_tracking)==0:
        return pd.DataFrame(), masks, original_tested_image, None


    if len(list_dataframes_trajectories) > 1:
        for i in range(1, len(list_dataframes_trajectories)):
            df_tracking = pd.concat([df_tracking, list_dataframes_trajectories[i]], ignore_index=True)
    df_tracking = df_tracking.reset_index(drop=True)    
    #print(df_tracking)
    # Plot histograms for the SNR
    selected_field = 'snr'  # options are: psf_sigma, snr, 'spot_int'
    plot_name_snr = results_folder.joinpath('spots_' + selected_field + '.png')
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
    # Plot histograms for the spot intensity
    selected_field = 'spot_int'
    plot_name_int = results_folder.joinpath('spots_' + selected_field + '.png')
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
            selected_field=selected_field + '_ch_' + str(channels_spots[0]),
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
        #plot_name_histogram = results_folder.joinpath('pixel_histogram_in_cell.png')
        plot_name_histogram = results_folder.joinpath('pixel_histogram_in_cell_'+str(channels_spots[i])+'.png')
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
    suptitle = f'Image: {data_folder_path.stem[:16]} - {list_names[selected_image]} - Cell_ID: {selected_image}'
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
        min_max_percentile=[0.05, 99.95],
        show_gird=False,
        save_plots=True,
        plot_name=plot_name_original_image_and_tracks
    )
    
    # Combine the original image and the image with tracks
    plot_name_complete_image = results_folder.joinpath('complete_image_tracking.png')
    mi.Utilities().combine_images_vertically([plot_name_original, plot_name_original_image_and_tracks], plot_name_complete_image, delete_originals=True)

    # Save the DataFrame
    df_tracking.to_csv(results_folder.joinpath('tracking_results.csv'), index=False)
    PLOT_FILTERED_IMAGES = True
    normalize_each_particle = True 
    crop_size = yx_spot_size_in_px + 5  # 3 pixels for the border
    # add 5 pixels to crop_size, check if the crop_size is odd, if not, add 1
    if crop_size % 2 == 0:
        crop_size += 1
    selected_time_point = None
    #if PLOT_FILTERED_IMAGES:
    filtered_image = mi.Utilities().gaussian_laplace_filter_image(corrected_image, list_spot_size_px, list_voxels)
    croparray_filtered, mean_crop_filtered, first_appearance, crop_size = mi.CropArray(image=filtered_image, df_crops=df_tracking, crop_size=crop_size, remove_outliers=False, max_percentile=99.95,selected_time_point=selected_time_point,normalize_each_particle=normalize_each_particle).run()
    #else:
    #    croparray_filtered, mean_crop_filtered, first_appearance, crop_size = mi.CropArray(image=tested_image, df_crops=df_tracking, crop_size=crop_size, remove_outliers=False, max_percentile=99.9,selected_time_point=selected_time_point,normalize_each_particle=normalize_each_particle).run()
    # Plot all crops  
    if save_croparray:  
        path_crop_array = results_folder.joinpath('crop_array.png')
        mi.Plots().plot_croparray(croparray_filtered, crop_size, save_plots=True,plot_name= path_crop_array,suptitle=None,show_particle_labels=True, cmap='binary_r',max_percentile = 99) # flag_vector=flag_vector
    # plot pair of crops
    plot_name_crops_filter = results_folder.joinpath('crops.png')
    mi.Plots().plot_matrix_pair_crops (mean_crop_filtered, crop_size,save_plots=True,plot_name=plot_name_crops_filter) # flag_vector=flag_vector
    # Calculate the Mean Squared Displacement
    plot_name_MSD = results_folder.joinpath('MSD_plot.png')
    
    #max_lag_for_MSD = 30
    if image_time_interval is None:
        image_time_interval = step_size_in_sec
        print(f'Warning: The image_time_interval was not provided. Using the step_size_in_sec as the image_time_interval: {step_size_in_sec} seconds.')
    else:
        image_time_interval = float(image_time_interval)
        # print a warning message indicating that we are using the step_size_in_sec as the image_time_interval
    
    if calculate_MSD:
        diffusion_coefficient, em, time_range, model_fit, trackpy_df = mi.ParticleMotion(df_tracking,
                                                                                         microns_per_pixel=pixel_xy_um,
                                                                                         step_size_in_sec=image_time_interval,
                                                                                         max_lagtime=max_lag_for_MSD,
                                                                                         show_plot=True,
                                                                                         remove_drift=False,
                                                                                         plot_name=plot_name_MSD).calculate_msd()
    else:
        diffusion_coefficient = None

    
    if calculate_correlations:
        # calculate and plot the autocorrelation
        array_ch0= mi.Utilities().df_trajectories_to_array(dataframe=df_tracking, selected_field='spot_int_ch_0', fill_value='nans')
        
        if 'spot_int_ch_1' in df_tracking.columns:
            array_ch1= mi.Utilities().df_trajectories_to_array(dataframe=df_tracking, selected_field='spot_int_ch_1', fill_value='nans') 
            intensity_array_ch0_short, intensity_array_ch1_short = mi.Utilities().shift_trajectories(array_ch0, array_ch1,max_percentage_empty_data_in_trajectory=max_percentage_empty_data_in_trajectory)
        else:
            array_ch1 = None
            intensity_array_ch0_short = mi.Utilities().shift_trajectories(array_ch0,max_percentage_empty_data_in_trajectory=max_percentage_empty_data_in_trajectory)
            intensity_array_ch1_short = None

        plot_name_intensity_matrix = results_folder.joinpath('intensity_matrix.png')
        mi.Plots().plot_matrix_sample_time(intensity_array_ch0_short, intensity_array_ch1_short,plot_name=plot_name_intensity_matrix)

        plot_name_AC_ch0 = results_folder.joinpath('AC_plot_ch0.png')
        mean_correlation_ch0, std_correlation_ch0, lags_ch0, correlations_array_ch0,dwell_time_ch0 = mi.Correlation(primary_data=intensity_array_ch0_short, max_lag=None, 
                                                                                                                nan_handling='ignore',shift_data=True,return_full=False,
                                                                                                                time_interval_between_frames_in_seconds=image_time_interval,
                                                                                                                show_plot=True,start_lag=1,fit_type='exponential',
                                                                                                                use_linear_projection_for_lag_0=True,save_plots=True,plot_name=plot_name_AC_ch0).run()
        if array_ch1 is not None:
            plot_name_AC_ch1 = results_folder.joinpath('AC_plot_ch1.png')
            mean_correlation_ch1, std_correlation_ch1, lags_ch1, correlations_array_ch1,dwell_time_ch1 = mi.Correlation(primary_data=intensity_array_ch1_short, max_lag=None, 
                                                                                                                    nan_handling='ignore',shift_data=True,return_full=False,
                                                                                                                    time_interval_between_frames_in_seconds=image_time_interval,
                                                                                                                    show_plot=True,start_lag=1,fit_type='exponential',
                                                                                                                    use_linear_projection_for_lag_0=True,save_plots=True,plot_name=plot_name_AC_ch1).run()
            
            # Plot cross-correlation
            plot_name_cross_correlation = results_folder.joinpath('cross_correlation.png')
            mean_cross_correlation, std_cross_correlation, lags_cross_correlation, cross_correlations_array, max_lag = mi.Correlation(primary_data=intensity_array_ch0_short, secondary_data=intensity_array_ch1_short, 
                                                                                                                            max_lag=None, nan_handling='ignore', shift_data=False, return_full=True,
                                                                                                                            time_interval_between_frames_in_seconds=image_time_interval,show_plot=True,
                                                                                                                            save_plots=True,plot_name=plot_name_cross_correlation).run()
        

    # plot napari visualizer
    if save_3d_visualization:
        mask_expanded = masks[np.newaxis, np.newaxis, :, :, np.newaxis]
        masked_image_TZYXC = filtered_image * mask_expanded
        # Apply Gaussian filter to reduce background noise
        #from scipy.ndimage import gaussian_filter
        masked_image_TZYXC = gaussian_filter(masked_image_TZYXC, sigma=1)
        # Remove extreme values from the image
        masked_image_TZYXC = mi.RemoveExtrema(masked_image_TZYXC, min_percentile=0.001, max_percentile=99.995).remove_outliers()
        plot_name_3d_visualizer = str(results_folder.joinpath('image_3d.gif'))
        mi.Plots().Napari_Visualizer(masked_image_TZYXC, df_tracking, z_correction=7, channels_spots=0, plot_name=plot_name_3d_visualizer)
    
    # print the process has finished for the selected image
    print(f'Image {list_names[selected_image]} has been processed.')
    
    return df_tracking, masks, original_tested_image, diffusion_coefficient
