# work on a code to count spots but not link trajectories.
import sys; from pathlib import Path
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))
from imports import *

def pipeline_particle_detection(data_folder_path, selected_image, channels_spots, max_spots_for_threshold=100000,
                               show_plot=True, channels_cytosol=None, channels_nucleus=None,
                               min_length_trajectory=5, yx_spot_size_in_px=3, z_spot_size_in_px=2,maximum_spots_cluster=4,
                               MINIMAL_SNR=0.5, diameter_cytosol=300, diameter_nucleus=200,particle_detection_threshold=None,
                               segmentation_selection_metric='max_area',recalculate_mask=False,optimization_segmentation_method='diameter_segmentation',
                               pretrained_model_cyto_segmentation=None,use_watershed=False,list_images_to_process=None,save_3d_visualization=False,
                               apply_photobleaching_correction=False,use_maximum_projection=False,link_particles=False):
    # detect if the data is a lif file
    list_images, list_names, pixel_xy_um, voxel_z_um, channel_names, number_color_channels, list_time_intervals, bit_depth = \
        mi.ReadLif(data_folder_path, show_metadata=False, save_tif=False, save_png=False, format='TZYXC').read()
    # if list_images_to_process is not None, select only the images with list_names in the list.
    if list_images_to_process is not None:
        selected_indices = [i for i in range(len(list_names)) if list_names[i] in list_images_to_process]
        # Filter all lists using the selected indices
        list_images = [list_images[i] for i in selected_indices]
        list_names = [list_names[i] for i in selected_indices]
        list_time_intervals = [list_time_intervals[i] for i in selected_indices]

    # if channels_spots
    # expanding the 
    # If selected_image is None, process all images
    if selected_image is None:
        list_df, list_masks, list_images_tested = [], [], []
        for idx in range(len(list_images)):
            df, masks,image = process_single_image(
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
                MINIMAL_SNR=MINIMAL_SNR,
                diameter_cytosol=diameter_cytosol,
                diameter_nucleus=diameter_nucleus,
                segmentation_selection_metric=segmentation_selection_metric,
                list_images=list_images,
                list_names=list_names[idx],
                pixel_xy_um=pixel_xy_um,
                voxel_z_um=voxel_z_um,
                channel_names=channel_names,
                list_time_intervals=list_time_intervals[idx],
                recalculate_mask=recalculate_mask,
                optimization_segmentation_method=optimization_segmentation_method,
                pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation,
                use_watershed=use_watershed,
                save_3d_visualization=save_3d_visualization,
                apply_photobleaching_correction=apply_photobleaching_correction,
                use_maximum_projection=use_maximum_projection,
                link_particles=link_particles,
                particle_detection_threshold=particle_detection_threshold,
            )
            # rename the field image_id to idx
            df['image_id'] = idx
            list_df.append(df)
            list_masks.append(masks)
            list_images_tested.append(image)
        if len(list_df) >1 :
            final_df = pd.concat(list_df, ignore_index=True)
        else:
            final_df = df
        return final_df, list_df, list_masks, list_images_tested
    else:
        # Process single image
        df,masks,image = process_single_image(
            data_folder_path=data_folder_path,
            selected_image=selected_image,
            channels_spots=channels_spots,
            max_spots_for_threshold=max_spots_for_threshold,
            show_plot=show_plot,
            channels_cytosol=channels_cytosol,
            channels_nucleus=channels_nucleus,
            min_length_trajectory=min_length_trajectory,
            yx_spot_size_in_px=yx_spot_size_in_px,
            maximum_spots_cluster=maximum_spots_cluster,
            MINIMAL_SNR=MINIMAL_SNR,
            diameter_cytosol=diameter_cytosol,
            diameter_nucleus=diameter_nucleus,
            segmentation_selection_metric=segmentation_selection_metric,
            list_images=list_images,
            list_names=list_names[selected_image],
            pixel_xy_um=pixel_xy_um,
            voxel_z_um=voxel_z_um,
            channel_names=channel_names,
            list_time_intervals = list_time_intervals[selected_image],
            recalculate_mask=recalculate_mask,
            optimization_segmentation_method=optimization_segmentation_method,
            pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation,
            use_watershed=use_watershed,
            save_3d_visualization=save_3d_visualization,
            apply_photobleaching_correction=apply_photobleaching_correction,
            use_maximum_projection=use_maximum_projection,
            link_particles=link_particles,  
            particle_detection_threshold=particle_detection_threshold,
        )
        return df, [df], [masks], [image]


@mi.Utilities().metadata_decorator(metadata_folder_func=mi.Utilities().get_metadata_folder,exclude_args=['list_images',]) # exclude_args=['list_images', 'list_names' ]
def process_single_image(data_folder_path, selected_image, channels_spots, max_spots_for_threshold=100000,
                         show_plot=True, channels_cytosol=None, channels_nucleus=None,
                         min_length_trajectory=5, yx_spot_size_in_px=3,  z_spot_size_in_px=2 ,maximum_spots_cluster=4,
                         MINIMAL_SNR=0.5, diameter_cytosol=300, diameter_nucleus=200, segmentation_selection_metric='area',
                         list_images=None, list_names=None, pixel_xy_um=None, voxel_z_um=None,
                         channel_names=None, optimization_segmentation_method='diameter_segmentation',
                         recalculate_mask=False,use_watershed=False, pretrained_model_cyto_segmentation=None,particle_detection_threshold=None,
                         list_time_intervals=None,save_3d_visualization=False,apply_photobleaching_correction=False,use_maximum_projection=False,link_particles=False):
    # Ensure lists are properly formatted
    channels_spots = [channels_spots] if not isinstance(channels_spots, list) else channels_spots
    channels_cytosol = [channels_cytosol] if not isinstance(channels_cytosol, list) else channels_cytosol
    channels_nucleus = [channels_nucleus] if not isinstance(channels_nucleus, list) else channels_nucleus    

    # Convert pixel and voxel sizes to nm
    pixel_xy_nm = int(pixel_xy_um * 1000)
    voxel_z_nm = int(voxel_z_um * 1000)
    list_voxels = [voxel_z_nm, pixel_xy_nm]
    list_spot_size_px = [z_spot_size_in_px, yx_spot_size_in_px]

    # Selecting the image to be analyzed
    tested_image = list_images[selected_image]  # TZYXC
    original_tested_image = tested_image.copy()
    # Creating the results folder
    results_name = 'results_' + data_folder_path.stem + '_cell_id_' + str(selected_image)
    current_dir = pathlib.Path().absolute()
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
                                                                       axis=(0,1)), 
                                                                        footprint_size=5,
                                                                        threshold_method='li',
                                                                        markers_method='distance',
                                                                        separation_size=5 ).apply_watershed()
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
        corrected_image =  mi.Photobleaching(image_TZYXC=tested_image,mask_YX=masks, show_plot=False, mode='inside_cell',plot_name=file_path_photobleacing).apply_photobleaching_correction() #mi.PhotobleachingCorrection(tested_image).apply_correction()
    else:
        corrected_image = tested_image
    # Calculate the threshold for spot detection
    plot_name_threshold = results_folder.joinpath('threshold_spot_detection.png')
    if particle_detection_threshold is None:
        starting_threshold = mi.Utilities().calculate_threshold_for_spot_detection(
                                        corrected_image, list_spot_size_px, list_voxels, channels_spots,
                                        max_spots_for_threshold=max_spots_for_threshold,
                                        show_plot=show_plot,plot_name=plot_name_threshold
                                    )
    else:
        starting_threshold = [particle_detection_threshold]*len(channels_spots)

    
    # Run the particle tracking
    list_dataframes_trajectories, _ = mi.ParticleTracking(
        image=corrected_image,
        channels_spots=channels_spots,
        masks=masks,
        list_voxels=list_voxels,
        #list_psfs=list_psfs,
        channels_cytosol=channels_cytosol,
        channels_nucleus=channels_nucleus,
        min_length_trajectory=min_length_trajectory,
        threshold_for_spot_detection=starting_threshold,
        yx_spot_size_in_px=yx_spot_size_in_px,
        maximum_spots_cluster=maximum_spots_cluster,
        link_particles = link_particles
    ).run()
    #df_tracking = list_dataframes_trajectories[0]
    df_tracking = list_dataframes_trajectories[0]
    if len(list_dataframes_trajectories) > 1:
        for i in range(1, len(list_dataframes_trajectories)):
            df_tracking = pd.concat([df_tracking, list_dataframes_trajectories[i]], ignore_index=True)
    df_tracking = df_tracking.reset_index(drop=True)
    #df_tracking['particle'] = df_tracking.groupby('particle').ngroup()

    #threshold_tracking = starting_threshold
    
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
        #list_colors=channel_names,
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
        #list_colors=channel_names,
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
    for i in range(len(channels_spots)):
        plot_name_histogram = results_folder.joinpath('pixel_histogram_in_cell_'+str(channels_spots[i])+'.png')
        masked_data = corrected_image * masks[np.newaxis, np.newaxis, :, :, np.newaxis].astype(float)
        mi.Plots().plot_image_pixel_intensity_distribution(
            image=np.mean(masked_data, axis=(0, 1)),
            figsize=(8, 2),
            bins=100,
            remove_outliers=True,
            remove_zeros=True,
            save_plots=True,
            plot_name=plot_name_histogram,
            single_color=None,
            #list_colors=channel_names,
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
        use_maximum_projection=False,
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
    
    # PLOT_FILTERED_IMAGES = True
    filtered_image = mi.Utilities().gaussian_laplace_filter_image(corrected_image, list_spot_size_px, list_voxels)
   
    # plot image and detected spots
    selected_color_channel = 0
    selected_time_point = 0 
    # select only the first time point from the dataframe
    mask_expanded = masks[np.newaxis, np.newaxis, :, :, np.newaxis]
    masked_image_TZYXC = corrected_image * mask_expanded
    df_tracking_plot = df_tracking[df_tracking['frame']==selected_time_point]
    plot_name_detected_particles = results_folder.joinpath('detected_particles.png')

    time_point = 0
    zoom_size=15
    selected_spot =0
    time_point = 0
    mi.Plots().plot_cell_zoom_selected_crop(image_TZYXC=masked_image_TZYXC, 
                        df=df_tracking,
                        use_gaussian_filter = True, 
                        image_name=plot_name_detected_particles,
                        microns_per_pixel=pixel_xy_um,
                        time_point = time_point,
                        list_channel_order_to_plot=[0,1,2], 
                        list_max_percentile=[99.9,99.9], 
                        min_percentile=1,
                        save_image=False,
                        show_spots_ids=False,
                        zoom_size=zoom_size,
                        selected_spot=selected_spot)
    # plot napari visualizer
    if save_3d_visualization:
        mask_expanded = masks[np.newaxis, np.newaxis, :, :, np.newaxis]
        masked_image_TZYXC = filtered_image * mask_expanded
        # Remove extreme values from the image
        masked_image_TZYXC = mi.RemoveExtrema(masked_image_TZYXC,min_percentile=0.001, max_percentile=99.995).remove_outliers() 
        plot_name_3d_visualizer = str(results_folder.joinpath('image_3d.gif'))
        mi.Plots().Napari_Visualizer(masked_image_TZYXC, df_tracking, z_correction=7,channels_spots=0,plot_name=plot_name_3d_visualizer)
    return df_tracking, masks, original_tested_image
