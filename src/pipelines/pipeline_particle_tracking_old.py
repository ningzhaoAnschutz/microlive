import sys; from pathlib import Path
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))
from imports import *




def metadata_decorator(metadata_folder_func=None, metadata_filename=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Capture function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            # Build metadata dictionary
            metadata = {}
            # Add Date and Time
            metadata["Date and Time"] = pd.Timestamp.now().round('min')
            # Add computer user name
            metadata["computer_user_name"] = getpass.getuser()
            # Add function arguments
            for name, value in bound_args.arguments.items():
                metadata[name] = value

            try:
                # Call the original function
                result = func(*args, **kwargs)
                #metadata["Function Result"] = str(result)
            except Exception as e:
                # Capture exception details
                metadata["Exception"] = str(e)
                metadata["Traceback"] = traceback.format_exc()
                raise  # Re-raise the exception after logging
            finally:
                # Determine metadata folder path
                if metadata_folder_func is not None:
                    metadata_folder = metadata_folder_func(*args, **kwargs)
                else:
                    metadata_folder = 'temp_metadata_folder'  # default folder

                # Create metadata folder if it doesn't exist
                metadata_folder_path = Path(metadata_folder)
                metadata_folder_path.mkdir(parents=True, exist_ok=True)

                # Customize metadata file name
                if metadata_filename:
                    metadata_file = f"{metadata_filename}.txt"
                else:
                    metadata_file = f"{func.__name__}_metadata.txt"
                metadata_file_path = metadata_folder_path / metadata_file

                # Write metadata to text file
                with open(metadata_file_path, 'w') as f:
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")

            return result
        return wrapper
    return decorator


def get_metadata_folder(*args, **kwargs):
    # Extract data_folder_path and selected_image from args or kwargs
    sig = inspect.signature(pipeline_particle_tracking)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    data_folder_path = Path(bound_args.arguments['data_folder_path'])
    selected_image = bound_args.arguments['selected_image']
    results_name = 'results_' + data_folder_path.stem + '_cell_id_' + str(selected_image)
    current_dir = Path().absolute()
    results_folder = current_dir.joinpath('results_live_cell', results_name)
    return results_folder


#@metadata_decorator(metadata_folder)
@metadata_decorator(metadata_folder_func=get_metadata_folder)
def pipeline_particle_tracking(data_folder_path, selected_image, channels_spots, max_spots_for_threshold = 100000, show_plot = True, channels_cytosol = None, channels_nucleus = None, min_length_trajectory = 5, yx_spot_size_in_px = 3, maximum_spots_cluster = 4, MINIMAL_SNR = 0.5,diameter_cytosol = 300, diameter_nucleus = 200,segmentation_selection_metric = 'area'):
    # if channels_spots is not a list make it a list.
    channels_spots = [channels_spots] if not isinstance(channels_spots, list) else channels_spots
    # if channels_cytosol is not a list make it a list.
    channels_cytosol = [channels_cytosol] if not isinstance(channels_cytosol, list) else channels_cytosol
    # if channels_nucleus is not a list make it a list.
    channels_nucleus = [channels_nucleus] if not isinstance(channels_nucleus, list) else channels_nucleus
    list_images, list_names, pixel_xy_um, voxel_z_um, channel_names, number_color_channels,list_time_intervals, bit_depth = mi.ReadLif(data_folder_path,show_metadata=False,save_tif=False,save_png=False,format='TZYXC').read()
    #  Converting the pixel and voxel sizes to nm
    pixel_xy_nm = int(pixel_xy_um*1000)
    voxel_z_nm = int(voxel_z_um*1000)
    list_voxels =   [voxel_z_nm , pixel_xy_nm]  # , [voxel_z_nm , pixel_xy_nm] ]
    list_psfs =  [voxel_z_nm ,pixel_xy_nm]  #, [voxel_z_nm , pixel_xy_nm] ]
    
    # Selecting the image to be analyzed
    tested_image =list_images[selected_image] # TZYXC
    # Creating the results folder
    results_name = 'results_'+data_folder_path.stem + '_cell_id_'+str(selected_image)
    current_dir = pathlib.Path().absolute()
    results_folder = current_dir.joinpath('results_live_cell',results_name)
    results_folder.mkdir(parents=True, exist_ok=True)
    mi.Utilities().clear_folder_except_substring(results_folder, 'mask')
    # Plot the original image
    plot_name = results_folder.joinpath('original_image.png')
    mi.Plots().plot_images(image_ZYXC=tested_image[0],figsize=(12, 5), show_plot=show_plot, use_maximum_projection=True, use_gaussian_filter=True,cmap='binary',min_max_percentile=[0.5,99.9],show_gird=False,save_plots=True,plot_name=plot_name)
    
    # Read the tif files containing the masks as numpy arrays
    mask_file_name = 'mask_'+data_folder_path.stem+'_image_'+str(selected_image)+'.tif'
    if os.path.exists(str(results_folder.joinpath(mask_file_name))):
        masks = imread(str(results_folder.joinpath(mask_file_name))).astype(bool)
    else: # use cellpose to create the masks
        masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = mi.CellSegmentation(tested_image[0], channels_cytosol = channels_cytosol, channels_nucleus= channels_nucleus, diameter_cytosol = diameter_cytosol, diameter_nucleus = diameter_nucleus, optimization_segmentation_method='diameter_segmentation', remove_fragmented_cells=False, show_plot= show_plot, image_name = None,NUMBER_OF_CORES=1,selection_metric=segmentation_selection_metric).calculate_masks()
        # Selecting the mask that is in the center of the image
        selected_mask_id = masks_complete_cells[masks_complete_cells.shape[0] // 2, masks_complete_cells.shape[1] // 2]
        if selected_mask_id > 0:
            masks = masks_complete_cells == selected_mask_id
        else:
            # select the largest mask that is not the background mask (0).
            mask_sizes = [np.sum(mask) for mask in masks_complete_cells]
            selected_mask_id = np.argmax(mask_sizes[1:]) + 1
            masks = masks_complete_cells == selected_mask_id
        # Save the mask
        masks = masks.astype(np.uint8)
        tifffile.imwrite(str(results_folder.joinpath(mask_file_name)), masks,dtype='uint8',)

    # Calculate the threshold for spot detection
    starting_threshold = mi.Utilities().calculate_threshold_for_spot_detection(tested_image,list_psfs,list_voxels,channels_spots, max_spots_for_threshold = max_spots_for_threshold, show_plot=show_plot)
    # Run the particle tracking
    list_dataframes_trajectories, _ = mi.ParticleTracking (image=tested_image,channels_spots=channels_spots, masks=masks, list_voxels=list_voxels,list_psfs=list_psfs, channels_cytosol=channels_cytosol,channels_nucleus=channels_nucleus,min_length_trajectory=min_length_trajectory,threshold_for_spot_detection=starting_threshold,yx_spot_size_in_px=yx_spot_size_in_px,maximum_spots_cluster=maximum_spots_cluster).run()
    df_tracking= list_dataframes_trajectories[0]
    threshold_tracking = starting_threshold
    filtered_image = mi.Utilities().gaussian_laplace_filter_image(tested_image,list_psfs,list_voxels)
    # Plot histigrams for the SNR
    selected_field = 'snr'  # options are: psf_sigma, snr, 'spot_int'
    plot_name_snr = results_folder.joinpath('spots_'+selected_field+'.png')
    mean_snr = mi.Plots().plot_histograms_from_df(df_tracking, selected_field=selected_field,figsize=(8,2), plot_name=plot_name_snr, bin_count=60, save_plot=True, list_colors= channel_names,remove_outliers=True)
    # Plotting histograms for the spot intensity
    selected_field = 'spot_int'  # options are: psf_sigma, snr, 'spot_int'
    plot_name_int = results_folder.joinpath('spots_'+selected_field+'.png')
    mean_snr = mi.Plots().plot_histograms_from_df(df_tracking, selected_field=plot_name_int,figsize=(8,2), plot_name=plot_name_snr, bin_count=60, save_plot=True, list_colors= channel_names,remove_outliers=True)
    # Remove tracks with low SNR in the tracking channel
    if MINIMAL_SNR is not None:
        # filtering the tracks based on the SNR on a selected channel [channels_spots]
        array_selected_field= mi.Utilities().df_trajectories_to_array(dataframe=df_tracking, selected_field=selected_field+'_ch_'+str(channels_spots[0]), fill_value='nans') 
        mean_snr = np.nanmean(array_selected_field, axis=1)
        indices_low_quality_tracks = np.where(mean_snr <  MINIMAL_SNR)[0]
        df_tracking = df_tracking[~df_tracking['particle'].isin(indices_low_quality_tracks)]
        df_tracking = df_tracking.reset_index(drop=True)
        df_tracking['particle'] = df_tracking.groupby('particle').ngroup()
    # plot image intensity histogram
    plot_name_histogram = results_folder.joinpath('pixel_histogram_in_cell.png')
    masked_data = tested_image * masks[np.newaxis, np.newaxis, :, :, np.newaxis].astype(float)
    list_median_intensity = mi.Plots().plot_image_pixel_intensity_distribution(image=np.mean(masked_data[:,:,:,:,:],axis=0),figsize=(8, 2),bins=100,remove_outliers=True,remove_zeros=True,save_plots=True, plot_name=plot_name_histogram ,single_color =None,list_colors=channel_names,tracking_channel = channels_spots[0],threshold_tracking=threshold_tracking)
    suptitle = 'Image: ' + data_folder_path.stem[:16]+'- '+list_names[selected_image] +' - Cell_ID: '+ str(selected_image)
    plot_name_original_image_and_tracks = results_folder.joinpath('original_image_tracking.png')
    mi.Plots().plot_images(image_ZYXC=tested_image[0], df=df_tracking, masks=masks, show_trajectories=True, suptitle=suptitle,figsize=(12, 3), show_plot=True,selected_time=0, use_maximum_projection=True, use_gaussian_filter=True,cmap='binary',min_max_percentile=[0.05,99.95],show_gird=False,save_plots=True,plot_name=plot_name_original_image_and_tracks)
    return df_tracking 