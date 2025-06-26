import sys; from pathlib import Path
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))
from imports import *


def metadata_folding_efficiency(filename,computer_user_name,original_lif_name, SNR_SELECTION_FOR_CHANNEL_1, SNR_SELECTION_FOR_CHANNEL_0, MIN_LEN_TRAJECTORY, MEMORY, 
                  SPOT_SIZE_PX, PLOT_FILTERED_IMAGES, MIN_INTENSITY_FOR_BACKGROUND, MIN_SPOTS_FOR_BACKGROUND,use_max_tem_projection_for_plotting,
                    max_spots_for_threshold, channels_cytosol, channels_nucleus, pixel_xy_um, voxel_z_um, channel_for_tracking, channel_folding,
                    CROP_SIZE_PX, max_crops_to_display, selected_time_point, list_quality_text,maximum_spots_cluster,ml_threshold,use_ml_for_spot_clasification):
    
    metadata = {
        "computer_user_name": computer_user_name,
        "Date and Time": pd.Timestamp.now().round('min'),
        "original_lif_name": original_lif_name,
        "SNR_SELECTION_FOR_CHANNEL_1": SNR_SELECTION_FOR_CHANNEL_1,
        "SNR_SELECTION_FOR_CHANNEL_0": SNR_SELECTION_FOR_CHANNEL_0,
        "MIN_LEN_TRAJECTORY": MIN_LEN_TRAJECTORY,
        "MEMORY": MEMORY,
        "SPOT_SIZE_PX": SPOT_SIZE_PX,
        "PLOT_FILTERED_IMAGES": PLOT_FILTERED_IMAGES,
        "MIN_INTENSITY_FOR_BACKGROUND": MIN_INTENSITY_FOR_BACKGROUND,
        "MIN_SPOTS_FOR_BACKGROUND": MIN_SPOTS_FOR_BACKGROUND,
        "use_max_tem_projection_for_plotting": use_max_tem_projection_for_plotting,
        "max_spots_for_threshold": max_spots_for_threshold,
        "channels_cytosol": channels_cytosol,
        "channels_nucleus": channels_nucleus,
        "pixel_xy_nm": int(pixel_xy_um ),
        "voxel_z_nm": int(voxel_z_um ),
        "list_voxels": [int(voxel_z_um ), int(pixel_xy_um )],
        "list_psfs": [int(voxel_z_um ), int(pixel_xy_um )],
        "channel_for_tracking": channel_for_tracking,
        "channel_folding": channel_folding,
        "CROP_SIZE_PX": CROP_SIZE_PX,
        "max_crops_to_display": max_crops_to_display,
        "selected_time_point": selected_time_point,
        "maximum_spots_cluster": maximum_spots_cluster,
        "ml_threshold": ml_threshold,
        "use_ml_for_spot_clasification": use_ml_for_spot_clasification
    }

    with open(filename, 'w') as file:
        max_key_length = max(len(key) for key in metadata.keys())
        for key, value in metadata.items():
            file.write(f"{key.ljust(max_key_length)} : {value}\n")

        file.write("\nProcessed Files:\n")
        for index, name in enumerate(list_quality_text, start=1):
            file.write(f"{index}. {name}\n")
    return None


def pipeline_folding_efficiency(original_lif_name, list_images,list_images_names, data_folder_path, current_dir, list_psfs, list_voxels, 
                    max_spots_for_threshold, MIN_INTENSITY_FOR_BACKGROUND,MIN_SPOTS_FOR_BACKGROUND, use_max_tem_projection_for_plotting,
                      channel_for_tracking,channel_folding, channel_names,crop_size,max_crops_to_display,results_folder_summary,SNR_SELECTION_FOR_CHANNEL_0,
                      SNR_SELECTION_FOR_CHANNEL_1,selected_time_point,voxel_z_nm,channels_cytosol,channels_nucleus, PLOT_FILTERED_IMAGES,SPOT_SIZE_PX,
                      MEMORY,MIN_LEN_TRAJECTORY,low_quality_pdf,maximum_spots_cluster,ml_threshold,use_ml_for_spot_clasification,pixel_xy_um):
    list_images_quality = []
    list_results_df =[]
    date_lif = data_folder_path.stem[:8]
    construct_lif = data_folder_path.stem[9:]
    list_image_paths_for_pdf = []
    path_summary_df = results_folder_summary.joinpath('results_quantification_'+original_lif_name+ '.csv')
    path_summary_pdf= results_folder_summary.joinpath('results_quantification_'+original_lif_name+'.pdf')
    path_summary_metadata = results_folder_summary.joinpath('metadata_'+original_lif_name+'.txt')
    path_summary_wisker_plot = results_folder_summary.joinpath('results_efficiency_'+original_lif_name+ '.png')
    path_summary_croparray = results_folder_summary.joinpath('croparray_'+original_lif_name+'.pdf')
    list_quality_text = []
    list_crop_array_paths = []
    list_tracking_success = []  
    for selected_image, image_TZYXC in enumerate( list_images):
        print('Processing image:', selected_image)
        results_name = f'results_{data_folder_path.stem}_cell_id_{selected_image}'
        results_folder = current_dir.joinpath('results_folding', results_name)
        results_folder.mkdir(parents=True, exist_ok=True)
        mi.Utilities().clear_folder_except_substring(results_folder, 'mask')
        # Clean up existing files
        results_df = results_folder.joinpath('results_df.csv')
        path_efficiency_df = results_folder.joinpath('results_df_efficiency.csv')
        path_quantification_image = results_folder.joinpath('results_image.png')
        path_tracking_df = results_folder.joinpath('results_df_tracking.csv')
        path_crop_array = results_folder.joinpath('crop_array.png')
        list_crop_array_paths.append(path_crop_array)
        for path in [path_efficiency_df, path_quantification_image, path_tracking_df, results_df, path_summary_df, path_summary_pdf, path_summary_metadata, path_summary_wisker_plot,path_summary_croparray]:
            if path.exists():
                path.unlink()
        # Read the masks and calculate the threshold
        mask_file_name = f'mask_{data_folder_path.stem}_image_{selected_image}.tif'
        masks = imread(str(results_folder.joinpath(mask_file_name))).astype(bool)
        threshold_tracking = mi.Utilities().calculate_threshold_for_spot_detection(image_TZYXC, list_psfs, list_voxels, [channel_for_tracking], max_spots_for_threshold, show_plot=True)
        print('Threshold for tracking:', threshold_tracking)
        # Plot histograms and check image quality
        plot_name_histogram = results_folder.joinpath('pixel_histogram_in_cell.png')
        masked_data = image_TZYXC * masks[np.newaxis, np.newaxis, :, :, np.newaxis].astype(float)
        list_median_intensity = mi.Plots().plot_image_pixel_intensity_distribution(image=np.mean(masked_data, axis=0), figsize=(14, 3), bins=100, remove_outliers=True, remove_zeros=True, save_plots=True, plot_name=plot_name_histogram, single_color=None, list_colors=channel_names, tracking_channel=channel_for_tracking, threshold_tracking=threshold_tracking)
        # Image quality assessment
        text_image_quality = ' - [LOW QUALITY IMAGE]' if threshold_tracking < MIN_INTENSITY_FOR_BACKGROUND else ''
        image_to_plot, suptitle_suffix = (np.max(image_TZYXC, axis=0), '- Maximum time projection') if use_max_tem_projection_for_plotting else (image_TZYXC[0], '')
        suptitle = f'Image: {data_folder_path.stem[:16]}- {list_images_names[selected_image]} - Cell_ID: {selected_image} {text_image_quality} {suptitle_suffix}'
        plot_name_original_image = results_folder.joinpath('original_image.png')
        mi.Plots().plot_images(image_to_plot, df=None, masks=masks, figsize=(14, 3), suptitle=suptitle, show_plot=True, selected_time=0, use_maximum_projection=True, use_gaussian_filter=True, cmap='binary', min_max_percentile=[0.01, 99.2], show_gird=False, save_plots=True, plot_name=plot_name_original_image)
        if threshold_tracking < MIN_INTENSITY_FOR_BACKGROUND:
            path_image_quality = results_folder.joinpath('results_image_quality.png')
            mi.Utilities().combine_images_vertically([plot_name_original_image, plot_name_histogram], path_image_quality, delete_originals=True)
            list_image_paths_for_pdf.append(path_image_quality)
            list_quality_text.append(list_images_names[selected_image] + '    Rejected : Low quality')
            list_tracking_success.append(False)
        else:            
            # particle tracking
            try:
                list_dataframes_trajectories, _ = mi.ParticleTracking (image=image_TZYXC,channels_spots= [channel_for_tracking], masks=masks, memory=MEMORY ,list_voxels=list_voxels,list_psfs=list_psfs, channels_cytosol=channels_cytosol,channels_nucleus=channels_nucleus,min_length_trajectory=MIN_LEN_TRAJECTORY,threshold_for_spot_detection=threshold_tracking,yx_spot_size_in_px=SPOT_SIZE_PX,maximum_spots_cluster=maximum_spots_cluster).run()    
                df_tracking= list_dataframes_trajectories[0]    
            except:
                df_tracking = pd.DataFrame()
            if df_tracking.empty:
                list_quality_text.append(list_images_names[selected_image] + '    Rejected : No spots detected')
                list_tracking_success.append(False)
            else:
                # remove low quality tracks. those that have a SNR less a threshold
                field_for_quality = 'snr_ch_1' # 'snr_ch_1'
                array_selected_field_ch1= mi.Utilities().df_trajectories_to_array(dataframe=df_tracking, selected_field=field_for_quality, fill_value='nans')    
                mean_selected_field_quality = np.nanmean(array_selected_field_ch1, axis=1)
                indices_low_quality_tracks = np.where(mean_selected_field_quality <  SNR_SELECTION_FOR_CHANNEL_1  )[0] # SNR_SELECTION_FOR_CHANNEL_1
                # removing low quality tracks
                df_tracking = df_tracking[~df_tracking['particle'].isin(indices_low_quality_tracks)]
                df_tracking = df_tracking.reset_index(drop=True)
                df_tracking['particle'] = df_tracking.groupby('particle').ngroup()
                if df_tracking.empty:
                    list_quality_text.append(list_images_names[selected_image] + '    Rejected : No spots detected')
                    list_tracking_success.append(False)
                    continue
                else:
                    list_images_quality.append(selected_image)
                plot_name_original_image_spots = results_folder.joinpath('original_image_spots.png')
                if use_max_tem_projection_for_plotting:
                    image_to_plot = np.max(image_TZYXC,axis=0)
                    max_time_projection_title = '- Maximum time projection'
                    selected_time = None
                else:
                    image_to_plot = image_TZYXC[0]
                    max_time_projection_title = ''
                    selected_time = 0
                suptitle = 'Image: ' + data_folder_path.stem[:16]+'- '+list_images_names[selected_image] +' - Cell_ID: '+ str(selected_image) + max_time_projection_title
                mi.Plots().plot_images(image_ZYXC=image_to_plot, df=df_tracking, masks=masks,figsize=(14, 3), show_trajectories=True, suptitle=suptitle,show_plot=True,selected_time=selected_time, use_maximum_projection=True, use_gaussian_filter=True,cmap='binary',min_max_percentile=[0.01,99.2],show_gird=False,save_plots=True,plot_name=plot_name_original_image_spots)
                # crops
                selected_field = 'snr'  # options are: psf_sigma, snr, 'spot_int'
                plot_name_selected_field = results_folder.joinpath('spots_'+selected_field+'.png')
                array_selected_field_ch0= mi.Utilities().df_trajectories_to_array(dataframe=df_tracking, selected_field=selected_field+'_ch_0', fill_value='nans') 
                array_selected_field_ch1= mi.Utilities().df_trajectories_to_array(dataframe=df_tracking, selected_field=selected_field+'_ch_1', fill_value='nans')
                mi.Plots().plot_crops_properties(list_particles_arrays=[array_selected_field_ch0, array_selected_field_ch1],figsize=(15, 3),save_plots=True,plot_name=plot_name_selected_field,selection_threshold=SNR_SELECTION_FOR_CHANNEL_0, label =selected_field,list_colors=channel_names)
                # plot snr histogram
                plot_name_snr = results_folder.joinpath('histogram_snr.png')
                mean_snr = mi.Plots().plot_histograms_from_df(df_tracking, selected_field=selected_field,figsize=(8,2), plot_name=plot_name_snr, bin_count=60, save_plot=True, list_colors= channel_names,remove_outliers=True)
                # plot crops
                normalize_each_particle = True 
                if PLOT_FILTERED_IMAGES:
                    filtered_image = mi.Utilities().gaussian_laplace_filter_image(image_TZYXC,list_psfs,list_voxels)
                    croparray_filtered, mean_crop_filtered, first_appearance, crop_size = mi.CropArray(image=filtered_image, df_crops=df_tracking, crop_size=crop_size, remove_outliers=False, max_percentile=99.9,selected_time_point=selected_time_point,normalize_each_particle=normalize_each_particle).run()
                else:
                    croparray_filtered, mean_crop_filtered, first_appearance, crop_size = mi.CropArray(image=image_TZYXC, df_crops=df_tracking, crop_size=crop_size, remove_outliers=False, max_percentile=99.9,selected_time_point=selected_time_point,normalize_each_particle=normalize_each_particle).run()
                # extracting crops from the croparray
                number_particles = croparray_filtered.shape[1]//crop_size
                number_time_points = croparray_filtered.shape[0]//crop_size

                list_crops_selected_particle_all_time_points = []
                for particle_id in range(number_particles):
                    list_crops_selected_particle = []
                    for time_point in range(number_time_points):
                        crop = croparray_filtered[time_point * crop_size: (time_point + 1) * crop_size, particle_id * crop_size: (particle_id + 1) * crop_size, :]
                        list_crops_selected_particle.append(crop)
                    list_crops_selected_particle_all_time_points.append(list_crops_selected_particle)
                # detect spots in Channel 0
                if use_ml_for_spot_clasification:
                    list_crops_nomalized = mi.Utilities().normalize_crop_return_list(array_crops_YXC=mean_crop_filtered,crop_size=crop_size,selected_color_channel=channel_folding,normalize_to_255=True)
                    flag_vector = ML.predict_crops(model_ML, list_crops_nomalized,threshold=ml_threshold)              
                    #flag_vector= mi.Utilities().test_particle_presence_all_frames_with_ML(croparray=croparray_filtered,crop_size=crop_size,selected_color_channel=0,minimal_number_spots_in_time=4,ml_threshold=ml_threshold)      
                else:
                    number_crops = mean_crop_filtered.shape[0]//crop_size
                    flag_vector = np.zeros(number_crops, dtype=bool)
                    for crop_id in range(number_crops):
                        flag_vector[crop_id]= mi.Utilities().is_spot_in_crop(crop_id, crop_size=crop_size, selected_color_channel=channel_folding, array_crops_YXC=mean_crop_filtered,show_plot=False)
                plot_name_crops_filter = results_folder.joinpath('crops.png')
                mi.Plots().plot_matrix_pair_crops (mean_crop_filtered, crop_size,save_plots=True,plot_name=plot_name_crops_filter,flag_vector=flag_vector)
                # Calculating folding efficiency and saving to dataframe
                number_of_detected_particles_ch1 = array_selected_field_ch1.shape[0]
                if number_of_detected_particles_ch1 < MIN_SPOTS_FOR_BACKGROUND:
                    list_quality_text.append(list_images_names[selected_image] + '    Rejected : less than ' + str(MIN_SPOTS_FOR_BACKGROUND)+ ' spots detected')
                    list_tracking_success.append(False)
                else:
                    list_quality_text.append(list_images_names[selected_image] + '    Accepted')
                    list_tracking_success.append(True)
                    particles_above_threshold = np.sum(flag_vector) 
                    efficiency = particles_above_threshold / number_of_detected_particles_ch1
                    df_folding_efficiency = pd.DataFrame({'Series': list_images_names[selected_image],'cell_index': np.array([selected_image]),
                                                        'spots_ch1':number_of_detected_particles_ch1, 
                                                        'spots_ch0_above_ts':particles_above_threshold,
                                                        'ts_int_ch1': threshold_tracking, 
                                                        'ts_snr': SNR_SELECTION_FOR_CHANNEL_0,
                                                        'mean_snr_ch0':np.round(mean_snr[0],2),'mean_snr_ch1':np.round(mean_snr[1],2),
                                                        'median_int_ch0': np.round(list_median_intensity[0],2), 'median_int_ch1': np.round(list_median_intensity[1],2),
                                                        'efficiency':np.round(efficiency,4) })
                    df_folding_efficiency['date'] = date_lif
                    df_folding_efficiency['construct'] = construct_lif
                    # save df_tracking to csv to the results folder
                    list_results_df.append(df_folding_efficiency)
                    df_folding_efficiency.to_csv(path_efficiency_df, index=False)
                    df_tracking.to_csv(path_tracking_df, index=False)
                # plotting the complete croparray
                mi.Plots().plot_croparray(croparray_filtered, crop_size, save_plots=True,plot_name= path_crop_array,suptitle=None,show_particle_labels=True, cmap='binary_r',max_percentile = 99,flag_vector=flag_vector)
                # save the results
                mi.Utilities().combine_images_vertically([plot_name_crops_filter, plot_name_selected_field], results_folder.joinpath('results_quantification.png'), delete_originals=True)
                mi.Utilities().combine_images_vertically([plot_name_original_image,plot_name_original_image_spots, plot_name_histogram,plot_name_snr], results_folder.joinpath('results_image_quality_processed.png'), delete_originals=True)
                mi.Utilities().combine_images_vertically([results_folder.joinpath('results_image_quality_processed.png'), results_folder.joinpath('results_quantification.png')], path_quantification_image, delete_originals=True)
                list_image_paths_for_pdf.append(path_quantification_image)
    #  concatenate the final dataframes with the results
    df_quantification = pd.concat(list_results_df)
    df_quantification = df_quantification.reset_index(drop=True)
    df_quantification = df_quantification[df_quantification.columns[-2:].tolist() + df_quantification.columns[:-2].tolist()]
    df_quantification.to_csv(path_summary_df, index=False)
    # create wisker plot
    fig, ax = plt.subplots(figsize=(8, 5))
    df_quantification['location'] = 1
    boxplot = df_quantification.boxplot(column='efficiency', by='location', ax=ax, grid=False, showfliers=False,
                        boxprops=dict(color="k", linewidth=2),
                        whiskerprops=dict(color="k", linewidth=2),
                        medianprops=dict(color="orangered", linewidth=2))
    jitter = 0.02
    df_quantification['jitter'] = np.random.uniform(-jitter, jitter, df_quantification.shape[0])
    df_quantification['lif_name_jitter'] = df_quantification['location'] + df_quantification['jitter']
    scatter = ax.scatter(df_quantification['lif_name_jitter'], df_quantification['efficiency'], color='red', marker='o', edgecolor='black', s=50, alpha=0.7)
    # Customize plot aesthetics
    plt.title('Efficiency of Folding')
    plt.suptitle('')  # Remove the default suptitle
    plt.xticks([1], [original_lif_name], fontsize=14)  # Set tick labels (enclose original_lif_name in a list if it's a single string)
    plt.yticks(fontsize=14)  # Set y-tick labels
    plt.ylabel('Efficiency', fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(path_summary_wisker_plot, dpi=300, bbox_inches='tight')
    plt.show()
    # Create PDF with images and quality text
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for i, image_path in enumerate(list_image_paths_for_pdf):
        pdf.add_page()
        pdf.set_xy(10, 10)
        pdf.cell(0, 10, list_quality_text[i], 0, 1, 'L')
        if low_quality_pdf:
            img = Image.open(image_path)
            base_width = 150  # Desired width in mm in the PDF
            w_percent = (base_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))  # Height in mm to maintain aspect ratio
            # Temporarily save resized image for quality adjustment
            temp_path = Path(image_path).with_name(Path(image_path).stem + '_temp').with_suffix('.jpg')
            img.save(temp_path, 'JPEG', quality=85)  # You can adjust quality to manage file size
            pdf.image(str(temp_path), x=25, y=25, w=base_width, h=h_size)  # Now specifying both width and height
            temp_path.unlink()  # Delete the temporary file
        else:
            # Directly embed the image at specified dimensions without resizing beforehand
            img = Image.open(image_path)
            w_percent = (150 / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))  # Calculate height to maintain aspect ratio
            pdf.image(str(image_path), x=25, y=25, w=150, h=h_size)
    pdf.output(path_summary_pdf)
    
    # save metadata
    metadata_folding_efficiency(
        path_summary_metadata,
        computer_user_name=computer_user_name,
        original_lif_name=original_lif_name,
        SNR_SELECTION_FOR_CHANNEL_1=SNR_SELECTION_FOR_CHANNEL_1,
        SNR_SELECTION_FOR_CHANNEL_0=SNR_SELECTION_FOR_CHANNEL_0,
        MIN_LEN_TRAJECTORY=MIN_LEN_TRAJECTORY,
        MEMORY=MEMORY,
        SPOT_SIZE_PX=SPOT_SIZE_PX,
        PLOT_FILTERED_IMAGES=PLOT_FILTERED_IMAGES,
        MIN_INTENSITY_FOR_BACKGROUND=MIN_INTENSITY_FOR_BACKGROUND,
        MIN_SPOTS_FOR_BACKGROUND=MIN_SPOTS_FOR_BACKGROUND,
        use_max_tem_projection_for_plotting=use_max_tem_projection_for_plotting,
        max_spots_for_threshold=max_spots_for_threshold,
        channels_cytosol=channels_cytosol,
        channels_nucleus=channels_nucleus,
        pixel_xy_um=pixel_xy_um, 
        voxel_z_um=voxel_z_nm, 
        channel_for_tracking=channel_for_tracking, 
        channel_folding=channel_folding, 
        CROP_SIZE_PX=crop_size,    
        max_crops_to_display=max_crops_to_display,  
        selected_time_point=selected_time_point,
        list_quality_text = list_quality_text,
        maximum_spots_cluster = maximum_spots_cluster,
        ml_threshold = ml_threshold,
        use_ml_for_spot_clasification = use_ml_for_spot_clasification)

    return df_quantification