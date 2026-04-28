# This file is intended to be used in the command line to process FRAP data.
# The script reads the data from a lif file, segments the nuclei and background of the image, detects the roi, and quantifies the intensity of the roi.
# The script saves the results in a pdf file and a csv file with the quantification of the intensity of the roi.
# The script also fits the FRAP data to  single and double exponential models and saves the results in a pdf file and a csv file.
# The script also removes the tracking errors by removing the elements where the code is unable to detect the roi and saves the results in a pdf file and a csv file.

# To run this script in the command line, you need to provide the path to the lif file as an argument.
# An example of how to run the script in the command line is shown in : FRAP_processing.ipynb
#
# Usage:
#   python FRAP_line_command.py <data_folder> <frap_time> <stable_ch> <quant_ch> <roi_radius>
#       <save_individual> <immobile_fraction> <start_frame> <step_size> <use_frap_roi>
#       <pretrained_model|None> <roi_drop_threshold|0.6> <frame_list_csv>

import sys
from pathlib import Path
from microlive.pipelines.pipeline_FRAP import *
from microlive.imports import *
from microlive import microscopy as mi
current_dir = Path().resolve()

# Reading the parameters passed in the command line
data_folder_path = Path(sys.argv[1])
frap_time = int(sys.argv[2]) #10
stable_FRAP_channel = int(sys.argv[3]) #1
FRAP_channel_to_quantify = int(sys.argv[4]) #0
radius_roi_size_px = int(sys.argv[5])#7
save_individual_dataframes = mi.Utilities().parse_bool_or_int(sys.argv[6])#False
fit_model_considering_immobile_fraction = mi.Utilities().parse_bool_or_int(sys.argv[7]) #False
starting_changing_frame=int(sys.argv[8]) #40
step_size_increase=int(sys.argv[9]) #5
use_frap_time_for_roi_detection = mi.Utilities().parse_bool_or_int(sys.argv[10])

# Optional: pretrained segmentation model path (pass 'None' or 'auto' to use defaults)
if len(sys.argv) >= 14:
    _model_arg = sys.argv[11]
    if _model_arg.lower() in ('none', 'auto'):
        pretrained_model_segmentation = None
    else:
        pretrained_model_segmentation = _model_arg
    roi_drop_threshold = float(sys.argv[12])
elif len(sys.argv) >= 13:
    _model_arg = sys.argv[11]
    if _model_arg.lower() in ('none', 'auto'):
        pretrained_model_segmentation = None
    else:
        pretrained_model_segmentation = _model_arg
    roi_drop_threshold = 0.6
else:
    pretrained_model_segmentation = None
    roi_drop_threshold = 0.6

list_selected_frame_values_real_time = list(map(int, sys.argv[-1].split(',')))
#list_selected_frames=[0, 10, 20, 70, 100, 139]
#list_selected_frame_values_real_time =[0, 10, 50, 100, 300, 500] # this list contain the real time values of the selected frames
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx


# load the images
list_images, list_names, pixel_xy_um, voxel_z_um, channel_names, number_color_channels,list_time_intervals, bit_depth, _,_,_ = mi.ReadLif(data_folder_path,show_metadata=False,save_tif=False,save_png=False,format='TZYXC').read()
# concatenating images pre and post FRAP time
list_concatenated_images, list_names_concatenated_images, list_time_concatenated = concatenate_images(list_images, list_names,convert_to_8bit=False,list_time_intervals=list_time_intervals)
number_images = len(list_concatenated_images)
print('Total number of concatenated images:',number_images)

# creating a folder for the results
if fit_model_considering_immobile_fraction == True:
    results_folder = current_dir.joinpath('results','results_FRAP_'+data_folder_path.stem + '_immobile_fraction')
else:
    results_folder = current_dir.joinpath('results','results_FRAP_'+data_folder_path.stem)
results_folder.mkdir(parents=True,exist_ok=True)



list_combined_image_paths = []
for i in range(len(list_names_concatenated_images)):
    name_frap_plot = list_names_concatenated_images[i].replace(' ', '_')+'.png'
    name_frap_plot_quantification = list_names_concatenated_images[i].replace(' ', '_')+'quantification.png'
    combined_image_path = results_folder.joinpath('results_quantification_frap_'+str(i)+'.png')
    # This function segments the nuclei and background of the image and return the image in the correct format for the FRAP analysis
    image_TZXYC, image_TZXYC_masked, image_TXY, masks_TXY, pseudo_cytosol_masks_TXY, background_mask,frame_values = create_image_arrays(list_concatenated_images, selected_image=i,FRAP_channel_to_quantify=FRAP_channel_to_quantify,pretrained_model_segmentation=pretrained_model_segmentation,frap_time=frap_time,starting_changing_frame=starting_changing_frame, step_size_increase=step_size_increase, min_diameter=radius_roi_size_px*2)
    # Conver the used selected frames to the index of the frames
    if masks_TXY is not None:
        list_selected_frames = (  [find_nearest(frame_values, time) for time in list_selected_frame_values_real_time])
        # This function calculates the intensity of the nucleus and background of the image
        mask_intensity_nucleus, mask_intensity_background ,mask_intensity_pseudo_cytosol = calculate_mask_and_background_intensity(image_TXY, masks_TXY,background_mask, pseudo_cytosol_masks_TXY)
        #temporal_mean_intensity_nucleus, temporal_mean_intensity_background = calculate_mask_and_background_intensity(image_TXY, masks_TXY,background_mask)
    else:
        mask_intensity_nucleus = None
        mask_intensity_background = None
        mask_intensity_pseudo_cytosol = None
    if mask_intensity_nucleus is None or mask_intensity_background is None:
        mean_roi_frap = None
        print("--------------------")
        print('No mask detected')
        print("--------------------")
    else:
        # this section detects the roi in the image.
        mean_roi_frap,mean_roi_frap_normalized, coordinates_roi, df_selected_trajectory = find_frap_roi(image_TZXYC_masked,image_TZXYC, masks_TXY, frap_time, min_diameter=radius_roi_size_px*2,FRAP_channel_to_quantify=FRAP_channel_to_quantify,stable_FRAP_channel=stable_FRAP_channel,show_binary_plot=False,mask_intensity_background=mask_intensity_background,use_frap_time_for_roi_detection=use_frap_time_for_roi_detection,roi_drop_threshold=roi_drop_threshold)
    if mean_roi_frap is None:
        print("--------------------")
        print('No ROI detected')
        print("--------------------")
    else:
        subtitle = list_names_concatenated_images[i]
        plot_images_frap(image_TZXYC, list_selected_frames=list_selected_frames, subtitle=subtitle, show_grid=False, cmap='viridis',
                        selected_color_channel=FRAP_channel_to_quantify, coordinates_roi=coordinates_roi, radius_roi_size_px=radius_roi_size_px,save_plot=True,
                        plot_name=name_frap_plot,list_selected_frame_values_real_time=list_selected_frame_values_real_time,
                        masks_TXY=masks_TXY, pseudo_cytosol_masks_TXY=pseudo_cytosol_masks_TXY)

        plot_frap_quantification(frame_values, mean_roi_frap_normalized,mean_roi_frap, mask_intensity_nucleus,mask_intensity_pseudo_cytosol, mask_intensity_background, frap_time,save_plot=True, plot_name=name_frap_plot_quantification)

        mi.Utilities().combine_images_vertically([name_frap_plot, name_frap_plot_quantification], combined_image_path, delete_originals=True)

        df_tracking = pd.DataFrame({'frame': frame_values, 'mean_roi_frap_normalized': mean_roi_frap_normalized, 'mean_roi_frap': mean_roi_frap, 'mask_intensity_nucleus': mask_intensity_nucleus, 'mask_intensity_background': mask_intensity_background, 'mask_intensity_pseudo_cytosol': mask_intensity_pseudo_cytosol})
        # add a column to the dataframe with the name of the image
        df_tracking['image_name'] = list_names_concatenated_images[i]
        if save_individual_dataframes:
            # path to save the dataframe
            path_tracking_df = results_folder.joinpath('results_quantification_frap_'+str(i)+'.csv')
            df_tracking.to_csv(path_tracking_df, index=False)
        print('Processing image number:',i)
        list_combined_image_paths.append(combined_image_path)
        # concatenate all the dataframes
        if i == 0 or 'df_tracking_all' not in locals():
            df_tracking_all = df_tracking
        else:
            df_tracking_all = pd.concat([df_tracking_all, df_tracking], ignore_index=True)
# save the dataframe
path_tracking_df_all = results_folder.joinpath('df_'+data_folder_path.stem+'.csv')
if 'df_tracking_all' in locals():
    df_tracking_all.to_csv(path_tracking_df_all, index=False)
else:
    print("WARNING: No images were successfully processed — no CSV or PDF generated.")

if list_combined_image_paths:
    pdf_name = results_folder.joinpath('results_FRAP_'+data_folder_path.stem+'.pdf')
    create_pdf(list_combined_image_paths,pdf_name, remove_original_images=True)


name_plot_frap = results_folder.joinpath('results_plot_FRAP_'+data_folder_path.stem+'.png')
_, _,_ =plot_frap_quantification_all_images(df_tracking_all, save_plot=True, plot_name=name_plot_frap)


# remove tracking errors by removing the elements where the code is unable to detect the roi
threhsold_to_remove_tracking_errors=0.1  # use values between 0.05 and 0.1
df_tracking_removed_roi_no_detected= remove_cell_without_roi_detection(df_tracking_all,threhsold=threhsold_to_remove_tracking_errors)
path_tracking_df_all_removed_roi_no_detected  = results_folder.joinpath('df_'+data_folder_path.stem+'_removed_no_roi_detected.csv')
df_tracking_removed_roi_no_detected.to_csv(path_tracking_df_all_removed_roi_no_detected, index=False)

name_plot_frap_removed_not_detected_roi = results_folder.joinpath('results_plot_FRAP_'+data_folder_path.stem+'_removed_not_detected_roi.png')
frames, mean_int_values,std_values = plot_frap_quantification_all_images(df_tracking_removed_roi_no_detected, save_plot=True, plot_name=name_plot_frap_removed_not_detected_roi)


list_combined_image_paths_fits = []
plot_name_fit_average = results_folder.joinpath('results_fit_FRAP_'+data_folder_path.stem+'_removed_not_detected_roi.png')
if fit_model_considering_immobile_fraction:
    t_half_single, t_half_double_1st_process,t_half_double_2nd_process, r_squared_single, r_squared_double = fit_model_to_frap_immobile_fraction(time=frames, intensity=mean_int_values,frap_time=frap_time,suptitle='FRAP [Average Plot]' +' Fit immobile fraction', save_plot=True, plot_name=plot_name_fit_average)
else:
    t_half_single, t_half_double_1st_process,t_half_double_2nd_process, r_squared_single, r_squared_double = fit_model_to_frap(time=frames, intensity=mean_int_values,frap_time=frap_time,suptitle='FRAP [Average Plot]', save_plot=True, plot_name=plot_name_fit_average)
list_combined_image_paths_fits.append(plot_name_fit_average)
# make a copy of file plot_name_fit_average and save as plot_name_fit_average_second for the final summary report
plot_name_fit_average_second = results_folder.joinpath('copy_results_fit_FRAP_' + data_folder_path.stem + '_removed_not_detected_roi_second.png')
plot_name_fit_average_second.write_bytes(plot_name_fit_average.read_bytes())



# calculate the mean intensity for all time points for the background, nucleus and pseudo cytosol



# fit FRAP for each image
fit_results = []
for i, name in enumerate( df_tracking_removed_roi_no_detected['image_name'].unique() ):
    plot_name = results_folder.joinpath('results_fit_FRAP_'+data_folder_path.stem+'_'+name.replace(' ', '_')+'.png')
    df_selected = df_tracking_removed_roi_no_detected[df_tracking_removed_roi_no_detected['image_name'] == name]

    # calculate the mean intensity  for THE FIRST TIME POINT for the background, nucleus and pseudo cytosol IN df_selected
    mask_intensity_nucleus = df_selected['mask_intensity_nucleus'].iloc[0]
    mask_intensity_background = df_selected['mask_intensity_background'].iloc[0]
    mask_intensity_pseudo_cytosol = df_selected['mask_intensity_pseudo_cytosol'].iloc[0]


    if fit_model_considering_immobile_fraction == True:
        t_half_single, t_half_double_1st_process,t_half_double_2nd_process, r_squared_single, r_squared_double = fit_model_to_frap_immobile_fraction(time=df_selected['frame'], intensity=df_selected['mean_roi_frap_normalized'],frap_time=frap_time, suptitle=data_folder_path.stem+'_'+name.replace(' ', '_')+' [Fit considering immobile fraction]', save_plot=True, plot_name=plot_name)
    else:
        t_half_single, t_half_double_1st_process, t_half_double_2nd_process, r_squared_single, r_squared_double = fit_model_to_frap(time=df_selected['frame'], intensity=df_selected['mean_roi_frap_normalized'],frap_time=frap_time, suptitle=data_folder_path.stem+'_'+name.replace(' ', '_'), save_plot=True, plot_name=plot_name)
    plt.close( )
    fit_results.append({
    'image_name': name,
    't_half_single': f"{t_half_single:.4g}" if not math.isnan(t_half_single) else np.nan,
    'r_2_single': f"{r_squared_single:.4g}" if not math.isnan(r_squared_single) else np.nan,
    't_half_double_1st_process': f"{t_half_double_1st_process:.4g}" if not math.isnan(t_half_double_1st_process) else np.nan,
    't_half_double_2nd_process': f"{t_half_double_2nd_process:.4g}" if not math.isnan(t_half_double_2nd_process) else np.nan,
    'r_2_double': f"{r_squared_double:.4g}" if not math.isnan(r_squared_double) else np.nan,
    'intensity_nucleus_before_bleach': f"{mask_intensity_nucleus:.4g}",
    'intensity_background_before_bleach': f"{mask_intensity_background:.4g}",
    'intensity_pseudo_cytosol_before_bleach': f"{mask_intensity_pseudo_cytosol:.4g}"
    })
    df_fit = pd.DataFrame(fit_results)
    list_combined_image_paths_fits.append(plot_name)
#make all values numeric except the image name
df_fit = df_fit.apply(pd.to_numeric, errors='ignore')
# save the dataframe
plt.close('all')
# save the dataframe
path_fit_df = results_folder.joinpath('df_FRAP_fit_'+data_folder_path.stem+'.csv')
df_fit.to_csv(path_fit_df, index=False)

pdf_name = results_folder.joinpath('results_fits_FRAP_'+data_folder_path.stem+'.pdf')
create_pdf(list_combined_image_paths_fits,pdf_name,remove_original_images=True)


plot_name_fit_summary = results_folder.joinpath('summary_fits_'+data_folder_path.stem+'.png')
if fit_model_considering_immobile_fraction:
    suptitle=data_folder_path.stem+' - [Fit considering immobile fraction]'
else:
    suptitle=data_folder_path.stem
plot_t_half_values(df_fit, r2_threshold=0.8, suptitle=suptitle,save_plot=True, plot_name=plot_name_fit_summary)


list_summary_results = [plot_name_fit_summary,name_plot_frap,name_plot_frap_removed_not_detected_roi,plot_name_fit_average_second]
pdf_name_summary = results_folder.joinpath('summary_FRAP_'+data_folder_path.stem+'.pdf')
create_pdf(list_summary_results,pdf_name_summary,remove_original_images=True)
