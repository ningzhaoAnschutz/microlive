import sys; from pathlib import Path
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))
from imports import *
from skimage.feature import canny
from skimage.draw import circle_perimeter
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks


##################################################################################
##################### General functions #########################################
##################################################################################


def read_lif_files_in_folder(folder_path):
    # create funtion that read all the .lif files in a folder and return the list of images
    list_folders = list(folder_path.glob('*.lif'))
    return list_folders

def find_nearest(array, value):
    # function to find the index of the nearest value in an array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_frame_values( image_TZXYC, starting_changing_frame=40, step_size_increase=5):
    N = image_TZXYC.shape[0]
    frame_durations = np.ones(N)
    frame_durations[starting_changing_frame:] = step_size_increase
    frame_values = np.cumsum(frame_durations) - 1  # Subtract 1 to start from 0
    return frame_values

def remove_border_masks(array,min_size=50):
    """
    Remove masks that touch the border of the array and reorder the remaining masks.
    
    Parameters:
    - array (np.array): The input 2D array with masks.
    
    Returns:
    - np.array: The modified array with border-touching masks removed and remaining masks reordered.
    """
    # Get the mask values along the border of the array
    
    # Define the minimum size threshold for mask areas
    
    # removing small masks
    cleaned_image = array.copy()
    cleaned_image_after_size_threshold = np.zeros_like(array)
    labels_to_keep = np.unique(array)[1:]  # Exclude the background
    for label in labels_to_keep:
        mask_area = (cleaned_image == label).sum()  # Count the number of pixels in this mask
        if mask_area >= min_size**2:
            cleaned_image_after_size_threshold[cleaned_image == label] = label

    array = cleaned_image_after_size_threshold
    
    top_border = array[0, :]
    bottom_border = array[-1, :]
    left_border = array[:, 0]
    right_border = array[:, -1]
    # Combine all border mask values into a set
    border_masks = set(np.concatenate([top_border, bottom_border, left_border, right_border]))
    # Remove zero (background) from the border mask set
    border_masks.discard(0)
    # Set border touching masks to zero
    for mask_value in border_masks:
        array[array == mask_value] = 0
    # Reorder the remaining masks to have continuous values
    unique_masks = np.unique(array)
    unique_masks = unique_masks[unique_masks != 0]
    # Create a mapping from old mask values to new mask values
    mask_mapping = {old_mask: new_mask for new_mask, old_mask in enumerate(unique_masks, start=1)}
    # Apply the mapping to renumber the remaining masks
    for old_mask, new_mask in mask_mapping.items():
        array[array == old_mask] = new_mask
    return array



def interpolate_masks(masks, step_size, total_frames):
    interpolated_masks = np.zeros((total_frames, masks.shape[1], masks.shape[2]), dtype=masks.dtype)
    for i in range(0, total_frames, step_size):
        start_mask = masks[i // step_size]
        end_mask = masks[min((i // step_size) + 1, masks.shape[0] - 1)]
        for j in range(step_size):
            if i + j < total_frames:
                t = j / step_size
                interpolated_masks[i + j] = (1 - t) * start_mask + t * end_mask
    return interpolated_masks

def find_frap_coordinates(image_TXY, frap_time, stable_FRAP_channel, min_diameter):
    """
    Find the coordinates of the region of interest (ROI) for fluorescence recovery after photobleaching (FRAP).

    Parameters:
    image_TZXYC_masked : numpy.ndarray
        5D image array with dimensions corresponding to Time, Z-slice, X, Y, and Channel.
    frap_time : int
        Time point of the FRAP event (1-indexed).
    stable_FRAP_channel : int
        Channel index to be analyzed.
    min_diameter : float
        Minimum diameter for ROI detection in pixels.

    Returns:
    tuple
        Coordinates of the ROI center and its radius if a suitable region is found, else None.
    """
    z = 0  # Use first Z-slice, adjust if needed
    channel = stable_FRAP_channel  # The channel to use for detection
    if frap_time <= 0:
        raise ValueError("frap_time must be greater than 0 to compare frames.")

    # Extract frames before and after FRAP
    image_before_frap = image_TXY[frap_time - 1] #, :, :, channel]
    image_after_frap = image_TXY[frap_time]#, :, :, channel]

    # Compute the difference image
    diff_image = image_after_frap.astype(np.float32) - image_before_frap.astype(np.float32)
    neg_diff_image = -diff_image  # Regions with decreased intensity are positive

    # Threshold the negative difference image using Otsu's method
    threshold = threshold_otsu(neg_diff_image)
    binary_diff = neg_diff_image > threshold

    # Clean up the binary image using morphological operations
    binary_diff = binary_opening_ndi(binary_diff, structure=np.ones((3, 3)))
    binary_diff = binary_closing_ndi(binary_diff, structure=np.ones((3, 3)))

    # Label regions and extract properties
    labeled_regions = label(binary_diff)
    props = regionprops(labeled_regions)

    # Expected area based on min_diameter
    expected_area = np.pi * (min_diameter / 2) ** 2
    area_tolerance = expected_area * 0.5  # Adjust as needed

    # Initialize variables to keep track of the best region
    best_region = None
    best_circularity = 0
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter == 0:
            continue  # Avoid division by zero
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        area_difference = abs(area - expected_area)
        # Check if area is within acceptable range and circularity is high
        if area_difference <= area_tolerance and circularity > best_circularity:
            best_region = prop
            best_circularity = circularity
            break

    # Return the coordinates and radius if a suitable region was found
    if best_region:
        #print(best_region.centroid[1], best_region.centroid[0] )
        return best_region.centroid[1], best_region.centroid[0] 
    else:
        return None, None


def segment_image(image_TXY, step_size=5, pretrained_model_segmentation=None, frap_time=None, pixel_dilation_pseudo_cytosol=10,stable_FRAP_channel=0,min_diameter=10):
    num_pixels_to_dilate = 1
    use_gpu = False  # or True if you want to try MPS on Apple Silicon
    if pretrained_model_segmentation is not None:
        model_nucleus = models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=pretrained_model_segmentation
        )
    else:
        model_nucleus = models.CellposeModel(
            gpu=use_gpu,
            model_type='nuclei'
        )
    #model_cyto = models.Cellpose(gpu=False, model_type='cyto2')
    model_cyto = models.CellposeModel( gpu=use_gpu, model_type='cyto2')
    num_steps = (image_TXY.shape[0] + step_size - 1) // step_size
    list_masks = []
    list_selected_mask_id = []
    list_selected_masks = []
    list_masks_cyto = []
    # If frap_time is provided, segment the FRAP images and select the mask with maximum intensity change
    if frap_time is not None:
        # Ensure frap_time is within valid range
        if frap_time < 1 or frap_time >= image_TXY.shape[0] - 1:
            raise ValueError("frap_time must be within the range of the image stack.")
        # Segment the image at frap_time
        #masks_frap = model_nucleus.eval(image_TXY[frap_time], normalize=True, channels=[0,0], flow_threshold=0.8, diameter=150, min_size=100)[0]
        masks_frap = model_nucleus.eval(
            image_TXY[frap_time],
            channels=[0, 0],          # ← add this!
            normalize=True,
            flow_threshold=1,
            diameter=150,
            min_size=50
        )[0]
        # remove all the maks that are touching the border
        masks_frap =remove_border_masks(masks_frap,min_size=50)
        # Get unique mask labels (excluding background)
        mask_labels = np.unique(masks_frap)
        mask_labels = mask_labels[mask_labels != 0]
        if mask_labels is not None:
            for label in mask_labels:
                mask = masks_frap == label
                image_TXY_masked = image_TXY * mask
                x_coord_frap_roi, y_coord_frap_roi = find_frap_coordinates(image_TXY_masked, frap_time, stable_FRAP_channel, min_diameter=min_diameter)
                if x_coord_frap_roi is not None:
                    selected_mask_frap = mask
                    selected_mask_frap = binary_dilation(selected_mask_frap, iterations=num_pixels_to_dilate).astype('int')
                    break
            else:
                #selected_mask_id_frap = None
                selected_mask_frap = None
        else:
            #selected_mask_id_frap = None
            selected_mask_frap = None
    else:
        #selected_mask_id_frap = None
        selected_mask_frap = None
    if selected_mask_frap is None:
        return None, None, None
    
    for step in range(num_steps):
        i = step * step_size
        # Detecting masks in i-th frame
        masks = model_nucleus.eval(
            image_TXY[i],
            channels=[0, 0],         
            normalize=True,
            flow_threshold=1,
            diameter=150,
            min_size=50
        )[0]
        list_masks.append(masks)
        masks =remove_border_masks(masks,min_size=50)
        # Detect cytosol masks only every `step_size` frames
        if step % 2 == 0:
            masks_cyto = model_cyto.eval(image_TXY[i], normalize=True, flow_threshold=0.5, diameter=250, min_size=100)[0]
            list_masks_cyto.append(masks_cyto)
        if frap_time is None:
            # Selecting the mask that is in the center of the image
            selected_mask_id = masks[masks.shape[0] // 2, masks.shape[1] // 2]
            if selected_mask_id == 0 and step > 0:
                selected_mask_id = list_selected_mask_id[step - 1]
            list_selected_mask_id.append(selected_mask_id)
            selected_masks = masks == selected_mask_id
        else:
            # Find the mask in `masks` that overlaps most with `selected_mask_frap`
            labels = np.unique(masks)
            labels = labels[labels != 0]
            max_overlap = 0
            selected_mask_id = None
            for label in labels:
                current_mask = masks == label
                overlap = np.sum(current_mask & selected_mask_frap)
                if overlap > max_overlap:
                    max_overlap = overlap
                    selected_mask_id = label
            if selected_mask_id is None:
                # Use the previous selected_mask_id or default to the first label
                if step > 0:
                    selected_mask_id = list_selected_mask_id[step - 1]
                else:
                    selected_mask_id = labels[0] if len(labels) > 0 else 0
            list_selected_mask_id.append(selected_mask_id)
            selected_masks = masks == selected_mask_id
        # Dilate the selected mask
        selected_masks = binary_dilation(selected_masks, iterations=num_pixels_to_dilate).astype('int')
        list_selected_masks.append(selected_masks)
    # Ensure selected nuclear masks persist if segmentation fails in a frame
    for idx, mask in enumerate(list_selected_masks):
        if np.sum(mask) == 0:
            if idx > 0:
                list_selected_masks[idx] = list_selected_masks[idx - 1]
            else:
                # Use first non-empty mask for initial frame if needed
                for m in list_selected_masks:
                    if np.sum(m) > 0:
                        list_selected_masks[0] = m
                        break
    # Interpolating masks for frames between the steps
    masks_TXY = interpolate_masks(np.array(list_selected_masks), step_size, image_TXY.shape[0])
    # Ensure that each time point has a non-empty mask
    non_empty_indices = np.where(np.sum(masks_TXY, axis=(1, 2)) > 0)[0]
    if not non_empty_indices.size:
        raise ValueError("All masks are empty. Check the segmentation parameters.")
    for i in range(masks_TXY.shape[0]):
        if np.sum(masks_TXY[i]) == 0:
            # Find the closest non-empty mask
            closest_idx = non_empty_indices[np.argmin(np.abs(non_empty_indices - i))]
            masks_TXY[i] = masks_TXY[closest_idx]
    all_cyto_masks = interpolate_masks(np.array(list_masks_cyto), step_size * 2, image_TXY.shape[0])
    all_nucleus_masks = np.array(list_masks)
    sum_masks = binary_dilation(np.sum(all_nucleus_masks, axis=0), iterations=20).astype('int')
    sum_cyto_masks = binary_dilation(np.sum(all_cyto_masks, axis=0), iterations=20).astype('int')
    background_mask = (sum_masks + sum_cyto_masks) == 0

    # Create pseudo-cytosol masks
    pseudo_cytosol_masks_TXY = np.zeros_like(masks_TXY)
    # Dilate the nucleus masks
    dilated_nucleus_masks = binary_dilation(masks_TXY, iterations=5)
    for i in range(masks_TXY.shape[0]):   
        dilated_nucleus_masks[i] = binary_dilation(masks_TXY[i], iterations=pixel_dilation_pseudo_cytosol).astype('int') 
        # Subtract the dilated nucleus masks from the sum of all masks
        temp_pseudo_cytosol_masks_TXY =  dilated_nucleus_masks[i] - masks_TXY[i] 
        pseudo_cytosol_masks_TXY[i] = temp_pseudo_cytosol_masks_TXY > 0
    pseudo_cytosol_masks_TXY = pseudo_cytosol_masks_TXY.astype('int')
    return masks_TXY, background_mask, pseudo_cytosol_masks_TXY




def create_image_arrays(list_concatenated_images, selected_image=0, FRAP_channel_to_quantify=0,pretrained_model_segmentation=None,frap_time=None, starting_changing_frame=40, step_size_increase=5,min_diameter=10):
    image_TZXYC = list_concatenated_images[selected_image] # shape (T Z Y X C)
    print('Image with shape (T Z Y X C):\n ' ,list_concatenated_images[selected_image].shape) # TZYXC
    print('Original Image pixel ', 'min: {:.2f}, max: {:.2f}, mean: {:.2f}, std: {:.2f}'.format(np.min(image_TZXYC), np.max(image_TZXYC), np.mean(image_TZXYC), np.std(image_TZXYC)) )
    
    
    image_TXY = image_TZXYC[:,0,:,:,FRAP_channel_to_quantify] # shape (T X Y)
    image_TXY_8bit = (image_TXY - np.min(image_TXY)) / (np.max(image_TXY) - np.min(image_TXY)) * 255

    # create image_TXY_8bit_stable_FRAP_channel
    if FRAP_channel_to_quantify == 0:
        stable_FRAP_channel = 1
    else:
        stable_FRAP_channel = 0
    image_TXY_stable_FRAP = image_TZXYC[:,0,:,:,stable_FRAP_channel] # shape (T X Y)
    image_TXY_stable_FRAP_8bit = (image_TXY_stable_FRAP - np.min(image_TXY_stable_FRAP)) / (np.max(image_TXY_stable_FRAP) - np.min(image_TXY_stable_FRAP)) * 255

    masks_TXY, background_mask, pseudo_cytosol_masks_TXY = segment_image(image_TXY_stable_FRAP_8bit, step_size=5, pretrained_model_segmentation=pretrained_model_segmentation,frap_time=frap_time,stable_FRAP_channel=FRAP_channel_to_quantify,min_diameter=min_diameter)
   
    if masks_TXY is None:
        return None, None, None, None, None, None, None
    masks_TZXYC = masks_TXY[..., np.newaxis, np.newaxis]  # Now shape is (T, Y, X, 1, 1)
    masks_TZXYC = np.transpose(masks_TZXYC, (0, 3, 1, 2, 4)).astype('bool')  # Change to (T, 1, Y, X, 1)
    image_TZXYC_masked = image_TZXYC * masks_TZXYC
    frame_values = create_frame_values(image_TZXYC, starting_changing_frame=starting_changing_frame, step_size_increase=step_size_increase)
    return image_TZXYC, image_TZXYC_masked, image_TXY, masks_TXY, pseudo_cytosol_masks_TXY, background_mask, frame_values


def concatenate_images(list_images, list_names, convert_to_8bit=False, list_time_intervals=None):
    """
    Concatenates sets of images based on scene and series order, ensuring consistency across scenes and series order.

    Args:
    list_images (list of numpy.ndarray): List of images to be concatenated.
    list_names (list of str): Corresponding names of the images indicating scene/series.

    Returns:
    tuple: Two lists containing the concatenated images and their corresponding names.
    """
    # Reading the file names and extracting the scene and series
    list_scenes = [name.split('/')[0] for name in list_names]
    list_unique_scenes = list(set(list_scenes))
    # sort list_unique_scenes
    list_unique_scenes.sort()
    list_concatenated_images = []
    list_names_concatenated_images = []
    list_time_concatenated = []
    for i, cell_id in enumerate(list_unique_scenes):
        # find the index of list_names that contain the cell_id and string 'Pre'
        try:
            index_pre = [i for i, name in enumerate(list_names) if cell_id in name and 'Pre' in name][0]
        except Exception:
            continue
        # find the index of list_names that contain the cell_id and string 'Pb1'
        try:
            index_pb1 = [i for i, name in enumerate(list_names) if cell_id in name and 'Pb1' in name][0]
        except Exception:
            continue

        # attempt to find second post-bleach ("Pb2"), but allow if missing
        try:
            index_pb2 = [i for i, name in enumerate(list_names) if cell_id in name and 'Pb2' in name][0]
        except IndexError:
            index_pb2 = None

        # build list of images to concatenate: always Pre and Pb1, add Pb2 if it exists
        images_to_concat = [list_images[index_pre], list_images[index_pb1]]
        if index_pb2 is not None:
            images_to_concat.append(list_images[index_pb2])

        # build list of time intervals similarly
        if list_time_intervals is not None:
            times_to_concat = [list_time_intervals[index_pre], list_time_intervals[index_pb1]]
            if index_pb2 is not None:
                times_to_concat.append(list_time_intervals[index_pb2])
        else:
            times_to_concat = None

        # concatenate along the time axis
        image = np.concatenate(images_to_concat, axis=0)
        list_time = times_to_concat

        if convert_to_8bit:
            # number of channels
            number_channels = image.shape[-1]
            for ch in range(number_channels):
                image[..., ch] = (image[..., ch] - np.min(image[..., ch])) / (np.max(image[..., ch]) - np.min(image[..., ch])) * 255
                image[..., ch] = image[..., ch].astype(np.uint8)
        list_concatenated_images.append(image)
        list_names_concatenated_images.append(cell_id)
        list_time_concatenated.append(list_time)
    return list_concatenated_images, list_names_concatenated_images, list_time_concatenated



def calculate_mask_and_background_intensity(image_TXY, masks_TXY,background_mask,pseudo_cytosol_masks_TXY):
    number_frames = image_TXY.shape[0]
    mask_intensity_nucleus = np.zeros((number_frames))
    mask_intensity_background = np.zeros((number_frames))
    mask_intensity_pseudo_cytosol = np.zeros((number_frames))
    for i in range(number_frames):
        mask = masks_TXY[i] == 1
        pseudo_cytosol_mask = pseudo_cytosol_masks_TXY[i] == 1
        mask_intensity_nucleus[i] = np.mean(image_TXY[i][mask>0])
        mask_intensity_background[i] = np.mean(image_TXY[i][background_mask>0])
        mask_intensity_pseudo_cytosol[i] = np.mean(image_TXY[i][pseudo_cytosol_mask>0])
    return mask_intensity_nucleus, mask_intensity_background, mask_intensity_pseudo_cytosol

def get_roi_pixel_values(
    image_TZXYC, 
    coordinates_roi, 
    radius_roi_size_px, 
    selected_color_channel=0, 
    mask_intensity_background=None):

    list_roi_pixel_values = []
    for j in range(coordinates_roi.shape[0]):
        x = coordinates_roi[j, 0]
        y = coordinates_roi[j, 1]
        # Define bounding box around the ROI
        x_min = int(np.floor(x - radius_roi_size_px))
        x_max = int(np.ceil(x + radius_roi_size_px))
        y_min = int(np.floor(y - radius_roi_size_px))
        y_max = int(np.ceil(y + radius_roi_size_px))
        # Ensure the indices are within image bounds
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, image_TZXYC.shape[3] - 1)
        y_max = min(y_max, image_TZXYC.shape[2] - 1)
        # Extract the sub-image for the current time point and channel
        sub_image = image_TZXYC[j, 0, y_min:y_max+1, x_min:x_max+1, selected_color_channel]
        # Create a grid of coordinates within the bounding box
        yy_indices = np.arange(y_min, y_max+1)
        xx_indices = np.arange(x_min, x_max+1)
        yy, xx = np.meshgrid(yy_indices, xx_indices, indexing='ij')
        # Compute the distance of each pixel from the ROI center
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        # Create a circular mask
        mask = distance <= radius_roi_size_px
        # Apply the mask to the sub-image to get pixels within the circle
        roi_pixels = sub_image[mask]
        # Compute the mean intensity of the pixels within the circle
        mean_intensity = np.mean(roi_pixels)
        list_roi_pixel_values.append(mean_intensity)
    mean_roi_frap = np.array(list_roi_pixel_values)
    if mask_intensity_background is not None:
        mean_roi_frap = mean_roi_frap - mask_intensity_background
    return mean_roi_frap


def clean_binary_image(binary_image):
    selem = disk(1)
    cleaned_image = binary_closing(binary_opening(binary_image, selem), selem)
    return cleaned_image.astype(bool)

# Define the image_binarization function
def image_binarization(image_TXYC, stable_FRAP_channel=0, invert_image=True, max_percentile=80, binary_iterations=2):
    image_TXY = image_TXYC[:, :, :, stable_FRAP_channel]
    image_TXY = gaussian_filter(image_TXY, sigma=1)
    image_TXY_binarized = np.zeros_like(image_TXY, dtype=np.uint8)
    for i in range(image_TXY.shape[0]):
        # Adaptive thresholding
        threshold = np.percentile(image_TXY[i], max_percentile)
        binary_mask = image_TXY[i] > threshold
        # Clean the binary image using binary morphological operations
        binary_mask = clean_binary_image(binary_mask)
        # Dilate the binary mask using scipy.ndimage's binary_dilation
        if invert_image:
            dilated_mask = binary_dilation(binary_mask, iterations=binary_iterations).astype('int')
            image_TXY_binarized[i] = 1 - dilated_mask
        else:
            image_TXY_binarized[i] = binary_dilation(binary_mask, iterations=binary_iterations).astype('int')
    return image_TXY_binarized

def find_roi_centroids(image_TXY,masks_TXY, min_diameter=10):
    """
    Segment images and find centroids of objects larger than a given minimum diameter.
    
    Args:
    image_TXY (numpy.ndarray): A 3D numpy array where each slice is a frame.
    min_diameter (float): Minimum diameter of objects to consider.
    
    Returns:
    pd.DataFrame: DataFrame with columns ['frame', 'x', 'y'] containing centroids of filtered objects.
    """
    all_centroids = []
    min_area = np.pi * (min_diameter / 2) ** 2  # Convert diameter to minimum area for a circular approximation
    for i, image in enumerate(image_TXY):
        # Label the objects in the image
        labeled_array, _ = ndi_label(image) # label
        props = regionprops(labeled_array)
        # Filter properties based on area and append centroids
        for prop in props:
            if prop.area >= min_area:  # Check if the area of the object is greater than the minimum area
                # create a temporal df containing only the x and y coordinates 
                df_located_elements = pd.DataFrame(prop.coords, columns=['y', 'x'])
                df_elements_in_mask = mi.Utilities().spots_in_mask(df_located_elements, masks_TXY[i])
                if df_elements_in_mask['In Mask'].sum() > 0:
                    centroid = prop.centroid  # Centroid as (row, col)
                    all_centroids.append({
                        'frame': i,
                        'y': centroid[0],  # Row is Y coordinate
                        'x': centroid[1],  # Column is X coordinate
                        'solidity': prop.solidity,
                        'equivalent_diameter' : prop.equivalent_diameter,
                        'filled_area': prop.filled_area,    
                        'perimeter': prop.perimeter,
                        'extent': prop.extent,
                        'area':prop.area,
                        # calculate the intensity of a crop around the centroid
                        'mean_intensity': np.mean(image[int(centroid[0])-10:int(centroid[0])+10,int(centroid[1])-10:int(centroid[1])+10])
                    })
    # Create DataFrame
    centroids_df = pd.DataFrame(all_centroids)
    return centroids_df


def detect_roi_by_difference(
    image_TZXYC_masked,  # shape (T, Z, Y, X, C)
    image_TZXYC,         # shape (T, Z, Y, X, C)
    masks_TXY,           # shape (T, Y, X)
    frap_time,           # int
    min_diameter,        # float
    stable_FRAP_channel, # int
    max_roi_displacement_px=None
):
    """
    Detect the FRAP ROI by comparing pre- and post-bleach frames, then
    validate each candidate by its intensity trace in the masked image.
    Returns:
      coordinates_roi: np.ndarray of shape (T,2) giving (x,y) per frame
      df_selected_trajectory: pd.DataFrame with columns ['frame','x','y']
    or (None, None) if detection fails.
    """

    T = image_TZXYC.shape[0]
    z = 0
    ch = stable_FRAP_channel

    # sanity check
    if frap_time <= 0 or frap_time >= T:
        return None, None

    # compute difference image
    before = image_TZXYC_masked[frap_time-1, z, :, :, ch].astype(np.float32)
    after  = image_TZXYC_masked[frap_time  , z, :, :, ch].astype(np.float32)
    diff   = gaussian_filter(after - before, sigma=1)
    neg    = -diff

    # threshold + clean
    try:
        thr = threshold_otsu(neg)
    except ValueError:
        return None, None
    bw = neg > thr
    bw = binary_opening(bw, footprint=np.ones((3,3)))
    bw = binary_closing(bw, footprint=np.ones((3,3)))

    # find candidate regions
    lbl   = label(bw)
    props = regionprops(lbl)
    exp_area = np.pi * (min_diameter/2)**2
    tol      = exp_area * 0.5

    candidates = []
    for p in props:
        if p.perimeter == 0:
            continue
        area = p.area
        circ = (4*np.pi*area) / (p.perimeter**2)
        if abs(area - exp_area) <= tol:
            candidates.append((circ, p.centroid))  # centroid = (y,x)

    # Hough fallback if no regionprops
    if not candidates:
        edges = canny(neg, sigma=2)
        r = int(round(min_diameter/2))
        hres = hough_circle(edges, [r])
        acc, cx, cy, rad = hough_circle_peaks(hres, [r], total_num_peaks=1)
        if cx.size:
            candidates = [(1.0, (cy[0], cx[0]))]

    # sort by circularity
    candidates.sort(key=lambda x: x[0], reverse=True)

    radius = min_diameter/2

    # test each candidate by its ROI intensity trace
    for circ, (cy, cx) in candidates:
        # build track
        coords = np.tile([cx, cy], (T,1))

        # clamp displacements
        if max_roi_displacement_px is not None:
            deltas = np.diff(coords, axis=0)
            too_big = np.sum(deltas**2, axis=1) > max_roi_displacement_px**2
            for idx in np.where(too_big)[0] + 1:
                coords[idx] = coords[idx-1]

        # sample intensity trace
        roi_int = get_roi_pixel_values(
            image_TZXYC=image_TZXYC_masked,
            coordinates_roi=coords,
            radius_roi_size_px=radius,
            selected_color_channel=ch,
            mask_intensity_background=None
        )

        # validate: stable before bleach
        base   = roi_int[:frap_time]
        baseline, pre_std = base.mean(), base.std()
        drop      = roi_int[frap_time+1]
        recovery  = roi_int[-1] - roi_int[frap_time]

        if drop <= baseline*0.6: # and recovery >= drop*0.1:
            # success
            df = pd.DataFrame({
                'frame': np.arange(T),
                'x': coords[:,0],
                'y': coords[:,1]
            })
            return coords, df

    # no candidate passed
    return None, None


def detect_roi_by_tracking(
    image_TZXYC_masked,
    image_TZXYC,
    masks_TXY,
    frap_time,
    min_diameter,
    stable_FRAP_channel,
    use_frap_time_for_roi_detection,
    max_roi_displacement_px,
    FRAP_channel_to_quantify=0,
    show_binary_plot=False,
    list_selected_frames=[0, 10, 40, 100, 139],
    list_selected_frame_values_real_time=None,
    mask_intensity_background=None,
):
    """
    Fallback: TrackPy-based detection, wrap in try/except.
    """
    try:
        # Use similar logic as in find_frap_roi's else branch
        list_max_percentile = np.linspace(95, 100, 5, dtype=int)
        list_binary_iterations = np.linspace(1, 10, 5, dtype=int)
        num_frames = image_TZXYC.shape[0]
        break_condition_met = False
        for max_percentile in list_max_percentile:
            for binary_iterations in list_binary_iterations:
                try:
                    image_TXY_binarized = image_binarization(
                        image_TZXYC_masked[:, 0, :, :, :],
                        stable_FRAP_channel=stable_FRAP_channel,
                        invert_image=True,
                        max_percentile=max_percentile,
                        binary_iterations=binary_iterations
                    )
                    min_frames_for_linking = image_TXY_binarized.shape[0] // 2
                    minimal_distance_between_centroids = 4
                    centroids_df_before = find_roi_centroids(image_TXY_binarized[:1], masks_TXY[:1], min_diameter )
                    centroids_df_after = find_roi_centroids(image_TXY_binarized[frap_time:], masks_TXY[frap_time:], min_diameter )
                    def min_distance_to_before(row, centroids_df_before):
                        distances = np.sqrt( (centroids_df_before['x'] - row['x']) ** 2 + (centroids_df_before['y'] - row['y']) ** 2)
                        return distances.min()
                    centroids_df_after['min_distance'] = centroids_df_after.apply(min_distance_to_before,axis=1, args=(centroids_df_before,))
                    centroids_df = centroids_df_after[
                        centroids_df_after['min_distance'] >= minimal_distance_between_centroids
                    ]
                    dataframe_linked_elements = tp.link(centroids_df, search_range=7, memory=1)
                    df_roi = tp.filter_stubs(dataframe_linked_elements, min_frames_for_linking)
                    if df_roi.empty:
                        continue
                    else:
                        list_particles = df_roi['particle'].unique()
                        for particle_id in list_particles:
                            df_selected_trajectory = df_roi[df_roi['particle'] == particle_id]
                            df_selected_trajectory.reset_index(drop=True, inplace=True)
                            number_frames_after_frap = image_TXY_binarized.shape[0] - frap_time
                            all_frames = pd.DataFrame({'frame': range(number_frames_after_frap)})
                            df_full = pd.merge(all_frames,df_selected_trajectory,on='frame', how='left')
                            df_full.fillna(method='ffill', inplace=True)
                            df_full.fillna(method='bfill', inplace=True)
                            df_selected_trajectory = df_full
                            num_new_rows = frap_time
                            new_rows = pd.DataFrame([df_selected_trajectory.iloc[0]] * num_new_rows)
                            new_rows['frame'] = range(0, num_new_rows)
                            df_selected_trajectory = pd.concat([new_rows, df_selected_trajectory], ignore_index=True)
                            df_selected_trajectory['frame'] = range(df_selected_trajectory.shape[0])
                            df_selected_trajectory.sort_values('frame', inplace=True)
                            df_selected_trajectory.reset_index(drop=True, inplace=True)
                            coordinates_roi = df_selected_trajectory[['x', 'y']].values
                            if max_roi_displacement_px is not None:
                                deltas = np.diff(coordinates_roi, axis=0)
                                d2 = np.sum(deltas**2, axis=1)
                                mask = d2 > (max_roi_displacement_px ** 2)
                                if np.any(mask):
                                    idxs = np.where(mask)[0] + 1
                                    for idx in idxs:
                                        coordinates_roi[idx] = coordinates_roi[idx-1]
                            # Ensure track stays within the selected nucleus mask
                            mask_hits = np.array([
                                masks_TXY[i, int(round(coord[1])), int(round(coord[0]))]
                                for i, coord in enumerate(coordinates_roi)
                            ])
                            # if less than 80% of points lie within the mask, skip this track
                            if np.mean(mask_hits) < 0.8:
                                continue
                            return coordinates_roi, df_selected_trajectory
                except Exception:
                    continue
        return None, None
    except Exception:
        return None, None


def find_frap_roi(
    image_TZXYC_masked,
    image_TZXYC,
    masks_TXY,
    frap_time,
    min_diameter=10,
    FRAP_channel_to_quantify=0,
    stable_FRAP_channel=0,
    show_binary_plot=False,
    list_selected_frames=[0, 10, 40, 100, 139],
    list_selected_frame_values_real_time=None,
    mask_intensity_background=None,
    use_frap_time_for_roi_detection=True,
    max_roi_displacement_px=5
):
    """
    New function to find FRAP ROI using difference or tracking.
    Returns (mean_roi_frap, mean_roi_frap_normalized, coordinates_roi, df_selected_trajectory)
    """
    coordinates_roi = None
    df_selected_trajectory = None
    if use_frap_time_for_roi_detection:
        coordinates_roi, df_selected_trajectory = detect_roi_by_difference(
            image_TZXYC_masked,
            image_TZXYC,
            masks_TXY,
            frap_time,
            min_diameter,
            stable_FRAP_channel,
            max_roi_displacement_px,
        )
    else:
        coordinates_roi, df_selected_trajectory = detect_roi_by_tracking(
            image_TZXYC_masked,
            image_TZXYC,
            masks_TXY,
            frap_time,
            min_diameter,
            stable_FRAP_channel,
            use_frap_time_for_roi_detection,
            max_roi_displacement_px,
            FRAP_channel_to_quantify=FRAP_channel_to_quantify,
            show_binary_plot=show_binary_plot,
            list_selected_frames=list_selected_frames,
            list_selected_frame_values_real_time=list_selected_frame_values_real_time,
            mask_intensity_background=mask_intensity_background,
        )
    if coordinates_roi is not None and df_selected_trajectory is not None:
        radius_roi_size_px = min_diameter / 2
        mean_roi_frap = get_roi_pixel_values(
            image_TZXYC=image_TZXYC,
            coordinates_roi=coordinates_roi,
            radius_roi_size_px=radius_roi_size_px,
            selected_color_channel=FRAP_channel_to_quantify,
            mask_intensity_background=mask_intensity_background,
        )
        # Reject if the ROI’s average intensity is too low
        # if np.mean(mean_roi_frap) < 10:
        #     return None, None, None, None
        mean_roi_frap_normalized = mean_roi_frap / np.mean(mean_roi_frap[:frap_time])
        return mean_roi_frap, mean_roi_frap_normalized, coordinates_roi, df_selected_trajectory
    else:
        return None, None, None, None








def process_selected_df(df_roi, frap_time, image_TZXYC, FRAP_channel_to_quantify, min_diameter):
    df_selected_trajectory = df_roi[df_roi['particle'] == df_roi['particle'].iloc[0]]  # Assuming selection of first found particle
    coordinates_roi = df_selected_trajectory[['x', 'y']].values
    mean_roi_frap = get_roi_pixel_values(image_TZXYC, coordinates_roi, min_diameter, FRAP_channel_to_quantify)
    mean_roi_frap_normalized = mean_roi_frap / np.mean(mean_roi_frap[:frap_time])
    return mean_roi_frap, mean_roi_frap_normalized, coordinates_roi, df_selected_trajectory


def remove_cell_without_roi_detection(df_tracking_all, threhsold=0.08):
    list_selected_df = []
    # removing elements where the code is unable to detect the roi
    for name in df_tracking_all['image_name'].unique():
        df_selected = df_tracking_all[df_tracking_all['image_name'] == name]
        # this code removes the elements where the normalized intensity is lower than
        dif_norm_int = np.diff (df_selected['mean_roi_frap_normalized'])
        # smooth the curve        
        dif_norm_int_smoothed = np.convolve(dif_norm_int, np.ones(3)/3, mode='same')
        peaks, _ = find_peaks(-dif_norm_int_smoothed, height=threhsold, distance=7)
        peaks_positive, _ = find_peaks(dif_norm_int_smoothed, height=threhsold, distance=7)
        # if the only one peak is detected, save the dataframe to list_selected_df and the peaks is located in position less than 14 frames
        if len(peaks) == 1 and peaks[0] < 14 and len(peaks_positive) == 0:
            print('Image:', name, 'Peak detected at frame:', peaks[0])
            list_selected_df.append(df_selected)
        else:
            print('Image:', name, 'No peak detected')
    # remove the elements where the code is unable to detect the roi
    df_tracking_removed_roi_no_detected = pd.concat(list_selected_df, ignore_index=True)
    return df_tracking_removed_roi_no_detected



##################################################################################
##################### Plotting functions #########################################
##################################################################################



def plot_images_frap(image_TZXYC, list_selected_frames=[0, 10, 40, 100, 139], subtitle= None, show_grid=False,
                      cmap='viridis', selected_color_channel=None, coordinates_roi=None, radius_roi_size_px=10,
                      save_plot=False, plot_name='temp.png',list_selected_frame_values_real_time=None, masks_TXY=None, pseudo_cytosol_masks_TXY=None):
    if selected_color_channel is not None:
        # Reducing dimensions to only selected color channel if specified
        image_TZXYC = image_TZXYC[..., selected_color_channel]
        # Add a dimension at the end to maintain a consistent 5D array format
        image_TZXYC = np.expand_dims(image_TZXYC, axis=-1)
    number_color_channels = image_TZXYC.shape[-1]
    # Check if we need to adjust the subplot setup for only one color channel
    if number_color_channels == 1:
        fig, ax = plt.subplots(1, len(list_selected_frames), figsize=(10, 5))
        ax = np.array([ax]).reshape(1, -1)  # Ensure ax is always a 2D array for consistency
    else:
        fig, ax = plt.subplots(number_color_channels, len(list_selected_frames), figsize=(14, 5))
    for ch in range(number_color_channels):
        for i, frame in enumerate(list_selected_frames):
            current_ax = ax[ch, i] if number_color_channels > 1 else ax[0, i]
            current_ax.imshow(image_TZXYC[frame, 0, :, :, ch], vmax=np.percentile(image_TZXYC[frame, 0, :, :, ch], 99.8), cmap=cmap)
            #if masks_TXY is not None:
            #    current_ax.contour(masks_TXY[i], colors='r', linewidths=1 , linestyles='solid')
            if pseudo_cytosol_masks_TXY is not None:
                current_ax.contour(pseudo_cytosol_masks_TXY[i], colors='w', linewidths=0.5, linestyles='solid')
            
            if list_selected_frame_values_real_time is not None:
                current_ax.set_title(f'{int(list_selected_frame_values_real_time[i])} s', fontsize=10)
                # plot the frame numbers as integers
            else:
                current_ax.set_title(f'Frame index ({int(frame)})', fontsize=10)
            #current_ax.set_title(f'Frame ({frame})', fontsize=10)
            if i == 0:
                current_ax.set_ylabel(f'Ch {ch}' if selected_color_channel is None else f'Selected Ch {selected_color_channel}', fontsize=10)
            if show_grid:
                current_ax.axvline(x=image_TZXYC.shape[3] // 2, color='w', linestyle='-', linewidth=0.5)
                current_ax.axhline(y=image_TZXYC.shape[2] // 2, color='w', linestyle='-', linewidth=0.5)
            else:
                current_ax.grid(False)
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            # plot the roi as a circle
            if coordinates_roi is not None:
                x = coordinates_roi[frame,0]
                y = coordinates_roi[frame,1]
                circle = plt.Circle((x, y), radius_roi_size_px+1, color='orangered', fill=False, linewidth=0.7)
                current_ax.add_artist(circle)
    if subtitle is not None:
        fig.suptitle(subtitle, fontsize=16)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_name, dpi=300)
    plt.show()


def plot_frap_quantification(frame_values, mean_roi_frap_normalized,mean_roi_frap, mask_intensity_nucleus, mask_intensity_pseudo_cytosol, mask_intensity_background, frap_time,save_plot=False, plot_name='temp.png'):
    use_normalized_frap_int = False
    # plot the intensity of the mask for all frames. 
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.5))
    sliding_window = 5
    if use_normalized_frap_int:
        frap_int = mean_roi_frap_normalized
    else:
        frap_int = mean_roi_frap
    
    ax[0].plot(frame_values, frap_int, '-r')
    frap_smoothed = np.convolve(frap_int[frap_time+1:], np.ones(sliding_window)/sliding_window, mode='same')
    ax[0].plot(frame_values[frap_time+1:-sliding_window], frap_smoothed[:-sliding_window], '-g', lw=2)
    ax[0].set_xlabel('Frames')
    ax[0].set_ylabel('Mean Pixel Value')
    ax[0].set_title('Intensity', fontsize=10)

    ax[1].plot(frame_values,mask_intensity_nucleus)
    ax[1].set_title(f'Nucleus', fontsize=10)

    ax[2].plot(frame_values,mask_intensity_pseudo_cytosol)
    ax[2].set_title(f'pseudo_Cytosol', fontsize=10)


    ax[3].plot(frame_values,mask_intensity_background)
    ax[3].set_title('Background', fontsize=10)

    ax[1].set_xlabel('Frame')
    ax[2].set_xlabel('Frame')
    ax[3].set_xlabel('Frame')

    ax[1].set_ylabel('Intensity')
    ax[2].set_ylabel('Intensity')
    ax[3].set_ylabel('Intensity')

    # ensure that the limits are the same for both plots
    ax[1].set_ylim([0, 1.01*np.max(mask_intensity_nucleus)])
    ax[2].set_ylim([0, 1.01*np.max(mask_intensity_nucleus)])
    ax[3].set_ylim([0, 1.01*np.max(mask_intensity_nucleus)])

    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_name, dpi=300)
    plt.show()
    return None


def plot_frap_quantification_all_images(df_tracking_all, save_plot=False, plot_name='temp.png'):
    # Create a figure and two horizontal subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Plot from the dataframe the mean_roi_frap_normalized vs the frame for each image_name 
    for name in df_tracking_all['image_name'].unique():
        df_selected = df_tracking_all[df_tracking_all['image_name'] == name]
        ax1.plot(df_selected['frame'], df_selected['mean_roi_frap_normalized'], label=name)
    ax1.set_title('Normalized ROI FRAP Mean by Image')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Normalized Mean ROI FRAP')
    # Plot the mean and std of the mean_roi_frap_normalized vs frame for all df_tracking_all
    mean_values = []
    std_values = []
    frames = df_tracking_all['frame'].unique()
    for frame in frames:
        df_selected = df_tracking_all[df_tracking_all['frame'] == frame]
        mean_values.append(np.mean(df_selected['mean_roi_frap_normalized']))
        std_values.append(np.std(df_selected['mean_roi_frap_normalized']))
    ax2.errorbar(frames, mean_values, yerr=std_values, fmt='-o')
    ax2.set_title('Mean and Standard Deviation of Normalized ROI FRAP')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Mean ± STD')
    # Display the plot
    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_name, dpi=300)
    plt.show()
    return  np.array(frames),  np.array(mean_values), np.array(std_values)


def create_pdf(list_combined_image_paths,pdf_name, remove_original_images=False):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for image_path in list_combined_image_paths:
        # if image_path exists, add it to the pdf
        if image_path.exists():
            pdf.add_page()
            pdf.image(str(image_path), x=10, y=10, w=190)
            # add page and text to the pdf indicating that the image does not exist
        else:
            pdf.add_page()
            pdf.set_xy(10, 10)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Image {image_path} was not processed. ", ln=True)
        if remove_original_images:
            image_path.unlink()
    # save the pdf
    pdf.output(str(pdf_name))   
    return None


def plot_t_half_values(df_fit, r2_threshold=0.5, suptitle=None,save_plot=True, plot_name='temp.png'):
    """
    Plot the half-time values (t_half_single, t_half_double_1st_process, t_half_double_2nd_process) as box plots.

    Parameters:
    - df_fit (pd.DataFrame): DataFrame containing the fit results.
    - r2_threshold (float): Minimum R^2 value to filter data. Default is 0.5.
    - y_lim_max (float or None): Maximum y-limit for the plots. If None, it will be set to the 95th percentile of the data.
    """
    # Remove rows where r_2_single is less than the threshold
    df_fit_selected = df_fit[df_fit['r_2_single'] > r2_threshold]
    # Concatenate t_half values into a single DataFrame
    tau_values = pd.concat([
        df_fit_selected['t_half_single'].dropna(),
        df_fit_selected['t_half_double_1st_process'].dropna(),
        df_fit_selected['t_half_double_2nd_process'].dropna(),
    ], axis=1)
    # Ensure that the columns are numeric
    tau_values = tau_values.apply(pd.to_numeric, errors='coerce')
    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    # List of tau columns and labels
    tau_columns = ['t_half_single', 't_half_double_1st_process', 't_half_double_2nd_process']
    labels = [r"$t_{1/2}$"+ " [single exponential]", r"$t_{1/2}$"+ " 1st process [double exponential]", r"$t_{1/2}$"+ " 2nd process [double exponential]"]
    for i, col in enumerate(tau_columns):
        axes[i].boxplot(tau_values[col].dropna(), labels=[labels[i]])
        # Show all the values as dots
        axes[i].plot(np.ones(len(tau_values[col]))*1, tau_values[col], 'ro', alpha=0.3)
        #axes[i].set_ylim([0, y_lim_max])
        axes[i].set_ylabel(r"$t_{1/2}$")
        #axes[i].set_title(f'Box Plot of {labels[i]}')
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    if save_plot:
        plt.savefig(plot_name, dpi=300)
    plt.tight_layout()
    plt.show()





##################################################################################
######################### Fit functions ##########################################
##################################################################################



def fit_model_to_frap(time, intensity, frap_time,suptitle=None, show_time_before_bleaching=True, save_plot=True, plot_name='temp.png'):
    # intial guesses for the single and double exponential models
    # p0_single = [intensity.max(), # max intensity
    #             (intensity[-1] - np.min(intensity) ) / (np.max(intensity) - np.min(intensity)), # difference between max and min intensity
    #             1.0 / (time.max() - time.min())] # rate constant
    # time and intensity before the bleach
    time_before = time[:frap_time]
    intensity_before = intensity[:frap_time]
    # time and intensity after the bleach
    time = time[frap_time:] 
    intensity = intensity[frap_time:] 
    # Define the single and double exponential models
    def frap_model_single_exp(t,I_0, a, b ):
    #     # i_fit = I_0 - a * exp(-b t) 
        return I_0 - (a * np.exp(-b * t))
    #def frap_model_single_exp(t, I_0, I_inf, tau):
    #    return I_0 + (I_inf - I_0) * (1 - np.exp(-tau * t))
    def frap_model_double_exp(t, I_0, a, b, g, d):
        # i_fit = I_0 - a * exp(-b t) - g * exp(-d t)
        return I_0 - (a * np.exp(-b * t)) - (g * np.exp(-d * t))
    # Initial guesses for parameters
    p0_single = [1, 0.5, 0.005]
    p0_double = [1, 0.5, 0.005, 1, 0.5]#, 50, 100, intensity.min()]
    # Fit the models
    flag_single = False
    flag_double = False
    try:
        params_single, _ = curve_fit(frap_model_single_exp, time, intensity, p0=p0_single)
        flag_single = True
        t_half_single = np.log(2)/params_single[2]
    except:
        params_single = [np.nan, np.nan, np.nan]
        t_half_single = np.nan
    try:
        params_double, _ = curve_fit(frap_model_double_exp, time, intensity, p0=p0_double)
        flag_double = True
        t_half_double_1st_process = np.log(2)/params_double[2]
        t_half_double_2nd_process = np.log(2)/params_double[4]
    except:
        params_double = [np.nan, np.nan, np.nan, np.nan, np.nan]
        t_half_double_1st_process = np.nan
        t_half_double_2nd_process = np.nan
    # Calculate R-squared for each model
    def compute_r_squared(data, fit):
        residuals = data - fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data - np.mean(data))**2)
        return 1 - (ss_res / ss_tot)
    if flag_single:
        r_squared_single = compute_r_squared(intensity, frap_model_single_exp(time, *params_single))
    else:
        r_squared_single = np.nan
    if flag_double:
        r_squared_double = compute_r_squared(intensity, frap_model_double_exp(time, *params_double))
    else:
        r_squared_double = np.nan
    # Create plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # Single exponential fit plot
    if show_time_before_bleaching:
        axes[0].plot(time_before, intensity_before, 'ro')
    axes[0].plot(time, intensity, 'ro', label='Data')
    if flag_single:
        axes[0].plot(time, frap_model_single_exp(time, *params_single), 'k-', label='Fit: Single Exp')
    axes[0].set_title('Single Exponential Fit')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    if flag_single:
        axes[0].text(0.1, 0.9, f"$t = {t_half_single:.2f}$", transform=axes[0].transAxes)
        axes[0].text(0.1, 0.8, f"$R^2 = {r_squared_single:.2f}$", transform=axes[0].transAxes)
    # Double exponential fit plot
    if show_time_before_bleaching:
        axes[1].plot(time_before, intensity_before, 'ro')
    axes[1].plot(time, intensity, 'ro', label='Data')
    if flag_double:
        axes[1].plot(time, frap_model_double_exp(time, *params_double), 'k-', label='Fit: Double Exp')
    axes[1].set_title('Double Exponential Fit')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()
    if flag_double:
        axes[1].text(0.1, 0.9, f"$t_{{1st}} = {t_half_double_1st_process:.2f}$", transform=axes[1].transAxes)
        axes[1].text(0.1, 0.8, f"$t_{{2nd}} = {t_half_double_2nd_process:.2f}$", transform=axes[1].transAxes)
        axes[1].text(0.1, 0.7, f"$R^2 = {r_squared_double:.2f}$", transform=axes[1].transAxes)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_name, dpi=300)
    plt.show()
    return t_half_single, t_half_double_1st_process,t_half_double_2nd_process, r_squared_single, r_squared_double


def fit_model_to_frap_immobile_fraction(time, intensity, frap_time,suptitle=None, show_time_before_bleaching=True,save_plot=True, plot_name='temp.png'):
    # time and intensity before the bleach
    time_before = time[:frap_time]
    intensity_before = intensity[:frap_time]
    # time and intensity after the bleach
    time = time[frap_time:] 
    intensity = intensity[frap_time:] 
    # Define the single and double exponential models
    def frap_model_single_exp(t, f_imm, tau, I_0):
        return (1 - f_imm) * (1 - np.exp(-t / tau)) + I_0 * f_imm
    def frap_model_double_exp(t, f_imm, tau1, tau2, I_0):
        return (1 - f_imm) * (1 - np.exp(-t / tau1)) + f_imm * (1 - np.exp(-t / tau2)) + I_0
    # Initial guesses for parameters
    p0_single = [1-intensity.max(), 50, intensity.min()]
    p0_double = [1-intensity.max(), 50, 5, intensity.min()]
    # Fit the models
    flag_single = False
    flag_double = False
    try:
        params_single, _ = curve_fit(frap_model_single_exp, time, intensity, p0=p0_single)
        flag_single = True
        t_half_single = params_single[1]

        if t_half_single < 0 or t_half_single > 2000:
            flag_single = True
            t_half_single = np.nan
            params_single = [np.nan, np.nan, np.nan]
    except:
        params_single = [np.nan, np.nan, np.nan]
        t_half_single = np.nan
    try:
        params_double, _ = curve_fit(frap_model_double_exp, time, intensity, p0=p0_double)
        flag_double = True
        t_half_double_1st_process = params_double[1]
        t_half_double_2nd_process = params_double[2]
    except:
        params_double = [np.nan, np.nan, np.nan, np.nan]
        t_half_double_1st_process = np.nan
        t_half_double_2nd_process = np.nan
    # Calculate R-squared for each model
    def compute_r_squared(data, fit):
        residuals = data - fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data - np.mean(data))**2)
        return 1 - (ss_res / ss_tot)
    if flag_single:
        r_squared_single = compute_r_squared(intensity, frap_model_single_exp(time, *params_single))
    else:
        r_squared_single = np.nan
    if flag_double:
        r_squared_double = compute_r_squared(intensity, frap_model_double_exp(time, *params_double))
    else:
        r_squared_double = np.nan
    # Create plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # Single exponential fit plot
    axes[0].plot(time, intensity, 'ro', label='Data')
    if show_time_before_bleaching:
        axes[0].plot(time_before, intensity_before, 'ro')
    if flag_single:
        axes[0].plot(time, frap_model_single_exp(time, *params_single), 'k-', label='Fit: Single Exp')
    axes[0].set_title('Single Exponential Fit')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    if flag_single:
        axes[0].text(0.1, 0.9, f"$t = {t_half_single:.2f}$", transform=axes[0].transAxes)
        axes[0].text(0.1, 0.8, f"$R^2 = {r_squared_single:.2f}$", transform=axes[0].transAxes)
    # Double exponential fit plot
    axes[1].plot(time, intensity, 'ro', label='Data')
    if show_time_before_bleaching:
        axes[1].plot(time_before, intensity_before, 'ro')
    if flag_double:
        axes[1].plot(time, frap_model_double_exp(time, *params_double), 'k-', label='Fit: Double Exp')
    axes[1].set_title('Double Exponential Fit')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()
    if flag_double:
        axes[1].text(0.1, 0.9, f"$t_{{1st}} = {t_half_double_1st_process:.2f}$", transform=axes[1].transAxes)
        axes[1].text(0.1, 0.8, f"$t_{{2nd}} = {t_half_double_2nd_process:.2f}$", transform=axes[1].transAxes)
        axes[1].text(0.1, 0.7, f"$R^2 = {r_squared_double:.2f}$", transform=axes[1].transAxes)
    plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_name, dpi=300)
    plt.show()
    return t_half_single, t_half_double_1st_process,t_half_double_2nd_process, r_squared_single, r_squared_double