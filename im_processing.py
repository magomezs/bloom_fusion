import rawpy
import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub



# --------------------------------- COORDINATES AND DISTANCES CALCULATION ----------------------------------------------

def get_intersection_rois(patch, background, pos):
    """ This function calculates the intersection among a background image and a patch located in the given pos.
    pos is the location of the background where the center of the patch is going to be placed
    Returns:
        sup_p, inf_p, left_p, right_p: coordinates of the intersection region in the patch image
        sup_b, inf_b, left_b, right_b: coordinates of the intersection region in the background image """

    [H, W, _] = background.shape
    [h, w, _] = patch.shape

    # coordinates initialization
    sup_p = 0
    inf_p = h - 1
    left_p = 0
    right_p = w - 1

    sup_b = int(sup_p + pos[1] - np.round(h / 2.0))
    inf_b = int(inf_p + pos[1] - np.round(h / 2.0))
    left_b = int(left_p + pos[0] - np.round(w / 2.0))
    right_b = int(right_p + pos[0] - np.round(w / 2.0))

    # correction if the coordinates exceeds the dimension limits
    if sup_b < 0: sup_b = 0
    if inf_b > H - 1: inf_b = H - 1
    if left_b < 0: left_b = 0
    if right_b > W - 1: right_b = W - 1

    sup_p = int(sup_b - pos[1] + np.round(h / 2.0))
    inf_p = int(inf_b - pos[1] + np.round(h / 2.0))
    left_p = int(left_b - pos[0] + np.round(w / 2.0))
    right_p = int(right_b - pos[0] + np.round(w / 2.0))

    return sup_p, inf_p, left_p, right_p, sup_b, inf_b, left_b, right_b

def min_distance_to_center(h, w, r, c, edge_top, edge_bottom, edge_left, edge_right):
    """ This function calculate the distance from a certain pixel in row r and column c to the center of
    an image with dimensions h x w.
    edge_top, edge_bottom, edge_left, edge_right are booleans. These indicate whether the corresponding edge of the
    patch is also an edge of the background image"""

    center_x = np.round(w / 2.0)
    center_y = np.round(h / 2.0)
    if h < w:
        comp_y = np.abs(r-center_y)
        if c < center_y:
            comp_x = center_y-c
        elif c > w-center_y:
            comp_x=c-int(w-int(center_y))
        else:
            comp_x = 0
    else:
        comp_x = np.abs(c-center_x)
        if r < center_x:
            comp_y=int(center_x)-r
        elif r > h-int(center_x):
            comp_y=r-int(h-int(center_x))
        else:
            comp_y = 0

    # correction to not soft the borders
    if edge_top and (r<center_y):
        comp_y=0
    if edge_bottom and (r>center_y):
        comp_y=0
    if edge_left and (c<center_x):
        comp_x=0
    if edge_right and (c>center_x):
        comp_x=0


    return np.sqrt(np.power(float(comp_x), 2)+np.power(float(comp_y),2))



# ------------------------------------------- EDITTING FUNCTIONS -------------------------------------------------------

def scale_and_locate(patch, background, pos):
    """ This function ask the user instructions to scale and locate a patch in a background image
    Parameters:
        patch: image patch to scale and to be located over the background image
        background: background image
        pos: initial insertion pos (point of the background image where the center of the patch is going to be located
    Returns:
          patch: new scaled patch
          pos: new location in the background image
    """

    scale=1.0 # current scale
    print('Press +/- to zoom out and zoom in. Press 2,4,6,8 to move the path. Press "enter" to insert. ')
    while True:

        sum_im = put_patch(patch, background, pos) # sum_im is the background with the patch overlapped in pos

        # USER DECISIONS
        show_image(sum_im, 'background_image', None, [-40, 250])
        key=cv.waitKey()
        if key == 13:
            cv.destroyWindow('background_image')
            break
        if key == 45:
            scale = scale-0.1
            patch = cv.resize(patch, (0, 0), fx=scale, fy=scale)
        if key == 43:
            scale = scale+0.1
            patch = cv.resize(patch, (0, 0), fx=scale, fy=scale)
        if key == 50: pos=(pos[0], pos[1]+1)
        if key == 56: pos=(pos[0], pos[1]-1)
        if key == 52: pos=(pos[0]-1, pos[1])
        if key == 54: pos=(pos[0]+1, pos[1])

    return patch, pos


def put_patch(patch, background, pos):
    """ This functions puts a patch over a background image in the given position (pos)
    pos is the location of the background where the center of the patch is going to be placed"""
    sup_p, inf_p, left_p, right_p, sup_b, inf_b, left_b, right_b= get_intersection_rois(patch, background, pos)
    result_im = background.copy()
    result_im[sup_b:inf_b, left_b:right_b]=patch[sup_p:inf_p, left_p:right_p]
    return result_im


def scale_im(original, pattern):
    """ Original image is resized to get the same dimensions as the pattern """

    [H, W, _]= pattern.shape
    [h, w, _]= original.shape
    factor_x= float(W)/float(w)
    factor_y= float(H)/float(h)
    return cv.resize(original, (0, 0), fx=factor_x, fy=factor_y)


def cropping(im, posList1, posList2, pad_factor):
    """ This function crop the image according to two given positions (upper left conner and lower right coner)
    This function also add padding to the resulting crop. The width of the added margin is a certain proportion of the
    mean size of the crop. That proportion is given by pad_factor"""
    if len(posList1)==0 or len(posList2)==0:
        return im
    else:
        # Cropping
        point1 = np.array(posList1[-1])  # convert to NumPy for later use
        point2 = np.array(posList2[-1])  # convert to NumPy for later use
        crop = im[min(point1[1],point2[1]):max(point1[1],point2[1]), min(point1[0],point2[0]):max(point1[0],point2[0])]

        # Padding
        if pad_factor > 0.0: # Pad the image
            [h, w, _] = im.shape
            margin = int(np.round(pad_factor * ((h + w) / 2.0)))
            padded = cv.copyMakeBorder(crop, margin, margin, margin, margin, cv.BORDER_REPLICATE)
            result=padded.copy()
        else:
            result=crop.copy()
    return result


def smooth_limits(patch, blurring, sup_b, inf_b, left_b, right_b, b_shape):
    """ This function generates a mask to smooth the borders of a patch, except when the patch borders are also the
    background borders.
    Parameters:
        patch: image whose borders are going to be smoothed
        blurring: percentage of border smoothed. 1 corresponds to the half of the smaller side.
        sup_b, inf_b, left_b, right_b= coordinates of the patch in the background image
        b_shape = shape of the background image
    Returns:
        final_mask: mask to smooth the borders of a patch
    """

    # Get parameters for the smoothing equation: patch and background shape, max_distance, starting_smooth
    [h, w, _] = patch.shape
    [H, W, _] = b_shape
    max_distance = min([int(np.round(w/2.0)), int(np.round(h/2.0))]) # half of the smaller side
    starting_smooth = 1 - blurring    # Distance from the center of the image to the starting of smoothing equation
                                      # (as factor from 0 to 1)
    abs_border=int(np.round(blurring*max_distance)) # width of the smooth border
    # Add pad for avoiding smoothing the limits of patch that corresponds to background limits
    margin_top=False
    margin_bottom=False
    margin_left=False
    margin_right=False
    if (sup_b == 0):
        margin_top = True
    if inf_b == H - 1:
        margin_bottom = True
    if left_b == 0:
        margin_left = True
    if right_b == W - 1:
        margin_right = True

    """if (sup_b < abs_border):
        margin_top = int(abs_border-sup_b)
    if inf_b > H - 1 - abs_border:
        margin_bottom = int(abs_border-(H-1-inf_b))
    if left_b<abs_border:
        margin_left = int(abs_border-left_b)
    if right_b > W - 1 - abs_border:
        margin_right = int(abs_border-(W-1-right_b))"""
    #padded = cv.copyMakeBorder(patch, margin_top, margin_bottom, margin_left, margin_right, cv.BORDER_CONSTANT)
    #[h, w, _] = padded.shape

    # create a mask to smooth the borders
    mask = np.ones(patch.shape, np.float64)
    b = 1.0/(starting_smooth-1.0)   # parameters of the smooth equation
    m = -1.0*b

    for r in range(h):
        for c in range(w):
            dist = min_distance_to_center(h, w, r, c, margin_top, margin_bottom, margin_left, margin_right)
            if dist < starting_smooth*max_distance: factor = 1.0
            else: factor = m+b*(dist/max_distance)
            mask[r, c, :] = factor#mask[r, c, :]*factor

    final_mask =saturate(mask, 0.0, 1.0)
    #final_mask= final_mask.astype(np.uint8)

    #show_image(final_mask, 'soft_mask', onMouseValue, -40, 250)
    #cv2.waitKey()
    return final_mask



# ------------------------------------------------- FUSION -------------------------------------------------------------

def fusion_bloom_synthetic(patch, content, background, annotation, pos, config):
    """ This function fusion a stylized patch with a background image so that the patch is inserted in the background
    in a realistic way
    Parameters:
        patch: patch image to be inserted in the background (already stylized patch)
        content: patch before transfer style. This is used as mask of content
        background: background image
        annotation: segmentation ground truth of the background image
        pos: position of the background image where the patch is going to be inserted (center of the patch)
        config: dictionary with a certain values configuration fot the program parameters"""

    sup_p, inf_p, left_p, right_p, sup_b, inf_b, left_b, right_b = get_intersection_rois(patch, background, pos)

    # get patch mask and image. An offset is added to the mask. The patch is colored.
    patch_mask = saturated_sum(content[sup_p:inf_p, left_p:right_p, :], 0.0, 1.0, 0.25)
    color=choose_color(background[sup_b:inf_b, left_b:right_b, :], config)
    patch_im = colored(patch[sup_p:inf_p, left_p:right_p, :], content[sup_p:inf_p, left_p:right_p, :], color, config)

    # get mask of regions with water
    mask_water = (annotation == 1 )*1.0 +(annotation == 5 )*1.0

    # process the patch
    product_patch = mask_water[sup_b:inf_b, left_b:right_b] * (patch_mask[sup_p:inf_p, left_p:right_p])
    smooth_patch = smooth_limits(product_patch, config['blurring'], sup_b, inf_b, left_b, right_b, background.shape)
    final_patch=product_patch*smooth_patch

    # get global mask
    mask = (0.0 * mask_water.copy())
    mask[sup_b:inf_b, left_b:right_b] = final_patch
    mask=saturate(mask, 0.0, 1.0)
    result_annotation=add_bloom_annotation(annotation, mask, 0.4)


    #merge images
    patch_padded=background*0.0
    patch_padded[sup_b:inf_b, left_b:right_b]=patch_im
    factor=saturate(mask*config['fusion_factor_synthetic'], 0.0, 1.1)
    result_im=saturate(background*(1.0-factor)+(patch_padded*factor), 0.0, 1.0)


    show_image(result_im, 'result_image', None, config['window_pos'])
    cv.waitKey()

    return [result_im, result_annotation]


def fusion_bloom_real(patch, background, annotation, pos, config):
    """ This function fusion a stylized patch with a background image so that the patch is inserted in the background
    in a realistic way
    Parameters:
        patch: patch image to be inserted in the background (already stylized patch)
        background: background image
        annotation: segmentation ground truth of the background image
        pos: position of the background image where the patch is going to be inserted (center of the patch)
        config: dictionary with a certain values configuration fot the program parameters"""

    sup_p, inf_p, left_p, right_p, sup_b, inf_b, left_b, right_b = get_intersection_rois(patch, background, pos)

    # get patch mask and image. An offset is added to the mask. The patch is colored.
    patch_mask = get_mask_from_real_bloom(patch)
    color=choose_color(background[sup_b:inf_b, left_b:right_b, :], config)
    patch_mask = colored(patch[sup_p:inf_p, left_p:right_p], patch_mask[sup_p:inf_p, left_p:right_p], color, config)

    # get mask of regions with water
    mask_water = (annotation == 1)*1.0 +(annotation == 5 )*1.0

    # process the patch
    product_patch = mask_water[sup_b:inf_b, left_b:right_b] * patch_mask
    smooth_patch = smooth_limits(product_patch, config['blurring'], sup_b, inf_b, left_b, right_b, background.shape)
    final_patch=product_patch*smooth_patch

    # get global mask
    mask = (0.0 * mask_water.copy())
    mask[sup_b:inf_b, left_b:right_b] = final_patch
    mask=saturate(mask, 0.0, 1.0)
    result_annotation=add_bloom_annotation(annotation, mask, 0.3)

    #merge images
    patch_padded=background*0.0
    patch_padded[sup_b:inf_b, left_b:right_b]=patch[sup_p:inf_p, left_p:right_p]
    factor=saturate(mask*config['fusion_factor_real'], 0.0, 1.1)
    result_im=saturate(background*(1.0-factor)+(patch_padded*factor), 0.0, 1.0)

    show_image(background, 'original_image', None, [600, 250])
    show_image(result_im, 'result_image', None, config['window_pos'])
    cv.waitKey()

    return [result_im, result_annotation]


def get_mask_from_real_bloom(patch):
    """ This function generates a gray-level image from the bloom patch. The gray_level image is going to be used as
    mask in the fusion process"""

    color_img=patch*255
    color_img=color_img.astype(np.uint8)
    gray_bloom = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    gray_bloom = saturated_sum(gray_bloom, 0, 200, 60)
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #mask_ch=clahe.apply(gray_bloom)##cv.equalizeHist(gray_bloom)/255.0
    mask_ch=cv.normalize(gray_bloom, None, 0, 255, cv.NORM_MINMAX)
    #mask_ch=saturated_sum(mask_ch, 0, 255, 20)

    #output with 3 channels
    mask = patch.copy()
    mask[:, :, 0] = mask_ch / 255.0
    mask[:, :, 1] = mask_ch / 255.0
    mask[:, :, 2] = mask_ch / 255.0
    return mask


# -------------------------------------------- COLORS AND LABELS -------------------------------------------------------
def choose_color(im, config):
    """ This function shows an assortment of colors to colore the bloom patch
    Parameters:
        im: region of the background image where the bloom patch is going to be placed
        config: dictionary with a certain values configuration fot the program parameters"""

    # MOUSE EVENT MANAGEMENT---------------
    global posList1
    posList1 = []
    def onMouseValue(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            posList1.append((x, y))

    # reference colors
    colors_ref = [[0, 0, 0],
                  [128, 255, 0],
                  [255, 255, 255],
                  [0, 255, 0],
                  [0, 0, 0],
                  [0, 255, 128],
                  [255, 255, 255],
                  [0, 255, 255],
                  [0, 0, 0],
                  [6, 213, 220],
                  [18, 84, 86],
                  [18, 86, 30],
                  [90, 229, 15],
                  [117, 229, 90],
                  [78, 106, 52],
                  [75, 152, 3]]
    n_colors=len(colors_ref)

    # GENERATION OF INTERMEDIATE COLOR BY INTERPOLATING RGB COORDINATES
    r_ref=[]
    b_ref=[]
    g_ref=[]
    for row in colors_ref:
        r_ref.append(row[0])
        g_ref.append(row[1])
        b_ref.append(row[2])

    ref_color_idx=np.linspace(0,(n_colors-1)*5,n_colors)
    color_idx=np.linspace(0, (n_colors-1)*5, (n_colors-1)*5+1)
    r=np.interp(color_idx, ref_color_idx, r_ref)
    g=np.interp(color_idx, ref_color_idx, g_ref)
    b=np.interp(color_idx, ref_color_idx, b_ref)



    # GENERATION OF COLORMAP
    [h, w, _]= im.shape
    colormap=np.zeros((60+h, max(((n_colors-1)*5+1)*20, w), 3), np.uint8)
    aux_im=im*255
    aux_im=aux_im.astype(np.uint8)
    colormap[60:, 0:w, :]=aux_im[:, :, :]
    for j in color_idx:
        i=int(j)
        colormap[:60, i*20:((i+1)*20)-1]=(int(r[i]), int(g[i]), int(b[i]))


    # USER SELECT A COLOR
    print('Choose a color and press any key')
    show_image(colormap, 'background_image', onMouseValue, config['window_pos'])
    cv.waitKey()
    if len(posList1)>0:
        point=posList1[-1]
        colorarray=colormap[point[1], point[0], :]
        return (colorarray[0], colorarray[1], colorarray[2])
    else:
        return None


def colored(im, mask, color, config):
    """ This function applies a certain color layer over an image.
    Parameters:
        im: original image that is going to be colored
        mask: the mask weights the proportion of new color against the original image at each pixel (same size as im)
        color: new color value
        config: dictionary with a certain values configuration fot the program parameters
    """

    if color is not None:
        # Generate color image (same size than im)
        color_im = np.zeros(im.shape, np.uint8)
        color_im[:, :] = color
        color_im=color_im/255.0
        cv.imshow('color', color_im)
        cv.imshow('mask', mask)
        cv.imshow('im', im)
        if not config['real_bloom']:
            # Extract texture from the original image
            texture=saturate((im-0.5*mask), 0.0, 1.0)

            # Colored the content mask and add texture
            result= saturate(config['color_weights_synthetic'][0]*(color_im)*mask +
                             config['color_weights_synthetic'][1]*im +
                             config['color_weights_synthetic'][2]*texture, 0.0, 1.0)
        else:
            result= saturate(config['color_weights_real'][0]*(color_im)*mask+
                             config['color_weights_real'][1]*im, 0.0, 1.0)
    else:
        result=im.copy()
    cv.imshow('result1', result)
    cv.waitKey()
    return result

def add_bloom_annotation(annotation, mask, th):
    """ This function adds the segmentation label of bloom (value 5 ) to the current annotations
    Parameter:
        annotation: image with the segmentation ground truth (without a class for blooms)
        mask: gray level images indicating the density of bloom at each pixel
        th: minimum threshold to classify a pixel with new class bloom"""
    result_annotation=annotation.copy()
    [H, W, _] = annotation.shape
    for y in range(H):
        for x in range(W):
            if (mask[y, x, 0]+mask[y,x,1]+mask[y,x,2])/3.0 > th:
                result_annotation[y, x,:]=(5, 5, 5)
    #output=(result_annotation==5)*1.0
    cv.imshow('annotation', result_annotation*50)
    #cv2.waitKey()
    return result_annotation

# ---------------------------------------------- SATURATION -----------------------------------------------------------

def saturated_sum(im, min, max, add):
    """ This function adds a certain value to every pixel and channel of an image and saturates the resulting values
    among a minimum and a maximum limit"""

    result=im.copy()
    if len(im.shape)==3:
        [h, w, ch]=im.shape
        # loops to access to every element
        for i in range(h):
            for j in range(w):
                for k in range(ch):
                    if im[i,j,k] + add > max:
                        result[i, j, k] = max
                    elif im[i,j,k] + add < min:
                        result[i, j, k] = min
                    else:
                        result[i, j, k] = im[i, j, k] + add

    if len(im.shape) == 2:
        [h, w] = im.shape
        # loops to access to every element
        for i in range(h):
            for j in range(w):
                if im[i, j] + add > max:
                    result[i, j] = max
                elif im[i, j] + add < min:
                    result[i, j] = min
                else:
                    result[i, j] = im[i, j] + add

    return result

def saturate(im, min, max):
    """ This function saturate the values of each channel and each pixel among a minimum and a maximum limit"""
    result=im.copy()
    [h, w, ch]=im.shape
    # loops to access to every element
    for i in range(h):
        for j in range(w):
            for k in range(ch):
                if im[i,j,k] > max:
                    result[i, j, k] = max
                elif im[i,j,k] < min:
                    result[i, j, k] = min
                else:
                    result[i, j, k] = im[i, j, k]
    return result


# ------------------------------------- FUNCTIONS RELYING IN NEURAL MODELS ---------------------------------------------

def get_water_style(background, annotation, sup_b, inf_b, left_b, right_b):
    """ This function get only the part of the image corresponding to water (or bloom) to get style from that and avoid
    the effect of other objects in style transfer process."""

    style_im= background[sup_b:inf_b, left_b:right_b, :]

    mask_water = (annotation == 1) * 1.0 + (annotation == 5) * 1.0  # Get mask of regions with water
    mask_water_region = mask_water[sup_b:inf_b, left_b:right_b, :]

    th_list=[0.06, 0.4, 0.2, 0.2]
    for th in th_list:
        # get indices of rows and columns that are completely black (0) = no water
        [rows, cols, _] = mask_water_region.shape
        sum_rows = np.sum(np.sum(mask_water_region, 1), 1) / (cols * 3)
        sum_cols = np.sum(np.sum(mask_water_region, 0), 1) / (rows * 3)

        #remove completely black rows of borders
        black_rows_idx= np.where(sum_rows<th)[0]
        black_rows_idx= black_rows_idx.tolist()
        if len(black_rows_idx)>0:
            i_sup=black_rows_idx[0]
        else:
            i_sup=0
        for i in black_rows_idx:
            if i-i_sup<=1:
                i_sup=i
        i_down = len(sum_rows)
        for i in reversed(black_rows_idx):
            if i_down-i <= 1:
                i_down = i
        style_im=style_im[i_sup:i_down, :, :]
        mask_water_region=mask_water_region[i_sup:i_down, :, :]


        # remove completely black cols of borders
        black_cols_idx = np.where(sum_cols < th)[0]
        black_cols_idx = black_cols_idx.tolist()
        if len(black_cols_idx)>0:
            i_left = black_cols_idx[0]
        else:
            i_left=0
        for i in black_cols_idx:
            if i - i_left <= 1:
                i_left = i
        i_right = len(sum_cols)
        for i in reversed(black_cols_idx):
            if i_right - i <= 1:
                i_right = i
        style_im=style_im[:, i_left:i_right, :]
        mask_water_region=mask_water_region[:, i_left:i_right, :]

    return style_im


def apply_style(patch, background, annotation, pos, config):
    """ This function get the region of the background image where the patch is going to be inserted and use that region
    as style reference. Then the patch is stylized with the reference style.
    Returns:
        stylized_im: patch image with the style og the area of the background indicated by pos"""

    # Get intersection coordinates
    sup_p, inf_p, left_p, right_p, sup_b, inf_b, left_b, right_b= get_intersection_rois(patch, background, pos)

    # Get style and content image
    style_im= get_water_style(background, annotation, sup_b, inf_b, left_b, right_b)* 255
    style_im = style_im.astype(np.uint8)
    content_im = patch[sup_p:inf_p, left_p:right_p, :]*255
    content_im=content_im.astype(np.uint8)


    # load and apply style transfer neural model
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_im= stylize_image(content_im, style_im, hub_model)/255.0

    # Show image resulting of inserting the stylized patch
    sum_im = put_patch(stylized_im, background, pos)
    show_image(sum_im, 'background_image', None, config['window_pos'])
    cv.waitKey()

    return(stylized_im)


def stylize_image(content_im, style_im, model):
    """ This function uses a neural model to transfer style from a style reference to a content image
    Parameters:
         content_im: image with the content. This is going to be stylized
         style:im: image with the style reference
         model: pretrained transfer style neural model
    Returns:
         stylized_im: content image after style transfer
        """

    #prepare inputs
    content_tensor=img2netInput(content_im)
    style_tensor=img2netInput(style_im)

    # load and apply model
    stylized_tensor = model(tf.constant(content_tensor), tf.constant(style_tensor))[0]

    # prepare output
    stylized_im = tensor_to_imageCV(stylized_tensor)
    stylized_im = scale_im(stylized_im, content_im)
    return stylized_im


# -------------------------------------------- TYPE CONVERSIONS --------------------------------------------------------

def img2netInput(source_img):
    """ This function converts an image in a tensor with the proper size for the used model
    Returns:
        tensor
    """

    # Convert image type
    img = tf.image.convert_image_dtype(source_img/255.0, tf.float32)

    # scale image
    max_dim = 512
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    # generate tensor
    tensor = tf.image.resize(img, new_shape)
    tensor = tensor[tf.newaxis, :]
    return tensor


def tensor_to_imageCV(tensor):
    """
    This function converts a tensor into a image in the format of openCV
    """
    tensor = tensor*255
    image = np.array(tensor, dtype=np.uint8)
    if np.ndim(image)>3:
        assert image.shape[0] == 1
        image = image[0]
    return image


# ---------------------------------------------- IO FUNCTIONS ----------------------------------------------------------

def show_image (image, winname, mouse_function, pos):
    """ This function shows an image in a cv2 window with the given name (winname).
    A mouse event can be managed in the window with the given function (mouse function)"""
    cv.namedWindow(winname)
    if mouse_function is not None:
        cv.setMouseCallback(winname, mouse_function)
    cv.moveWindow(winname, pos[0], pos[1])
    cv.imshow(winname, image)

def load_image(path):
    """ This function loads the image in the given path"""
    if (path.count('.CR2') > 0):
        raw = rawpy.imread(path)  # access to the RAW image
        rgb = raw.postprocess()  # a numpy RGB array
        im = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)  # the OpenCV image
    else:
        im = cv.imread(path)
    return im






""" 
def readInput( caption, default, timeout = 2):

    start_time = time.time()
    sys.stdout.write('%s(%s):'%(caption, default))
    sys.stdout.flush()
    input = ''
    while True:
        if msvcrt.kbhit():
            byte_arr = msvcrt.getche()
            if ord(byte_arr) == 13: # enter_key
                break
            elif ord(byte_arr) >= 32: #space_char
                input += "".join(map(chr,byte_arr))
        if len(input) == 0 and (time.time() - start_time) > timeout:
            print("timing out, using default value.")
            break

    print('')  # needed to move to next line
    if len(input) > 0:
        return input
    else:
        return default


def padding(pad_factor, im):
    if pad_factor > 0.0:
        [h, w, _] = im.shape
        margin = int(np.round(pad_factor * ((h + w) / 2.0)))
        padded = np.zeros((h + 2 * margin, w + 2 * margin, 3), np.float64)
        padded[margin:margin + h, margin:margin + w, :] = im
        #smooth borders
        output=padded.copy()
        [H, W, _]=output.shape
        for i in range(h):
            for j in range(w):
                f=int(np.round(1.2*margin))-i
                c=int(np.round(1.2*margin))-j
                m=3
                output[f, c, :]=np.sum(padded[f-m:f+m, c-m:c+m])/((2*m+1)*(2*m+1))
                f=H-(int(np.round(1.2*margin))-i)
                c=W-(int(np.round(1.2*margin))-j)
                m=3
                output[f, c, :]=np.sum(padded[f-m:f+m, c-m:c+m])/((2*m+1)*(2*m+1))
        cv2.imshow('soft_pad', output)
        cv2.waitKey()
        return output
    else:
        return im





#def img2netInput(source_img):
#  max_dim = 512
#  np_image_data = np.asarray(source_img)
#  img = tf.convert_to_tensor(np_image_data, dtype=tf.float32)
#  #img = tf.image.decode_image(img, channels=3)
#  img = tf.image.convert_image_dtype(img, tf.float32)

#  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
#  long_dim = max(shape)
#  scale = max_dim / long_dim

#  new_shape = tf.cast(shape * scale, tf.int32)

#  img = tf.image.resize(img, new_shape)
#  img = img[tf.newaxis, :]
#  return img



def tensor_to_imagePIL(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


"""

"""
def smooth_limits(patch, blurring, sup_b, inf_b, left_b, right_b, b_shape):
    This function generates a mask to smooth the borders of a patch, except when the patch borders are also the
    background borders.
    Parameters:
        patch: image whose borders are going to be smoothed
        blurring: percentage of border smoothed. 1 corresponds to the half of the smaller side.
        sup_b, inf_b, left_b, right_b= coordinates of the patch in the background image
        b_shape = shape of the background image
    Returns:
        final_mask: mask to smooth the borders of a patch


    #def onMouseValue(event, x, y, flags, param):
        # global posList
        #if event == cv.EVENT_LBUTTONDOWN:
        #    print(patch[x,y])

    #cv2.imshow('mask', patch)
    #cv2.waitKey()

    # Get parameters for the smoothing equation: patch and background shape, max_distance, starting_smooth
    [h_o, w_o, _] = patch.shape
    [H, W, _] = b_shape
    max_distance = min([int(np.round(w_o/2.0)), int(np.round(h_o/2.0))]) # half of the smaller side
    starting_smooth = 1 - blurring    # Distance from the center of the image to the starting of smoothing equation
                                      # (as factor from 0 to 1)
    abs_border=int(np.round(blurring*max_distance)) # width of the smooth border
    # Add pad for avoiding smoothing the limits of patch that corresponds to background limits
    margin_top=0
    margin_bottom=0
    margin_left=0
    margin_right=0
    if (sup_b == 0):
        margin_top = int(np.round(abs_border*6))
    if inf_b == H - 1:
        margin_bottom = int(np.round(abs_border*6))
    if left_b == 0:
        margin_left = int(np.round(abs_border*6))
    if right_b == W - 1:
        margin_right = int(np.round(abs_border*6))

    if (sup_b < abs_border):
        margin_top = int(abs_border-sup_b)
    if inf_b > H - 1 - abs_border:
        margin_bottom = int(abs_border-(H-1-inf_b))
    if left_b<abs_border:
        margin_left = int(abs_border-left_b)
    if right_b > W - 1 - abs_border:
        margin_right = int(abs_border-(W-1-right_b))
    padded = cv.copyMakeBorder(patch, margin_top, margin_bottom, margin_left, margin_right, cv.BORDER_CONSTANT)
    [h, w, _] = padded.shape

    # create a mask to smooth the borders
    mask = np.ones(padded.shape, np.float64)
    b = 1.0/(starting_smooth-1.0)   # parameters of the smooth equation
    m = -1.0*b
    center_x = np.round((w - margin_right - margin_left) / 2.0 + margin_left)
    center_y = np.round((h - margin_bottom - margin_top) / 2.0 + margin_right)

    print(center_x)
    print(w)
    print(margin_left)
    print(margin_right)
    for r in range(h):
        for c in range(w):
            dist = min_distance_to_center(h, w, center_y, center_x, r, c)
            if dist < starting_smooth*max_distance: factor = 1.0
            else: factor = m+b*(dist/max_distance)
            mask[r, c, :] = factor#mask[r, c, :]*factor

    #crop to remove the added margins
    cropped_mask=mask[margin_top:margin_top+h_o, margin_left: margin_left+w_o]
    final_mask =saturate(cropped_mask, 0.0, 1.0)
    #final_mask= final_mask.astype(np.uint8)

    #show_image(final_mask, 'soft_mask', onMouseValue, -40, 250)
    #cv2.waitKey()
    return final_mask
"""

