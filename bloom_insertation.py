# This code integrates a synthetic bloom in a real background image, using the style of the region where it is going to
# be placed (use blooms with gray scale)

#-----------------------------------------------IMPORTS----------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')
from os import listdir
from os.path import isfile, join
from datetime import datetime
from im_processing import *

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


#-----------------------------------------------FUNCTIONS---------------------------------------------------------------

def insert_bloom(background, annotation, bloom, config):
    """ This function inserts a bloom patch in a background image
    Parameters:
        background: background image
        annotation: segmentation ground truth of the background image
        bloom: image with the bloom patch to insert
        config: dictionary with a certain values configuration fot the program parameters """

    # MOUSE EVENT MANAGEMENT
    global insertion_pos
    insertion_pos = []

    def onMouseMain(event, x, y, flags, param):
        # global posList
        if event == cv.EVENT_LBUTTONDOWN:
            insertion_pos.append((x, y))

    # LOOP TO GET THE INSERTION POSITION
    while True:
        print('click to insert the bloom and press "enter": ')
        show_image(background, 'background_image', onMouseMain, config['window_pos'])
        key=cv.waitKey()
        if key == 13 and len(insertion_pos)>0:
            cv.destroyWindow('background_image')
            break

    # INSERT PROCESS (once the insertation position has been defined)
    # 1. zoom and locate the patch
    zoomed_im, pos = scale_and_locate(bloom, background, insertion_pos[-1])

    # 2. apply style to the patch
    if not config['real_bloom']:
        stylized_im = apply_style(zoomed_im, background, annotation, pos, config)


    # 3. fusion of the patch and the background
    if config['real_bloom']:
        [result_im, result_annotation] = fusion_bloom_real(zoomed_im, background, annotation, pos, config)
    else:
        [result_im, result_annotation] = fusion_bloom_synthetic(stylized_im, zoomed_im, background, annotation, pos, config)

    # 4. Prepare the output
    result_im=result_im*255
    result_im=result_im.astype(np.uint8)
    return [result_im, result_annotation]


def get_synthetic_image(background, annotation, bloom_folder_path, bloom_idx, config):
    """ This function returns a synthetic image and its annotation.
    This function allows to insert several blooms in a background image
    Parameters:
        background: image of the background where blooms are inserted
        annotation: segmentation ground truth of the background
        bloom_folder_path: path to the folder containing all the bloom patches
        bloom_idx: index to the bloom being inserted currently
        config: dictionary with a certain values configuration fot the program parameters"""

    # MOUSE EVENT MANAGEMENT
    global posList1, posList2
    posList1 = []
    posList2 = []

    def onMouseCrop(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(posList1) > len(posList2):
                posList2.append((x, y))
            else:
                posList1.append((x, y))

    # CODE TO GET A SYNTHETIC IMAGE ------------------------------------------------------------------------------------
    # get the names of the bloom files
    bloom_images = [f for f in listdir(bloom_folder_path) if isfile(join(bloom_folder_path, f))]

    # Show the background image
    show_image(background, 'background_image', None, config['window_pos'])
    cv.waitKey(200)

    # ------------- LOOP to select and insert bloom patches in the background image ------------------------------------
    background_im=background.copy()
    annotation_im=annotation.copy()
    while True:
        # GET THE BLOOM AND SHOW IT
        bloom_name = bloom_images[bloom_idx]
        bloom_im = load_image(os.path.join(bloom_folder_path, bloom_name))/255.0
        print('Press 4(<-) or 6(->) to select a new bloom. Select the crop and press "enter" to crop the image.'
              'Simply press "enter" to pass to the next background')
        show_image(bloom_im, 'bloom_image', onMouseCrop, [-20, -20])
        key=cv.waitKey()

        # OPERATION OPTIONS
        # CROP AND INSERT THE BLOOM PATCH
        if key == 13:
            if len(posList1)!=0 and len(posList2)!=0:
                cv.destroyWindow('bloom_image')
                cropped_im = cropping(bloom_im, posList1, posList2, 0.0)
                [result_im, result_annotation]=insert_bloom(background_im, annotation_im, cropped_im, config)
                # EDITING END
                print('Press "s" to save the image, "c" to cancel the edition, any other key to continue editing')
                key = cv.waitKey()
                if key == 115: break
                elif key == 99:
                    result_im = None
                    result_annotation = None
                    cv.destroyAllWindows()
                else:
                    background_im = result_im.copy()/255.0
                    annotation_im = result_annotation.copy()
                    cv.destroyAllWindows()
            else:
                result_im = None
                result_annotation = None
                break

        # SELECTION OF THE BLOOM: CHANGE THE BLOOM INDEX
        if key == 52: bloom_idx = bloom_idx-1
        if key == 54: bloom_idx = bloom_idx+1
        if bloom_idx < 0: bloom_idx = len(bloom_images)-1
        if bloom_idx > len(bloom_images)-1: bloom_idx = 0

    # -----------------------------------------  END LOOP  -------------------------------------------------------------
    return [result_im, result_annotation]


def automatic_bloom_insertation(bloom_folder_path, background_folder_path, annotations_folder_path, config):
    """ This function shows background images from a segmentation dataset. The user can insert one or more blooms in
    each background image and save the resulting image. Annotations of the new image will be also automatically saved.
    Parameters:
        bloom_folder_path: path to the folder containing the (synthetic) bloom images
        background_folder_path: path to the folder containing the background images (aquatic environments)
        annotations_folder_path: path to the folder containing the segmentation annotations for the background images
                                corresponding annotations and background images share names
        config: dictionary with a certain values configuration fot the program parameters
        """

    # CREATE NEW FOLDER FOR THE NEW DATASET (IMAGES AND GROUND TRUTH)
    synthetic_images_folder_path = os.path.join(background_folder_path[:background_folder_path.rfind('/')],
                                                'synthetic_images')
    if not os.path.exists(synthetic_images_folder_path):
        os.mkdir(synthetic_images_folder_path)
    synthetic_annotations_folder_path = os.path.join(background_folder_path[:background_folder_path.rfind('/')],
                                                     'synthetic_annotations')
    if not os.path.exists(synthetic_annotations_folder_path):
        os.mkdir(synthetic_annotations_folder_path)


    # GET NAMES OF FILES IN background_folder_path
    background_images = [f for f in listdir(background_folder_path) if isfile(join(background_folder_path, f))]

    # ITERATE THE BACKGROUND IMAGES
    bloom_idx=0     # index of the first bloom shown (all the blooms are available to be inserted in a background image)
    exit=False
    for background_name in background_images:
        if not(exit):
            #--------------------------------START BACKGROUND IMAGE PROCESSING----------------------------------------------
            try:

                # load images
                background_im = load_image(os.path.join(background_folder_path, background_name))/255.0
                annotation_im = load_image(os.path.join(annotations_folder_path, background_name))
                while True:
                    # get a synthetic image from the current background
                    [synthetic_image, synthetic_annotation]=get_synthetic_image(background_im, annotation_im,
                                                                                bloom_folder_path, bloom_idx, config)
                    # save the synthetic image and its annotations
                    if synthetic_image is not None:
                        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
                        synthetic_image_file = background_name[:-4] + '_' + date + '.jpg'
                        synthetic_image_path = os.path.join(synthetic_images_folder_path, synthetic_image_file)
                        cv.imwrite(synthetic_image_path, synthetic_image)
                        synthetic_annotation_file = background_name[:-4] + '_' + date + '.jpg'
                        synthetic_annotation_path = os.path.join(synthetic_annotations_folder_path,
                                                                 synthetic_annotation_file)
                        cv.imwrite(synthetic_annotation_path, synthetic_annotation)

                    # Ask for a new background image, or exit.
                    # If a different key is pressed, the user can continue inserting blooms in the current background
                    print('Press "enter" to pass to the next background. Press any other key to continue inserting '
                          'blooms in the same background. Press "esc" to exit')
                    key2=cv.waitKey()
                    cv.destroyAllWindows()
                    if key2== 27:
                        exit=True
                        break
                    if key2== 13:
                        break
            except:
                break
        #--------------------------------END BACKGROUND IMAGE PROCESSING------------------------------------------------




#------------------------------------------------ MAIN  ----------------------------------------------------------------
config={
    'real_bloom': False,                    # boolean. 1 if the bloom image is real, 0 if the bloom image is synthetic
    'window_pos': [-40, 250],               # position of the pop-up windows [x, y]
    'color_weights_real':[0.5, 0.5],        # weights of mask, and patch image
    'color_weights_synthetic': [0.4, 0.4, 0.2], # weights of mask, patch image and texture
    'blurring': 0.4,                        # size (as factor of the image size) of the blurred edge
    'fusion_factor_real': 1.3,
    'fusion_factor_synthetic': 0.65
}

bloom_folder_path='/home/ia_vision/PycharmProjects/StyleTransfer/content_images/set3' #synthetic blooms
#bloom_folder_path='/home/ia_vision/PycharmProjects/StyleTransfer/style_images/set1' #Real blooms
background_folder_path='/home/ia_vision/PycharmProjects/StyleTransfer/segmentation_dataset/background_images'
annotations_folder_path='/home/ia_vision/PycharmProjects/StyleTransfer/segmentation_dataset/annotations'

automatic_bloom_insertation(bloom_folder_path, background_folder_path, annotations_folder_path, config)