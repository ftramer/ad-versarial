
import cv2
import numpy as np
import glob
import os


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def blend_white(img):
    all_white = 255 * np.ones_like(img[:, :, :3])
    return blend_transparent(all_white, img)


def blend_transparent(face_img, overlay_t_img):

    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    assert(overlay_mask.shape[-1] == 1)
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def to_gray(logo):
    patch_gray = logo
    if not is_gray(logo):
        if has_alpha(logo):
            logo = blend_white(logo)
        patch_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    return patch_gray


def to_alpha(logo):
    if has_alpha(logo):
        return logo

    if is_gray(logo):
        return cv2.cvtColor(logo, cv2.COLOR_GRAY2BGRA)
    else:
        return cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)


def to_bgr(img):
    if has_alpha(img):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif is_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        return img


def is_gray(logo):
    return len(logo.shape) == 2


def has_alpha(logo):
    return (len(logo.shape) == 3) and (logo.shape[-1] == 4)


IMG_EXTENSIONS = ["png", "jpg", "jpeg"]


def get_image_paths(glob_path):
    img_paths = []

    if '.' in glob_path and glob_path.split('.')[-1] in IMG_EXTENSIONS:
        img_paths.append(glob_path)
    else:
        for ext in IMG_EXTENSIONS:
            img_paths += glob.glob(glob_path + '/*.{}'.format(ext))

    return img_paths
