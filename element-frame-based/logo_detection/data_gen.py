from utils import *


INPUT_DATA_DIR = "../external/hussain_ads/"
OUTPUT_DATA_DIR = "logo_detection/data"

AD_LOGOS_DATA_DIR = "../data/ad_logos/"
CLOSE_X = "../data/adchoices_google_x.png"
CLOSE_X_GRAY = "../data/adchoices_x_gray.png"


def resize_adchoices(logo, target_height=15):
    curr_height = logo.shape[0]
    mul = target_height / (1.0 * curr_height)
    return cv2.resize(logo, None, fx=mul, fy=mul)


def is_dark(patch):
    return np.mean(patch.astype(np.float32)) < 50


def get_logo_coordinates(base_ad, logo_img, max_padding):
    base_height, base_width = base_ad.shape[:2]
    logo_height, logo_width = logo_img.shape[:2]

    max_padding = min(max_padding, base_height//2, base_width//2)

    # choose corner, biased towards top right
    corner = np.random.choice([0, 1, 2, 3], p=[0.2, 0.6, 0.1, 0.1])
    padding_h = np.random.randint(0, max_padding + 1)
    padding_w = np.random.randint(0, max_padding + 1)

    if corner == 0:  # top left
        location_x = padding_w
        location_y = padding_h
    elif corner == 1:  # top right
        location_x = base_width - logo_width - padding_w
        location_y = padding_h
    elif corner == 2:  # bottom left
        location_x = padding_w
        location_y = base_height - logo_height - padding_h
    else:  # bottom right
        location_x = base_width - logo_width - padding_w
        location_y = base_height - logo_height - padding_h

    return location_x, location_y


def generate_dataset(ad_dataset_path, logos_dataset_path, close_x_path, close_x_gray_path,
                     output_dir, pos_frac, test_frac):

    blank_ads = get_image_paths(ad_dataset_path)
    adchoice_logos = [cv2.imread(adchoice_logo, -1) for adchoice_logo in get_image_paths(logos_dataset_path)]

    print(len(blank_ads))
    print(len(adchoice_logos))

    safe_mkdir(output_dir + '/adchoices/')

    close_x = cv2.imread(close_x_path, -1)
    close_x_gray = cv2.imread(close_x_gray_path, -1)

    count = 0
    non_rgb = 0
    too_large = 0
    pos_class = 0
    neg_class = 0

    reps = 2

    for ad_full_path in blank_ads:

        was_neg = False

        for i in range(reps):
            is_pos = was_neg or np.random.rand() <= pos_frac

            img = cv2.imread(ad_full_path, -1)
            if is_gray(img) or has_alpha(img):
                non_rgb += 1
                continue

            if img.shape[0] * img.shape[1] > 500*1000:
                too_large += 1
                continue

            if is_pos:
                pos_class += 1
                # Overlay an AdChoices logo

                ad_logo = np.random.choice(adchoice_logos, 1)[0].copy()
                ad_logo = resize_adchoices(ad_logo)

                if np.random.rand() < 0.05:
                    ad_logo = cv2.cvtColor(to_gray(ad_logo), cv2.COLOR_GRAY2BGR)

                # check if logo is transparent
                if has_alpha(ad_logo):

                    # randomly add a white background to transparent logos
                    white_background = np.random.rand() < 0.2
                    if white_background:
                        ad_logo = blend_white(ad_logo)

                # randomly add a "close" logo
                close_logo = None

                if not has_alpha(ad_logo):
                    close_logo = np.random.choice([None, close_x, close_x_gray], p=[0.6, 0.2, 0.2])

                if close_logo is not None:
                    transparent_padding = np.zeros((15, 1, 4), dtype=np.uint8)
                    ad_logo = np.concatenate([cv2.cvtColor(ad_logo, cv2.COLOR_BGR2BGRA),
                                              transparent_padding,
                                              cv2.cvtColor(close_logo, cv2.COLOR_BGR2BGRA)], axis=1)

                if np.random.rand() < 0.1:
                    max_padding = 150
                else:
                    max_padding = 2
                loc_x, loc_y = get_logo_coordinates(img, ad_logo, max_padding=max_padding)

                if has_alpha(ad_logo) and \
                    is_dark(img[loc_y:loc_y+ad_logo.shape[0], loc_x:loc_x+ad_logo.shape[1]]):
                    ad_logo = blend_white(ad_logo)

                # overlay the logo on the ad
                if has_alpha(ad_logo):
                    img[loc_y:loc_y+ad_logo.shape[0], loc_x:loc_x+ad_logo.shape[1]] = \
                        blend_transparent(img[loc_y:loc_y+ad_logo.shape[0], loc_x:loc_x+ad_logo.shape[1]], ad_logo)
                else:
                    img[loc_y:loc_y + ad_logo.shape[0], loc_x:loc_x + ad_logo.shape[1]] = ad_logo

                x_center = (loc_x + ad_logo.shape[1] // 2) / (1.0 * img.shape[1])
                y_center = (loc_y + ad_logo.shape[0] // 2) / (1.0 * img.shape[0])

                width = ad_logo.shape[1] / (1.0 * img.shape[1])
                height = ad_logo.shape[0] / (1.0 * img.shape[0])

                label_str = "{} {} {} {} {}".format(0, x_center, y_center, width, height)

            else:
                neg_class += 1
                label_str = ""
                was_neg = True

            cv2.imwrite(output_dir + '/adchoices/{}.png'.format(count), img)
            with open(output_dir + '/adchoices/{}.txt'.format(count), 'w') as f:
                f.write(label_str)

            count += 1

            if count % 100 == 0:
                print("parsed {} files".format(count))

    print("non_rgb = {}".format(non_rgb))
    print("too_large = {}".format(too_large))
    print("pos = {}".format(pos_class))
    print("neg = {}".format(neg_class))

    train_size = int((1.0 - test_frac) * count)
    with open(output_dir + '/train.txt', 'w') as f:
        f.writelines([output_dir + "/adchoices/{}.png\n".format(c) for c in range(train_size)])

    with open(output_dir + '/test.txt', 'w') as f:
        f.writelines([output_dir + "/adchoices/{}.png\n".format(c) for c in range(train_size, count)])


if __name__ == "__main__":
    # set seed to make all processes deterministic
    np.random.seed(1)
    training = False

    generate_dataset(INPUT_DATA_DIR, AD_LOGOS_DATA_DIR, CLOSE_X, CLOSE_X_GRAY, OUTPUT_DATA_DIR,
                     pos_frac=0.5, test_frac=0.1)
