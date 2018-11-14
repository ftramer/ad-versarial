# Code for training, evaluating and attacking a page-based perceptual ad-blocker.

The code in this directory used to load and run the YOLO-v3 model is inspired 
by [tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3).

### Data Collection and Training

The scripts for data collection can be found under [data-collection].

You need to install 

```bash
apt-get install python-selenium chromium-chromedriver mongodb
```

First, you want to collect website screenshots with replaced ads. Download the latest [easylist](https://easylist.to/easylist/easylist.txt), and optionally obtain the database of ghostery (Install ghostery in any browser). Then create a file with one URL per line and use:

```bash
python3 crawler.py <urls file> <database> --replace-others --easylist easylist.txt --ghostery <path to ghostery>/databases/bugs.json
```

The screenshots will be placed under `data`. Sort through the screenshots/templates and check that all ads were replaced and the monochrome boxes were detected correctly. Write all pathes to the template's directories (containing `main.png` and `main-boxes.json`) in a file. Create a directory and place all advertisements that you want to use in it.

```bash
python3 generator.py <templates file> <ads directory>
```

`-n x` will create `x` different images from one screenshot with different ads. `--recreate` will index the ads again which is required when new ads are added after running the script for the first time (this takes some time). There are some constants at the beginning of the script that you might want to tweak.

Images will be placed under `images` and label files under `labels`. Split this data with a method of your choice and create a list of images for training and a list of images for validation. The label file for image `image.png` has to be placed at `../labels/image.txt`.

Download and compile [YOLO](https://pjreddie.com/darknet/yolo/) with GPU support. The configuration files we used can be found under [data-collection/yolo-config]. You have to enter the paths to your training/validation list in [voc.data](data-collection/yolo-config/voc.data). Download the pretrained weights

```bash
wget https://pjreddie.com/media/files/darknet53.conv.74
```

Start the training with

```bash
./darknet detector train voc.data yolov3-voc.cfg darknet53.conv.74
```

If you run into memory problems you can reduce `batch` and `subdivisions` in `yolov3-voc.cfg`. We had the best results after 3600 iterations but you should train for more iterations and later evaluate all of them and pick the best.

To evaluate our weights, we used [https://github.com/AlexeyAB/darknet] with the command:

```bash
./darknet detector map voc.data yolov3-voc.cfg <weights file>
```


### Evaluation

A pre-trained classifier is available from https://github.com/ftramer/ad-versarial/releases
The below script expects `page_based_yolov3.weights` to be placed under [../models](../models).
The collected web data contained in the release should be under [../data](../data).

To evaluate the model on 20 screenshots of news websites 
(with outputs in a newly created directory `temp`), use:

```bash
python classify.py --input_dir=../data/page_based/web/test/ --output_dir=temp
```

### Attacks

Below is a list of commands for reproducing different attacks considered in our paper.
The attack samples will be saved in subdirectories under a global `output` directory.

#### Universal attacks for all webpages:

- *Evasion Attack (C4U in the paper)*: The publisher overlays a transparent mask on the full webpage to evade ad-blocking:
```bash
python -m attacks.web_evade_all_ads_overlay --input_dir=../data/page_based/web/
```

- Test the generated mask when scrolling over a full webpage:
```bash
python -m attacks.web_test_scrolling --full_page=../data/page_based/nytimes_full.png --mask=output/overlay/mask_
```

- *Evasion Attack (C4U' in the paper)*: The publisher overlays a transparent mask on the full 
webpage to disable ad-blocking by overflowing it with inccorect predictions:
```bash
python -m attacks.web_overflow_all_overlay --input_dir=../data/page_based/web/mask_100.png
```

- *Detection Attack (C1U in the paper)*: The publisher adds a footer added on the bottom 
of a page that triggers a false ad prediction:
```bash
python -m attacks.web_footer_false_positive --input_dir=../data/page_based/web/
```

#### Attacks fine-tuned for bbc.com:

- *Evasion Attack (C4 in the paper)*: The publisher perturbs bottom of ad frame to evade 
ad-blocking: 
```bash
python -m attacks.bbc_evade --input_dir=../data/page_based/bbc/
```

- *Evasion Attack (C3 in the paper)*: The ad-network perturbs ads using a universal 
perturbation to evade ad-blocking: 
```bash
python -m attacks.bbc_evade_ad_network --input_dir=../data/page_based/bbc/
```

- *Detection Attack (C1 in the paper)*: Publisher perturbs the page header to create a 
false ad prediction:
```bash
python -m attacks.bbc_false_positive --input_dir=../data/page_based/bbc/
```

#### Demo of abuse attack on Facebook
Include [keras-yolo3](keras-yolo3):

```bash
export PYTHONPATH=$PYTHONPATH:keras-yolo3
```

Convert model to Keras:
```bash
python -m keras-yolo3.convert ../models/page_based_yolov3.cfg ../models/page_based_yolov3.weights ../models/page_based_yolov3.h5
```

Run attack (A1 in the paper):
```bash
python -m attacks.fb_abuse 
```
