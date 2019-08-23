# Code for training, evaluating and attacking a page-based perceptual ad-blocker.

The code in this directory used to load and run the YOLO-v3 model is inspired 
by [tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3).

### Data Collection and Training

Our scripts for data collection can be found in [data-collection](data-collection).


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
