# Code for data collection and training

### Setup

You need to install 

```bash
apt-get install python-selenium chromium-chromedriver mongodb
```

and

```bash
pip3 install requirements.txt
```

### Data collection

First, you want to collect website screenshots with replaced ads. Download the latest [easylist](https://easylist.to/easylist/easylist.txt), and optionally obtain the database of ghostery (Install ghostery in any browser). Then create a file with one URL per line and use:

```bash
python3 crawler.py <urls file> <database> --replace-others --easylist easylist.txt --ghostery <path to ghostery>/databases/bugs.json
```

Do not use a database that is used for other purposes. The `--clear-database` option will delete all data from the database before using it.

The screenshots will be placed under `data`. Sort through the screenshots/templates and check that all ads were replaced and the monochrome boxes were detected correctly. Write all pathes to the template's directories (containing `main.png` and `main-boxes.json`) in a file.

Create a directory and place all advertisements that you want to use in it. A larger collection of ads can be found under [/external/hussain_ads](/external/README.md).

```bash
python3 generator.py <templates file> <ads directory>
```

`-n x` will create `x` different images from one screenshot with different ads. `--recreate` will index the ads again which is required when new ads are added after running the script for the first time (this takes some time). There are some constants at the beginning of the script that you might want to tweak.

Images will be placed under `images` and label files under `labels`. Split this data with a method of your choice and create a list of images for training and a list of images for validation. The label file for image `image.png` has to be placed at `../labels/image.txt`.


### Training

Download and compile [YOLO](https://pjreddie.com/darknet/yolo/) with GPU support. The configuration files we used can be found under [yolo-files](yolo-files). You have to enter the paths to your training/validation list in [voc.data](yolo-files/voc.data). Download the pretrained weights

```bash
wget https://pjreddie.com/media/files/darknet53.conv.74
```

Start the training with

```bash
./darknet detector train voc.data yolov3-voc.cfg darknet53.conv.74
```

If you run into memory problems you can reduce `batch` and `subdivisions` in `yolov3-voc.cfg`. We had the best results after 3600 iterations but you should train for more iterations and later evaluate all of them and pick the best.

To evaluate our weights, we used [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), a fork of the original project. Set `batch` and `subdivisions` to 1 and use:

```bash
./darknet detector map voc.data yolov3-voc.cfg <weights file>
```
