#!/usr/bin/env bash

for filename in ../data/web/ad_logos/text/*.png; do
    python sift/black_box_attack.py "$filename" ../data/web/ad_logos/text/adv_SIFT/ ../data/ad_logos/
done