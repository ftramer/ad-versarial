#!/usr/bin/env bash

for filename in ../data/web/ad_logos/text/*.png; do
    python OCR/evade_or_fp_attack.py --image="$filename" \
        --target_height=30 --target=adchoices --const=0.1 --iter=200 --lr=1.0 \
        --output_dir=../data/web/ad_logos/text/adv_OCR/ \
        --threshold=5
done