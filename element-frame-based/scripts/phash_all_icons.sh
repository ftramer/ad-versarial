#!/usr/bin/env bash

for filename in ../data/web/ad_logos/icon/*.png; do
    python phash/attack.py "$filename" ../data/web/ad_logos/icon/adv_PHASH/ ../data/ad_logos/
done