#!/usr/bin/env bash

for filename in ../data/web/ad_logos/text/*.png; do
    python phash/attack.py "$filename" ../data/web/ad_logos/text/adv_PHASH/ ../data/ad_logos/
done