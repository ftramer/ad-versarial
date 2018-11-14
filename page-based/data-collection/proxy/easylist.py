#! /usr/bin/env python3

from adblockparser import AdblockRules
import requests
import logging

"""Get an AdblockRules object and call should_block(url) on it to find out
whether it would be blocked by a rule."""


def load_easylist():
    """Fetches the most recent easylist and returns an AdblockRules object
    using it."""
    r = requests.get("https://easylist.to/easylist/easylist.txt")
    r.raise_for_status()
    easy = r.text
    return AdblockRules(easy.splitlines())


def load_file(file):
    """Returns an AdblockRules object using the rules specified in file."""
    with open(file) as f:
        rules = f.readlines()
        return AdblockRules(rules)
