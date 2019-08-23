#!/usr/bin/env python3

import re
import logging


def _processUrl(src):
    anchor = None
    src_host = None
    src_cleaned = None
    src_protocol = None

    index = src.find('#')
    if index >= 0:
        anchor = src[index + 1:]
        src = src[0:index]

    index = src.find('?')
    if index >= 0:
        src = src[0:index]

    src_cleaned = src

    index = src.find('http://')
    if index == 0:
        src_protocol = src[0:4]
        src = src[7:]
    else:
        index = src.find('https://')
        if index == 0:
            src_protocol = src[0:5]
            src = src[8:]
        else:
            index = src.find('//')
            if index == 0:
                src = src[2:]

    src = src.lower()
    index = src.find('/')
    r = src.find('@')

    if r >= 0 and (index == -1 or r < index):
        src = src[r + 1:]
        index = src.find('/')

    src_host = src[0:index] if index >= 0 else src
    src_path = src[index + 1:] if index >= 0 else ''

    index = src_host.find(':')
    if index >= 0:
        src_host = src_host[0:index]

    return {'protocol': src_protocol,
            'host': src_host,
            'path': src_path,
            'host_with_path': src,
            'anchor': anchor,
            'host_with_path_cleaned': src_cleaned
            }


class matcher:
    """Class to match requests to a ghostery bugId"""
    def __init__(self, bugs):
        """bugs is a json object read from a ghostery bugs database.
        The database can be found in the ghostery extension folder under '.../8.0.3.1_0/databases/bugs.json'"""
        self.bugs = bugs

    def stats(self):
        """Prints some stats about the database"""
        bugs = self.bugs
        version = bugs["version"]
        nApps = len(bugs["apps"])
        nBugs = len(bugs["bugs"])
        print("Version: {}, Apps: {}, Bugs: {}".format(version, nApps, nBugs))

    def isBug(self, src):
        """Returns a bug id if there is a database entry for 'src', None otherwise."""
        bugs = self.bugs
        found = None

        src = _processUrl(src)

        found = self._matchesHost(bugs['patterns']['host_path'], src['host'], src['path'])
        found = found or self._matchesHost(bugs['patterns']['host'], src['host'], None)
        found = found or self._matchesPath(src['path'])
        found = found or self._matchesRegex(src['host_with_path'])

        return found

    def bugId_to_app(self, bid):
        """Returns a dictionary representing the app database entry. The dictionary always contains 'name' and 'cat' (abbr. "category") and sometimes 'tags'."""
        aid = self._bug_to_app(bid)
        return self.bugs['apps'][aid]

    def _bug_to_app(self, bid):
        aid = self.bugs['bugs'][str(bid)]['aid']
        return str(aid)

    def _matchesHostPath(self, roots, src_path):
        for i in range(0, len(roots)):
            root = roots[i]
            if '$' not in root:
                continue

            paths = root['$']
            for j in range(0, len(paths)):
                if src_path.find(paths[j]['path']) == 0:
                    return paths[j]['id']

        return None

    def _matchesHost(self, root, src_host, src_path):
        node = root
        bug_id = None
        nodes_with_paths = []
        host_rev_arr = list(reversed(src_host.split('.')))
        for i in range(0, len(host_rev_arr)):
            host_part = host_rev_arr[i]

            if host_part in node:
                node = node[host_part]
                bug_id = node['$'] if '$' in node else bug_id

                if src_path is not None and '$' in node:
                    nodes_with_paths.append(node)
            else:
                if src_path is not None:
                    return self._matchesHostPath(nodes_with_paths, src_path)

                return bug_id

        if src_path is not None:
            return self._matchesHostPath(nodes_with_paths, src_path)

        return bug_id

    def _matchesRegex(self, src):
        regexes = self.bugs['patterns']['regex']

        for bug_id in regexes:
            regex = regexes[bug_id]
            if re.search(regex, src, re.IGNORECASE):
                return int(bug_id)

    def _matchesPath(self, src_path):
        paths = self.bugs['patterns']['path']

        src_path = '/' + src_path

        for path in paths:
            if src_path.find(path) >= 0:
                return paths[path]

        return None


if __name__ == "__main__":
    import sys
    import json
    with open(sys.argv[1], 'r') as f:
        bugs = json.loads(f.read())
    # some weak testing
    m = matcher(bugs)
    test1 = m.isBug('http://match.adsrvr.org/track/cmf/generic?ttd_pid=83i98y4&ttd_tpi=1')
    test2 = m.isBug('https://securepubads.g.doubleclick.net/gpt/pubads_impl_100.js')
    test3 = m.isBug('http://cdn.teads.tv/media/format.js')
    test4 = m.isBug('http://image2.pubmatic.com/AdServer/UCookieSetPug?rd=https%3a%2f%2fapi.retargetly.com%2fsync%3fpid%3d14%26sid%3d%23PM_USER_ID')

    print(m.bugId_to_app(test1))
    print(m.bugId_to_app(test2))
    print(m.bugId_to_app(test3))
    print(m.bugId_to_app(test4))
