from lxml import etree
from StringIO import StringIO
from collections import deque
import log
logger = log.getlogger("rendertree", log.DEBUG)

class RenderTree(object):
    def __init__(self, roots=[]):
        self.roots = roots

class RenderTreeNode(object):
    
    def __init__(self, info="", xml=None, children=[], parent=None):
        self.info = info
        self.xml = xml
        self.children = children
        self.parent = parent

    def __str__(self):
        return "info={}, xml={}".format(self.info, str(self.xml))

def parse_xmltree(xml):
    try:
        t = etree.parse(StringIO(xml))
    
    except Exception as e:
        logger.exception(e)
        logger.info("XML Parser failed parsing. Reparsing with recover=True")
        parser = etree.XMLParser(recover=True)
        t = etree.parse(StringIO(xml), parser)   

    return t

def merge_into_xhtmltree(main_page, iframes_content):
    t = parse_xmltree(main_page)

    
    for k, v in iframes_content.iteritems():
        if "/head" in k:
            continue

        el = t.xpath(k, namespaces={'xhtml':'http://www.w3.org/1999/xhtml'})
        
        if len(el) != 1: # if that happens, no match or xpath is non ambiguous
            logger.warning("{} iframes were found with {}".format(len(el), k))

        if len(el) > 0:
            subt_root = el[0]

            subt = parse_xmltree(v)
            subt_root.insert(0, subt.getroot())
        else:
            logger.warning("No iframes found.")

    return t



def _recurs_tree_creation(xml_n, rend_parent_n):
    rend_n = RenderTreeNode()

    rend_n.parent = rend_parent_n
    rend_n.xml = xml_n

    if isinstance(xml_n, etree._Comment):
        rend_n.info = "comment"
    else:
        rend_n.info = xml_n.tag

    rend_n.children = []
    for xml_c in xml_n.getchildren():
        rend_n.children.append(_recurs_tree_creation(xml_c, rend_n))

    return rend_n

def build_rendertree(xhtml):
    rend_tree = RenderTree()

    xml_root = xhtml.getroot()

    rend_root = _recurs_tree_creation(xml_root, rend_tree)

    rend_tree.roots = [rend_root]

    return rend_tree

_IGNORE_TAGS = ["html", "head", "script", "meta", "link", "base", "style", "title"]

def _remove_namespace(tag):
    NSes = ["{http://www.w3.org/1999/xhtml}", "{http://www.w3.org/2000/svg}"]
    for ns in NSes:
        if ns in tag:
            return tag[len(ns):]
    return tag

def remove_invisible_nodes(rend_tree):
    """
    We visit the render tree to delete all XML elements that 
    are not visible. If the node has children, they are inserted in 
    """
    Q = deque(rend_tree.roots) # initialize the queue with the roots (usually only one root)
    cnt = 0
    while len(Q) > 0:
        #print "QUEUE: ", ",".join([_remove_namespace(el.info) for el in Q])
        n = Q.popleft()

        if _remove_namespace(n.info) in _IGNORE_TAGS or n.info == "comment":
            #print " !!!!! DELETE !!!!! ", _remove_namespace(n.info)
            """
            - we visit n's children and update parent
            """
            for c in n.children:
                c.parent = n.parent

            """
            - then we visit n.parent and:
               1) delete n from children, and 
               2) insert at the position of n 
                  all the children of n
            """
            if isinstance(n.parent, RenderTree): # n is a root
                child_pos = n.parent.roots.index(n)
                n.parent.roots = n.parent.roots[0:child_pos] + n.children + n.parent.roots[child_pos+1:]

            else: # n is an inner node
                child_pos = n.parent.children.index(n)
                n.parent.children = n.parent.children[0:child_pos] + n.children + n.parent.children[child_pos+1:]

            cnt += 1

        Q.extend(n.children)

    print "Deleted {} nodes".format(cnt)
    return rend_tree

def save_xml_to_file(xml, file):
    import codecs
    with codecs.open(file, "w", encoding="utf-8") as f:
        f.write(etree.tostring(xml, pretty_print=True))

def load_xml_from_file(file):
    t = None
    with open(file, "r") as f:
        t = etree.parse(f)

    return t