import sys
from selenium import webdriver
import selenium.common.exceptions as selexcept
import time
from pyvirtualdisplay import Display
import os
import os.path
from urlparse import urlparse
import json
from datetime import datetime
import codecs
from PIL import Image, ImageDraw
import bs4
import timeout_decorator as tout_dec

import rendertree as rtree
from geometry import Rectangle
import log
logger = log.getlogger("fridolin", log.INFO)

MAX_TIMEOUT = 60


class MaximumRetryReached(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

def _get_xpath_selenium(driver, element):
    xpath = driver.execute_script("""gPt=function(c){
                                 if(c===document){
                                    return ""
                                 }
                                 if(c.id!==''){
                                     return '//*[@id="'+c.id+'"]'
                                 }
                                 var a=0;
                                 var e=c.parentNode.childNodes;
                                 for(var b=0;b<e.length;b++){
                                     var d=e[b];
                                     if(d===c){
                                         return gPt(c.parentNode)+'/xhtml:'+ c.tagName.toLowerCase()+'['+(a+1)+']'
                                     }
                                     if(d.nodeType===1&&d.tagName===c.tagName){
                                         a++
                                     }
                                 }
                             };
                             return gPt(arguments[0]);""", element)
    return xpath


def build_driver(use_adb=False, proxy=None, lang=None, timeout=MAX_TIMEOUT, fullscreen=True):
    opts = webdriver.ChromeOptions()

    #opts.add_argument("load-extension=../misc/jsinj/")

    if use_adb:
        opts.add_argument("load-extension=../misc/adblocker/unpacked/3.21.0_0/")

    if proxy:
        opts.add_argument("proxy-server={}".format(proxy))

    #opts.add_argument("start-maximized")
    if fullscreen:
        opts.add_argument("start-fullscreen")

    opts.add_argument("disable-infobars") # we remove the ugly yellow notification "Chrome is being controlled by automated test software"
    #opts.add_argument("disable-web-security") # We need this to disable SOP and access iframe content

    if lang:
        #print lang
        opts.add_argument("lang={}".format(lang))

    driver = webdriver.Chrome(executable_path="/usr/lib/chromium-browser/chromedriver", chrome_options=opts)

    if timeout:
        driver.set_page_load_timeout(timeout)

    return driver

def build_virtdisplay(size):
    vdisp = Display(visible=0, size=size)
    return vdisp

def _serialize_dom(driver):
    """

    To dump the DOM tree we do not use HTML. The problem is that the extracted
    HTML from Chrome/Firefox will produce unwanted parse trees bu libxml2-based
    HTML parsers (lxml, html5lib, ...). The problem is that libxml2 tries to
    create valid HTML parse tree and, for example, SCRIPT tags in the HEAD tag are
    relocated into the BODY section. Chrome/Firefox accept SCRIPT tags in the HEAD.

    This means that XPaths extracted from Chrome/Firefox are no longer valid on
    libxml2 parse trees.

    Our workaround is to serialize the DOM tree in valid XML. In this way, libxml2
    will not perform corrective actions like dropping nodes. Instead, it will keep
    the entire document.

    """
    return driver.execute_script("var __s = new XMLSerializer(); return __s.serializeToString(document)")

def get_iframe_display_data_webdriver(driver, max_tries=3):
    def _do_it():
        webelements = driver.find_elements_by_tag_name("iframe")
        displ_data = [(i, _.get_attribute("outerHTML"), _.is_displayed(), _.location, _.size) for i, _ in enumerate(webelements)]
        return displ_data

    for i in range(max_tries):
        try:
            logger.info("Retrieving IFRAMEs... (attempt {}/{})".format(i+1, max_tries))
            return _do_it()
        except Exception as e:
            logger.exception(e)

    raise MaximumRetryReached()

def get_iframe_display_data_injJS(driver, max_tries=3):
    displ_data = driver.execute_script("""
iframe = document.getElementsByTagName("iframe");
data = [];
for (i = 0; i<iframe.length; i++){
    f = iframe[i];
    box = f.getBoundingClientRect();
    data.push([i, f.outerHTML, f.style.display, [box["x"], box["y"]], [box["width"], box["height"]]]);
};
return data;
""")
    return displ_data;

def process_screnshot(folder, exclude=[]):
    # Open the main page to crop iframes and build the bitmask
    main = Image.open(os.path.join(folder, "main.png"))

    # Deserialize JSON
    with open(os.path.join(folder, "iframe_display_data.json"), "r") as f:
        displ_data = json.load(f)

    # Init the bitmask
    bitmask = Image.new('1', main.size, color=1)
    draw = ImageDraw.Draw(bitmask)

    # For each iframe, crop it from the main page
    for e in displ_data:
        i, html, is_displayed, loc, size = e


        # points of the iframe
        iframe_r = Rectangle(int(loc[0]), int(loc[1]), int(loc[0])+int(size[0]), int(loc[1])+int(size[1]))

        # points of the screenshot
        image_r = Rectangle(*main.getbbox())

        # intersect image and iframe
        visible_iframe_r = image_r.intersection(iframe_r)

        logger.info("{}: Image:{}, Iframe={}, Inters={}".format(i, image_r, iframe_r, visible_iframe_r))

        # if intersection has no size
        if visible_iframe_r is None or visible_iframe_r.width() == 0 or visible_iframe_r.height() == 0:
            continue

        # We extract the domain of the src attribute and use it in the filename
        p_html = bs4.BeautifulSoup(html, "lxml")
        src = p_html.iframe.get("src", "None")
        domain = src
        if len(domain) > 0:
            domain = urlparse(src).netloc

        if domain in exclude:
            continue

        # Crop the iframe and store
        if_img = main.crop(list(visible_iframe_r))
        if_img.save(os.path.join(folder, "{}-{}.png".format(i, domain)))

        # Draw black rectangle in the bitmask
        draw.rectangle(list(visible_iframe_r), fill=0)

    del draw
    bitmask.save(os.path.join(folder, "bitmask.png"))



@tout_dec.timeout(MAX_TIMEOUT, use_signals=False, timeout_exception=selexcept.TimeoutException)
def save_screenshot_and_iframe_metadata(driver, domain=""):
    t = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    folder = os.path.join("data", domain, t)

    if not os.path.exists(folder):
        os.makedirs(folder)

    displ_data = get_iframe_display_data_injJS(driver)

    # Save main page
    driver.save_screenshot(os.path.join(folder, "main.png"))

    # Dump IFRAME data
    ser_e = json.dumps(displ_data, indent=4)
    with codecs.open(os.path.join(folder, "iframe_display_data.json"), "w", encoding="utf-8") as f:
        f.write(ser_e)

    return folder

def dump_images_iframes_handler(driver, url):
    logger.info("Making screenshot and taking metadata...")
    folder = save_screenshot_and_iframe_metadata(driver, domain=urlparse(url).netloc)

    logger.info("Generating bitmask and cropping ads...")
    process_screnshot(folder)


def get_visible_bounding_boxes_injJS(driver, max_tries=3):
    displ_data = driver.execute_script("""
function isInViewport(element) {
  var rect = element.getBoundingClientRect();
  var html = document.documentElement;
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || html.clientHeight) &&
    rect.right <= (window.innerWidth || html.clientWidth)
  );
}

elems = document.body.getElementsByTagName("*");
data = [];
for (i = 0; i < elems.length; i++) {
    el = elems[i];
    box = el.getBoundingClientRect();
    if (el.style.display != "none" && isInViewport(elems[i]) && box.width > 0 && box.height > 0) {
      data.push([i, el.outerHTML, el.style.display, [box.x, box.y], [box.width, box.height]]);
    }
};
return data;
""")
    return displ_data;

@tout_dec.timeout(MAX_TIMEOUT, use_signals=False, timeout_exception=selexcept.TimeoutException)
def save_screenshot(driver, domain=""):
    t = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    folder = os.path.join("data", domain, t)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save main page
    driver.save_screenshot(os.path.join(folder, "main.png"))

    return folder

@tout_dec.timeout(MAX_TIMEOUT, use_signals=False, timeout_exception=selexcept.TimeoutException)
def save_screenshot_and_tag_metadata(driver, domain=""):
    t = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    folder = os.path.join("data", domain, t)

    if not os.path.exists(folder):
        os.makedirs(folder)

    displ_data = get_visible_bounding_boxes_injJS(driver)

    # Save main page
    driver.save_screenshot(os.path.join(folder, "main.png"))

    # Dump Visible TAGs data
    ser_e = json.dumps(displ_data, indent=4)
    with codecs.open(os.path.join(folder, "visible_tags_data.json"), "w", encoding="utf-8") as f:
        f.write(ser_e)

    return folder

def crop_visible_tags(folder):
    # Open the main page to crop tags
    main = Image.open(os.path.join(folder, "main.png"))

    # Deserialize JSON
    with open(os.path.join(folder, "visible_tags_data.json"), "r") as f:
        displ_data = json.load(f)

    # For each iframe, crop it from the main page
    for e in displ_data:
        i, html, is_displayed, loc, size = e

        # bounding box of visible element
        box = [int(loc[0]), int(loc[1]), int(loc[0])+int(size[0]), int(loc[1])+int(size[1])]


        exclude_tags = ["time", "p", "b", "i", "em", "span", "li", "ul", "ol", "h", "a"]
        if any(html[1:6].lower().startswith(el) for el in exclude_tags):
            logger.info("Element '{}...'' in exclude_tag (skipping)".format(html[:6].encode("ascii", "ignore")))
            continue

        # skip if too small
        if (int(size[0]) < 24 or int(size[1]) < 24):
            logger.info("Element '{}...'' too small {}x{}px (skipping)".format(html[:10].encode("ascii", "ignore"), int(size[0]), int(size[1])))
            continue


        # Crop the tag and store
        crop = main.crop(box)
        crop.save(os.path.join(folder, "{}.png".format(i)))



def dump_images_visible_tags_handler(driver, url):
    logger.info("Making screenshot and taking metadata...")
    folder = save_screenshot_and_tag_metadata(driver, domain=urlparse(url).netloc)

    logger.info("Cropping visible TAGs...")

    crop_visible_tags(folder)

def screenshot_handler(driver, url):
    logger.info("Making screenshota...")
    save_screenshot(driver, domain=urlparse(url).netloc)


def visit_url(url, handler=None, size=(1920, 9600), scroll=False, shake_size=False, sleep=8, timeout=MAX_TIMEOUT, use_adb=False, proxy=None):

    if isinstance(url, basestring):
        url = [url]

    with build_virtdisplay(size=size) as display:

        driver = build_driver(use_adb=use_adb, proxy=proxy, timeout=timeout)
        driver.set_window_size(size[0], size[1])

        for u in url:
            logger.info("Visiting {}".format(u))

            try:
                driver.get(u)

                if scroll:
                    # As a result of the EU law on cookies, web sites shows a banner
                    # for the use of cookies. They won't load Ads before the user accepts.
                    # To accept, the user can scroll down or click around.
                    time.sleep(4) # let's give some time to register handlers
                    driver.execute_script("window.scrollBy(0, 100);")
                    time.sleep(2) # let's give it some time also here

                    # By now, a cookie storing the choice of the user should be already set.
                    # We reload.
                    driver.get(u)

            except selexcept.TimeoutException as e:
                logger.info("Loading page timeout. Continue...")
            except Exception as e:
                logger.exception(e)
                continue


            if shake_size:
                # We resize back and forth. I observed this creates better screenshots (more HTML
                # are drawn)
                driver.set_window_size(size[0]-100, size[1]-100)
                driver.set_window_size(size[0], size[1])

            time.sleep(sleep)

            try:
                if handler:
                    handler(driver, u)

            except selexcept.TimeoutException as e:
                logger.info("Timeout when making screenshots.")
                logger.exception(e)
                continue
            except Exception as e:
                logger.exception(e)
                continue
            finally:
                # We do have a problem and the current driver may not be reused.
                try:
                    driver.close()
                    driver.quit()
                except:
                    pass
                # reinstantiate the driver
                driver = build_driver(use_adb=use_adb, proxy=proxy, timeout=timeout)
                driver.set_window_size(size[0], size[1])

        driver.close()
        driver.quit()

def _task_builder(**kwargs):
    def _(url):
        return visit_url(url, **kwargs)

    return _

def load_alexa(file):
    return load_csv(file, col=1, sep=",")

def load_csv(file, col=0, sep=","):
    domains = []
    with open(file, "r") as f:
        domains = f.readlines()

    domains = map(lambda e: e.split(sep)[col].strip(), domains)
    domains = map(lambda e: e if e.startswith("http://") or e.startswith("https://") else "http://{}".format(e), domains)
    domains = map(lambda e: e.replace("\"", ""), domains)
    return domains




def get_html_documents(driver, max_tries=3):
    def _do_get_docs():
        main_html = _serialize_dom(driver)

        webelements = driver.find_elements_by_tag_name("iframe")
        logger.info("Found {} iframes".format(len(webelements)))

        iframes_content = {}
        for i, webel in enumerate(webelements):
            # XPath position of webel is the index in our map
            xpath = _get_xpath_selenium(driver, webel)
            # The HTML inside the iframe is the value of our map
            driver.switch_to.frame(webel)
            iframe_html = _serialize_dom(driver)

            logger.debug("iframe {} xpath={}, web element={}".format(i, xpath, webel))

            # Store XPath + HTML in our map
            iframes_content.setdefault(xpath, iframe_html)

            # Switch bak to the main frame
            driver.switch_to_default_content()

        return main_html, iframes_content

    for i in range(max_tries):
        try:
            logger.info("Extracting HTML documents... (attempt {}/{})".format(i+1, max_tries))
            return _do_get_docs()
        except Exception as e:
            logger.exception(e)

    raise MaximumRetryReached()




def fetch_rendertree(url, **kv):
    driver = build_driver(**kv)
    xhtml_tree = None
    rend_tree = None
    try:
        driver.get(url)
        time.sleep(10)

        main_page, iframes_content = get_html_documents(driver)

        time.sleep(1)

        xhtml_tree = rtree.merge_into_xhtmltree(main_page, iframes_content)
        rend_tree  = rtree.build_rendertree(xhtml_tree)
        rend_tree  = rtree.remove_invisible_nodes(rend_tree)

    except:
        logger.exception(sys.exc_info()[0])
    finally:
        driver.close()

    return xhtml_tree, rend_tree


if __name__ == "__main__":

    url = sys.argv[1]
    proxy = sys.argv[2]

    visit_url(url, handler=screenshot_handler, size=(1920, 1080), proxy=proxy)
