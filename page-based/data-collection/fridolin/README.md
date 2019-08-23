# Dependency

PyVirtualDisplay, beautifulsoup4, coloredlogs, perhaps other (try, fail and correct ;)


## System (Ubuntu) dependencies:

Probably incomplete:

```
apt-get install python-selenium chromium-chromedriver
```

# Usage to dump images (iframes included) given a list of URLs

```python
import fridolin

urls = fridolin.load_csv("../G20-allyoucanread.csv")

# Dump images in data/
fridolin.dump_images(urls[1:2], size=(1280, 1960))

```

# Usage to get a driver

```python
import fridolin

# we get a preconfigured webdriver (chrome is fullscreen)
# you can use PyVirtualDisplay to move chrome to the virtual
# frame buffer
driver = fridolin.build_driver(use_adb=False)

# Load the main page
driver.get("http://cispa.saarland/")

driver.close()
driver.quit()
```

