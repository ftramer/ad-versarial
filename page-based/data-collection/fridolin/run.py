import fridolin
import sys

print "test"
print sys.argv[1]
urls = fridolin.load_csv(sys.argv[1])
print urls
fridolin.visit_url(urls, size=(1920, 1080), sleep=15, handler=fridolin.screenshot_handler)
