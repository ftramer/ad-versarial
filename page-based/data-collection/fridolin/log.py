'''
Created on 14 Sep 2010

@author: gianko
'''

from logging import *
import coloredlogs

def getlogger(component, level):
    # create logger
    logger = getLogger(component)
    logger.setLevel(level)


    
    # create console handler and set level to debug
    ch = StreamHandler()
    ch.setLevel(level)
    
    # create formatter
    formatter = Formatter(fmt="[%(asctime)s] %(name)s (%(levelname)s) %(message)s"
                                  , datefmt='%d/%b/%Y:%I:%M:%S')
    
    # add formatter to ch
    ch.setFormatter(formatter)
    
    # add ch to logger
    logger.addHandler(ch)
    coloredlogs.install(level, logger=logger)

    return logger
