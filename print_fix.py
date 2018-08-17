

import sys, os

'''
These two functions were only used to prevent PYMRT-module from printing it's name every
time it was imported.
'''

# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__