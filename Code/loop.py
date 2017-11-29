"""
Python bash wrapper script that runs 'code.py' on each file in the Images directory
"""

import subprocess
import os
import sys

#remove old scores sheet if it exists
try:
    os.remove('../scores.csv')
except:
    pass

#for filename in os.listdir('../Images'):
# for i in range(1,186+1):
for i in range(1,159+1):
    try:
        print i
        bashCommand = 'python code.py %s' % str(i)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        #raw_input('Process next image...')
    except KeyboardInterrupt:
        print 'Exiting...'
        sys.exit()
