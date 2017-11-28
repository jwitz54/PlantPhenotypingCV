import subprocess
import os
import sys

for filename in os.listdir('../Images'):
    try:
        print filename
        bashCommand = 'python code.py %s' % filename
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        raw_input('Process next image...')
    except KeyboardInterrupt:
        print 'Exiting...'
        sys.exit()
