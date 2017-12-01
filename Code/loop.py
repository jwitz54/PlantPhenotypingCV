"""
Python bash wrapper script that runs 'code.py' on each file in the Images directory
"""

import subprocess
import os
import sys
import shutil

def main():
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

    dst = '../scores' + sys.argv[1] + '.csv'
    shutil.copyfile('../scores.csv',dst)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a run id number")
        sys.exit()
    try:
        int(sys.argv[1])
    except:
        print("Specify an integer argument")
        sys.exit
    fname = '../scores' + sys.argv[1] + '.csv'
    if os.path.isfile(fname):
        print("File already exists")
        sys.exit()

    main()