"""
Python bash wrapper script that runs 'code.py' on each file in the Images directory

Command line usage: "python loop.py id"
                    where id is an integer ID used to write a unique .csv 

Developed for ECE 5470 Fall 2017 by Jeff Witz, Curran Sinha, and Cameron Schultz
"""

import subprocess
import os
import sys
import shutil

def main():
    #r emove old scores sheet if it exists
    try:
        os.remove('../scores.csv')
    except:
        pass
    
    # run code.py on each image pair
    for i in range(1,159+1):
        try:
            print 'Processing image ', i
            bashCommand = 'python code.py %s' % str(i)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        except KeyboardInterrupt:
            print 'Exiting...'
            break
    
    # copy temp csv to permanent unique csv
    dst = '../scores' + sys.argv[1] + '.csv'
    shutil.copyfile('../scores.csv',dst)
    sys.exit()

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