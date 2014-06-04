import subprocess
import sys
import time
import os


ILLUSTRATOR_BIN = r"/Applications/Adobe Illustrator CC/Adobe Illustrator.app/Contents/MacOS/Adobe Illustrator" 
SCRIPT = "svg2ai.jsx" 

fcount = len(sys.argv[1:])/2
files = []
for i in range(fcount):
    files.append((sys.argv[2*i+1], sys.argv[2*i+2]))


fid = file("filenames.txt", 'w')
for f in files:
    fid.write("%s %s\n" % f)
fid.close()

starttime = time.time() 

sp = subprocess.Popen([ILLUSTRATOR_BIN, SCRIPT])
print "sleeping on execution"

time.sleep(10)
# quitting -- all this because app.quit segfaults 
waiting  = True
while waiting:
    waiting  = False

    for f_in, f_out in files:
        if os.path.exists(f_out):

            ftime = os.path.getmtime(f_out)
            if ftime < starttime:
                waiting = True
        else:
            waiting = True

    time.sleep(2)

sp.terminate()


