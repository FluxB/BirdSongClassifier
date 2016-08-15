import csv
import sys, os
import urllib






f = csv.reader(open(str(sys.argv[1])))
## READ IN DATA
i =0
dataset = ()

prefix = str(sys.argv[2])

maxNumber = sys.maxint
maxSize   = sys.maxint

if len(sys.argv) > 4:
	maxNumber = int(sys.argv[4])


if len(sys.argv) > 3:
	maxSize = int(sys.argv[3])


data = ()

labelMap = dict()

for row in f:
	if i == 0:
		labels = row
		for col in range(len(labels)):
			label = labels[col].strip()
			labelMap.update({label:col})
	else:
		data = data + (row,)		
	i += 1

counter =0
print labelMap
for row in data:
    catNumber=row[labelMap["Catalogue number"]]
    if os.path.isfile(prefix+catNumber+".mp3"):
        continue
    common_name = row[labelMap["Common name"]]
    if not os.path.isdir(prefix+common_name):
        os.mkdir(prefix+common_name)
    url="http://www.xeno-canto.org/"+catNumber+"/download"
    site=urllib.urlopen(url)
    meta=site.info()
    size=int(meta.getheaders("Content-Length")[0])
    print "next file: "+url
    print "\tsize="+str(size)
    if  size <= maxSize:
        print "\tStart dowloading file " + url
        urllib.urlretrieve (url, prefix+common_name+"/"+catNumber+".mp3")
        print "\tFile has been downloaded."
        counter += 1
        if counter >= maxNumber:
            break
    else:
        print "\tFile is too large."



