# Usage: "bird.py path1 path2" where path1 is a csv file with all the
# species (specy name, country, number of files) that should be downloaded
# from Xeno-Canto (list.csv) and path2 is the folder in which the species
# sub-folders are placed.

import csv
import sys
import os
import urllib.request


path_list = str(sys.argv[1])
prefix = str(sys.argv[2])

# Read in species

species_list = csv.reader(open(path_list))

species = ()
for row in species_list:
    species = species + (row,)
    print(species)

for row in species:

    birdname = row[0].replace(" ", "+")
    country = row[1]
    maxNumber = int(row[2])

    url_csv = "http://www.xeno-canto.org/explore/csv?query=" + \
        birdname + "+cnt%3A%22" + country + "%22"
    path_csv = prefix + "csv/" + birdname + ".csv"

    if not os.path.isdir(prefix + "csv/"):
        os.mkdir(prefix + "csv/")

    print("\tStart dowloading file " + url_csv)
    path_csv, header = urllib.request.urlretrieve(url_csv)

    f = csv.reader(open(path_csv))

    # READ IN DATA
    i = 0
    dataset = ()

    data = ()

    labelMap = dict()

    for row in f:
        if i == 0:
            labels = row
            for col in range(len(labels)):
                label = labels[col].strip()
                labelMap.update({label: col})
        else:
            data = data + (row,)
        i += 1

    counter = 0
    print(labelMap)
    for row in data:
        catNumber = row[labelMap["Catalogue number"]]
        if os.path.isfile(prefix + catNumber + ".mp3"):
            continue
        common_name = row[labelMap["Common name"]]
        if not os.path.isdir(prefix + common_name):
            os.mkdir(prefix + common_name)
        url = "http://www.xeno-canto.org/" + catNumber + "/download"
        site = urllib.request.urlopen(url)
        meta = site.info()
        size = int(meta["Content-Length"])
        print("next file: " + url)
        print("\tsize=" + str(size))
        print("\tStart dowloading file " + url)
        urllib.request.urlretrieve(url, prefix + common_name +
                                   "/" + catNumber + ".mp3")
        print("\tFile has been downloaded.")
        counter += 1
        if counter >= maxNumber:
            break
