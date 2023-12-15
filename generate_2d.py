#!/usr/bin/env python3

###############################################
##  This code is used for producing 2D image ##
##   of specific line emission of galaxies   ##
###############################################

import h5py
import numpy as np
import math
import csv
import argparse
import convolution as conv

# halo id
id_min = 20
id_max = 21

img_size = 60.
npixel = 1024
interval = img_size / npixel

def main(sim, snapnum):

    hdf5_dir = '/home/lliang/Lichen/analysis/%s/hdf5/%03d/' % (sim, snapnum)

    for ihalo in range(id_min, id_max):
        hdf5_name = hdf5_dir + sim + '_%03d' % snapnum + '_%03d' % ihalo + '_line.hdf5'
        generate_2d(hdf5_name)

def generate_2d(hdf5_name):

    snapfile = h5py.File(hdf5_name,'r')

    #redshift and hubble parameter
    attrs = snapfile['Header'].attrs
    redshift = attrs['Redshift']
    ascale = 1. / (1. + redshift)
    hubble = attrs['HubbleParam']
    hinv = 1. / hubble

    x = snapfile['/PartType0/Coordinates'][:,0] * (hinv * ascale)
    y = snapfile['/PartType0/Coordinates'][:,1] * (hinv * ascale)
    z = snapfile['/PartType0/Coordinates'][:,2] * (hinv * ascale)
    h = snapfile['/PartType0/SmoothingLength'][:] * (hinv * ascale)
    l = snapfile['/PartType0/C2'][:]
    Ngas = snapfile['PartType0']['Masses'].shape[0]
    print('Ngas=', Ngas)

    output = np.zeros([npixel, npixel])
    lcii = 0.

    for iparticle in range(Ngas):

        x_index = int((x[iparticle] + 30.) / interval)
        y_index = int((y[iparticle] + 30.) / interval)
        z_index = int((z[iparticle] + 30.) / interval)
        kernel_size = int(h[iparticle] / interval)
        #print(kernel_size)

        if (x_index < 0) or (x_index >= npixel) or (y_index < 0) or (y_index >= npixel) or (z_index < 0) or (z_index >= npixel):
            continue

        if iparticle % 100 == 0:
            print(iparticle, l[iparticle], lcii, np.sum(output))

        lcii += l[iparticle]

        if kernel_size == 0:
             output[x_index, y_index] += l[iparticle]
        else:
            area = (2 * kernel_size + 1) ** 2
            for k in range(max(0, x_index - kernel_size), min(npixel, x_index + kernel_size + 1)):
                for j in range(max(0, y_index - kernel_size), min(npixel, y_index + kernel_size + 1)):
                    output[k, j] += l[iparticle] / (1. * area)

    print(lcii, np.sum(output))

    with open('./cii_2d_test.csv','w+') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(output)

        #wr = csv.writer(myfile) #, quoting=csv.QUOTE_ALL)
        #wr.writerow(output)

########## main function ###############

#import __main__ as main
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--snapnum', type=int, help='snapshot number')
parser.add_argument('-s', '--sim', type=str, help='FB15N1024 (default)')
args = parser.parse_args()

snapnum = args.snapnum
sim = args.sim
print(sim, snapnum)

main(sim, snapnum)
