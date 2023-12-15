#!/usr/bin/env python3
#################################################
#####  The code is used for calculating the #####
#####  properties of the FIREBox galaxies   #####
#################################################

import h5py
import numpy as np
import csv
import argparse

ihalo_max = 1000
f_vir = 0.1

def get_mass(sim, snapnum, k):
    """to compute the total line luminosity for the FIRE galaxies"""
    hdf5_dir = '/home/lliang/Lichen/analysis/%s/hdf5/%03d/' % (sim, snapnum)

    output_file = sim + '_snap%03d' % snapnum + '_masses.csv'
    fout = open(output_file, 'w')
    writer = csv.writer(fout)
    header = ['Gal ID', 'M_H2 (M_sun)', 'M_HI (M_sun)', 'M_HII (M_sun)']
    writer.writerow(header)

    for ihalo in range(ihalo_max):
        try:
            hdf5_file = hdf5_dir + sim + '_%03d' % snapnum + '_%03d' % ihalo + '_line.hdf5'
            # read in the hdf5 snapshot
            snapfile = h5py.File(hdf5_file,'r')

            #redshift and hubble parameter
            attrs = snapfile['Header'].attrs
            redshift = attrs['Redshift']
            ascale = 1. / (1. + redshift)
            hubble = attrs['HubbleParam']
            hinv = 1. / hubble

            x = np.array(snapfile['/PartType0/Coordinates'][:,0])
            x *= hinv * ascale
            y = np.array(snapfile['/PartType0/Coordinates'][:,1])
            y *= hinv * ascale
            z = np.array(snapfile['/PartType0/Coordinates'][:,2])
            z *= hinv * ascale

            r = (x*x + y*y + z*z)**0.5

            m_gas = np.array(snapfile['PartType0/Masses'][:])
            m_gas *= 1e10 / hubble # in unit of M_sun
            f_hii = np.array(snapfile['PartType0/HII_frac'][:])
            f_h1 = np.array(snapfile['PartType0/HI_frac'][:])
            f_h2 = 1. - f_hii - f_h1

            m_hii = m_gas * f_hii
            m_hi = m_gas * f_h1
            m_h2 = m_gas * f_h2

            if k == -1:
                rvir = find_kernel(sim, snapnum, ihalo, redshift) # in unit of ckpc / h
                kernel = f_vir * rvir
                kernel *= hinv * ascale
            elif k > 0:
                kernel = k

            M_HII = np.sum(m_hii[r < kernel])
            M_HI = np.sum(m_hi[r < kernel])
            M_H2 = np.sum(m_h2[r < kernel])

            print('%d %.2f %.2f %.2f\n' % (ihalo, np.log10(M_HII + 1e-20), np.log10(M_HI + 1e-20), np.log10(M_H2 + 1e-20)))
            row = [ihalo, np.log10(M_HII + 1e-20), np.log10(M_HI + 1e-20), np.log10(M_H2 + 1e-20)]
            writer.writerow(row)

        except IOError:
            print("Sorry! Halo %d has not been chosen for this study." % ihalo)
            continue

def find_kernel(sim, snapnum, ihalo, redshift):
    """to return the kernel size """
    z_snap="%s.000" % (str(redshift)[0])
    print(z_snap, float(redshift))

    halo_dir = '/net/cephfs/shares/feldmann.ics.mnf.uzh/data/FIREbox/analysis/AHF/%s/halo/%03d/' % (sim, snapnum)
    halo_file = halo_dir + "%s.z%s.AHF_halos"% (sim, z_snap)
    print('Reading halo catalogue', halo_file)

    #check if file exists
    try:
        HaloCatalogue = np.loadtxt(halo_file)
    except IOError:
        print('!!!!!!!!!!!!!! file is unreadable - permission problem? !!!!!!!!!!!!!!!')
        return None

    rvir = HaloCatalogue[ihalo,11]
    print('Rvir = %.5g ckpc/h' % float(rvir))

    return rvir

########## main function ###############
#import __main__ as main
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--snapnum', type=int, help='snapshot number')
parser.add_argument('-s', '--sim', type=str, help='FB15N1024 (default)')
parser.add_argument('-k', '--kernel', type=int, help='Specify kernel size. 1) k>0 , fixed kernel size (in unit of kpc) 2) k=-1, kernel is scaled to virial radius.', default=30)
args = parser.parse_args()

snapnum = args.snapnum
sim = args.sim
k = args.kernel
print(sim, snapnum)

get_mass(sim, snapnum, k)
