#!/usr/bin/env python3
#########################################################################
## The code is used for updating the hdf5 files with line luminosities ##
#########################################################################
import argparse
import sys, os
import h5py
import numpy as np
sys.path.insert(0, '/home/lliang/Lichen/rf_analysis/pfh_python_rf/')
sys.path.insert(0, '/home/lliang/Lichen/rf_analysis/pfh_python_rf/gadget_lib/')
import analysis_rf.rf_helper as rf

# physical parameters
MU = 1.25 # mean molecular weight
ProtonMass = 1.67e-24 # proton mass in unit of gram
MSUN = 2e33 # solar mass in unit of gram
KPC_TO_CM = 3.086e21 # convert 1 kilo parsec to centimeter

# other parameters
N_CLOUDY_MODELS = 41 # total number of CLOUDY simulation models
IHALO_MIN = 0
IHALO_MAX = 1000

def main(skirt_dir, hdf5_dir, hdf5_file, sim, snap_num):

    print('Reading CLOUDY template.....')
    f_hii_list, f_hi_list, c2_per_mass, c2ii_per_mass, c2i_per_mass, co_per_mass, o1a_per_mass, o1b_per_mass, o3_per_mass, n2a_per_mass, n2b_per_mass, flag = read_cloudy()
    print('Done.')

    for ihalo in range(IHALO_MIN, IHALO_MAX):

        print('Processing halo %d' % ihalo)
        try:
            # get ISRF strength and velocity dispersion of all gas particles
            g, sigma = read_g_sigma(skirt_dir, snap_num, ihalo)

            extension = '_%03d' % ihalo + '.hdf5'
            ppp, _, Lsob = rf.compute_fH2_ll(hdf5_dir, snap_num, extension, [0,0,0], 1e40)
            # sobolev length in unit of cm
            Lsob *= KPC_TO_CM
            m_gas = np.array(ppp['m'])
            # mass of gas particle in unit of M_sun
            m_gas *= 1e10
            print(np.median(m_gas))
            # gas metallicity in unit of solar metallicity (~0.02)
            if ppp['z'].ndim > 1:
                z_gas = ppp['z'][:, 0]
            else:
                z_gas = ppp['z']
            z_gas /= 0.02
            # number density of hydrogen
            n_H = m_gas * MSUN / (4./3.* np.pi * Lsob**3) / MU / ProtonMass

            # based on the physical properties (m_gas, z_gas, sigma, g, n_H), find the corresponding CLOUDY model
            a = (np.log10(m_gas + 1e-10) - 3.0) / 0.1
            m_index = np.array([min(max(0, int(x)), 40) for x in a])
            b = (np.log10(z_gas + 1e-10) + 2.2) / 0.4
            z_index = np.array([min(max(0, int(x)), 8) for x in b])
            c = (np.log10(sigma + 1e-10) + 0.2) / 0.4
            s_index = np.array([min(max(0, int(x)), 6) for x in c])
            d = (np.log10(g + 1e-10) + 2.25) / 0.5
            g_index = np.array([min(max(0, int(x)), 16) for x in d])
            e = (np.log10(n_H + 1e-10) + 2.25) / 0.5
            n_index = np.array([min(max(0, int(x)), 16) for x in e])
            index = m_index * (17 * 17 * 7 * 9) + n_index * (17 * 7 * 9) + g_index * (7 * 9) + z_index * 7 + s_index

            f_hii = f_hii_list[index]
            f_hi = f_hi_list[index]
            c2 = c2_per_mass[index] * m_gas
            c2ii = c2ii_per_mass[index] * m_gas
            c2i = c2i_per_mass[index] * m_gas
            co = co_per_mass[index] * m_gas
            o1a = o1a_per_mass[index] * m_gas
            o1b = o1b_per_mass[index] * m_gas
            o3 = o3_per_mass[index] * m_gas
            n2a = n2a_per_mass[index] * m_gas
            n2b = n2b_per_mass[index] * m_gas
            print('L_[CII] = {:e}'.format(np.sum(c2)))

            del ppp
            fname = hdf5_file + '_%03d' % ihalo + '.hdf5'
            OldSnapfile = h5py.File(fname,'r')
            print(f'{fname} done.')
            fname = hdf5_file + '_%03d' % ihalo + '_line.hdf5'
            NewSnapfile = h5py.File(fname,'w')

            #copy oldfashioned header:
            OldSnapfile.copy('/Header', NewSnapfile, '/Header')
            OldSnapfile.copy('/PartType4', NewSnapfile, '/PartType4')
            OldSnapfile.copy('/PartType1', NewSnapfile, '/PartType1')
            OldSnapfile.copy('/PartType0', NewSnapfile, '/PartType0')
            NewSnapfile['/PartType0/HII_frac'] = f_hii
            NewSnapfile['/PartType0/HI_frac'] = f_hi
            NewSnapfile['/PartType0/C2'] = c2
            NewSnapfile['/PartType0/C2_HII'] = c2ii
            NewSnapfile['/PartType0/C2_HI'] = c2i
            NewSnapfile['/PartType0/CO'] = co
            NewSnapfile['/PartType0/O1a'] = o1a
            NewSnapfile['/PartType0/O1b'] = o1b
            NewSnapfile['/PartType0/O3'] = o3
            NewSnapfile['/PartType0/N2a'] = n2a
            NewSnapfile['/PartType0/N2b'] = n2b

            print('L_[CO] = {:e}'.format(np.sum(co[co>0])))
        except IOError:
            print('Halo %d - No halo found!' % ihalo)
            continue

def read_cloudy():
    """read in the cloudy templates"""

    f_hii_list = []
    f_hi_list = []
    c2_per_mass = []
    c2ii_per_mass = []
    c2i_per_mass = []
    co_per_mass = []
    o1a_per_mass = []
    o1b_per_mass = []
    o3_per_mass = []
    n2a_per_mass = []
    n2b_per_mass = []
    flag = []

    for imodel in range(N_CLOUDY_MODELS):

        fname = cloudy_dir + 'cloudy_z0_em_%d.txt' % imodel
        try:
            with open(fname) as fp:
                print(fname)

                line = fp.readline().split()
                count = 0
                while line:
                    f_hii_list.append(float(line[5]))
                    f_hi_list.append(float(line[6]))
                    c2_per_mass.append(float(line[8]))
                    c2ii_per_mass.append(float(line[9]))
                    c2i_per_mass.append(float(line[10]))
                    co_per_mass.append(float(line[13]))
                    o1a_per_mass.append(float(line[19]))
                    o1b_per_mass.append(float(line[20]))
                    o3_per_mass.append(float(line[21]))
                    n2a_per_mass.append(float(line[22]))
                    n2b_per_mass.append(float(line[23]))
                    flag.append(int(line[25]))
                    line = fp.readline().split()
                    count += 1

                print('id %d - total number of models in this file: %d' % (imodel, count))

        except IOError:
            print('!!!!!!!!!!!!!!!!!!! CLOUDY file (id=%d) is not found. !!!!!!!!!!!!!!!!!!!' % imodel)
            sys.exit(1)

    f_hii_list = np.array(f_hii_list)
    f_hi_list = np.array(f_hi_list)
    c2_per_mass = np.array(c2_per_mass)
    c2ii_per_mass = np.array(c2ii_per_mass)
    c2i_per_mass = np.array(c2i_per_mass)
    co_per_mass = np.array(co_per_mass)
    o1a_per_mass = np.array(o1a_per_mass)
    o1b_per_mass = np.array(o1b_per_mass)
    o3_per_mass = np.array(o3_per_mass)
    n2a_per_mass = np.array(n2a_per_mass)
    n2b_per_mass = np.array(n2b_per_mass)

    return f_hii_list, f_hi_list, c2_per_mass, c2ii_per_mass, c2i_per_mass, co_per_mass, o1a_per_mass, o1b_per_mass, o3_per_mass, n2a_per_mass, n2b_per_mass, flag

def read_g_sigma(skirt_dir, snap_num, ihalo):
    """
        to read ISRF and velocity dispersion files
    """
    file_path = skirt_dir + 'snapshot_%03d/' % snap_num + '%05d/' % ihalo
    fname_list = os.listdir(file_path)
    disp_file = ''
    ISRF_file = ''

    for fname in fname_list:
        # file containing velocity dispersion
        if '_sigma.txt' in fname:
            disp_file = file_path + fname
        # file containing local radiative intensity
        if '_ISRF.txt' in fname:
            ISRF_file = file_path + fname

    # to get local velocity dispersion from disp_file
    with open(disp_file) as fp:
        sigma = []
        line = fp.readline().split()
        count1 = 0
        while line:
            sigma.append(float(line[1]))
            line = fp.readline().split()
            count1 += 1

    # to get local ISRF strength from ISRF_file
    with open(ISRF_file) as fp:
        g = []
        line = fp.readline().split()
        count2 = 0
        while line:
            g.append(float(line[2]))
            line = fp.readline().split()
            count2 += 1

    # check if the two files show the same number of gas particles
    if count1 != count2:
        print('Sorry! Either the ISRF or the dispersion file is corrupted.')
        sys.exit(1)

    print('%d gas particles have been found.' % count1)

    sigma = np.array(sigma)
    g = np.array(g)

    return g, sigma

########## main function ###############
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation', type=str, default='FB15N1024', help='simulation name (default "FB15N1024")')
    parser.add_argument('-n', '--snap_num', type=int, default=832, help='snapshot number')
    args = parser.parse_args()

    sim = args.simulation
    snap_num = args.snap_num
    print(sim, snap_num)

    # directories
    cloudy_dir = '/home/lliang/Lichen/cloudy_v2/' # the directory to the CLOUDY output data
    skirt_dir = '/home/lliang/FIREBox/'
    analysis_dir = '/home/lliang/Lichen/analysis/'

    hdf5_dir = os.path.join(analysis_dir, f'{sim}/hdf5/{snap_num:03d}/')
    hdf5_file = os.path.join(hdf5_dir, f'snapshot_{snap_num:03d}')

    main(skirt_dir, hdf5_dir, hdf5_file, sim, snap_num)
