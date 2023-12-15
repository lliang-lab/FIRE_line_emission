#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
import os
import argparse

class Setup:

    def __init__(self, sim, data_dir, snap_num, hdf5_dir):
        self.line_data = {"Gal_ID": [], "Kernel": [], "C2": [], "C2_HII": [], "C2_HI": [], "CO": [], "N2a": [], "N2b": [], "O1a": [], "O1b": [], "O3": []}
        self.ihalo_min = 0
        self.ihalo_max = 1000
        self.f_vir = 0.2
        self.MF_IDs = [0,1,2,2,10,17]
        self.sim = sim
        self.snap_num = snap_num
        self.hdf5_dir = hdf5_dir

    def extract_rvir(self, path):
        files_in_path = os.listdir(path)
        rvir = []
        for file_name in files_in_path:
            if 'halo' in file_name:
                file_path = path + file_name
                print(file_path)
                HaloCatalogue = np.loadtxt(file_path)
                rvir = HaloCatalogue[:,11]
                break
        return rvir

    def find_kernel(self, sim, data_dir, snap_num):
        if 'FB' in sim:
            path = f'{data_dir}/FIREbox/analysis/AHF/{sim}/halo/{snap_num:03d}/'
            if os.path.exists(path):
                rvir = self.extract_rvir(path)
                self.line_data["Gal_ID"] = range(self.ihalo_min, self.ihalo_max)
                self.line_data["Kernel"] = self.f_vir * np.array(rvir[self.ihalo_min:self.ihalo_max])

        elif sim == 'HR_sn1dy300ro100ss':
            for ihalo in range(self.ihalo_min, self.ihalo_max):
                path = f'{data_dir}/MassiveFIRE2/analysis/AHF/HR/h{ihalo:d}_{sim}/halo/{snap_num:3d}/'
                if os.path.exists(path):
                    self.line_data["Gal_ID"].append(ihalo)
                    self.line_data["Kernel"].append(self.f_vir * float(self.extract_rvir(path)[0]))

        elif 'MassiveFIRE' in sim:
            self.line_data["Gal_ID"] = self.MF_IDs
            paths = []
            for ihalo in self.MF_IDs[0:3]:
                paths.append(f'{data_dir}/MassiveFIRE2/analysis/AHF/HR/B762_N1024_z6_TL{ihalo:05d}_baryon_toz6_HR/halo/{snap_num:3d}/')
            for ihalo in self.MF_IDs[3:]:
                paths.append(f'{data_dir}/MassiveFIRE2/analysis/AHF/HR/B400_N512_z6_TL{ihalo:05d}_baryon_toz6/halo/{snap_num:3d}/')

            for path in paths:
                if os.path.exists(path):
                    rvir = self.extract_rvir(path)
                    self.line_data["Kernel"].append(self.f_vir * float(rvir[0]))
                else:
                    self.MF_IDs.remove(ihalo)
            self.line_data["Gal_ID"][3]=3
        else:
            raise IOError(f'Sorry. The simulation suite is not known.')

def com_to_phys(snapfile):
    attrs = snapfile['Header'].attrs
    redshift = attrs['Redshift']
    hubble = attrs['HubbleParam']
    return 1. / hubble / (1. + redshift)

def main(setup):
    hdf5_dir = setup.hdf5_dir
    snap_num = setup.snap_num
    IDs = setup.line_data["Gal_ID"]
    kernels = setup.line_data["Kernel"]
    print(IDs)
    print(kernels)

    hinv_ascale = 1
    for ihalo, kernel in zip(IDs, kernels):
        try:
            hdf5_file = f'{hdf5_dir}/snapshot_{snap_num:03d}_{ihalo:03d}_line016.hdf5'
            snapfile = h5py.File(hdf5_file,'r')

            if hinv_ascale == 1:
                hinv_ascale = com_to_phys(snapfile)

            coordinates = np.array(snapfile['/PartType0/Coordinates'])
            d = np.linalg.norm(coordinates, axis=1)

            for item in snapfile['PartType0'].keys():
                if item in setup.line_data:
                    lum_of_par = np.array(snapfile['PartType0/{}'.format(item)])
                    lum_of_gal = np.sum(lum_of_par[(d < kernel) & (lum_of_par > 0)]) # total luminosity of the galaxy
                    print(ihalo, item, lum_of_gal)
                    setup.line_data[item].append(lum_of_gal)

        except IOError:
            print("Sorry! Halo %d has not been chosen for this study." % ihalo)
            continue

    setup.line_data["Kernel"] = [(k*hinv_ascale) for k in kernels] # Correct the scaling here
    df = pd.DataFrame(setup.line_data)
    file_path = '%s_snap%03d_lines016.csv' % (sim, snap_num)
    df.to_csv(file_path, index=False)

########## main function ###############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation', type=str, default='FB15N1024', help='simulation name (default "FB15N1024")')
    parser.add_argument('-n', '--snap_num', type=int, default=104, help='snapshot number')
    args = parser.parse_args()

    sim = args.simulation
    snap_num = args.snap_num
    print(sim, snap_num)
    hdf5_dir = f'/home/lliang/Lichen/analysis/{sim}/hdf5/{snap_num:03d}'
    data_dir = f'/net/cephfs/shares/feldmann.ics.mnf.uzh/data'

    setup = Setup(sim, data_dir, snap_num, hdf5_dir)
    setup.find_kernel(sim, data_dir, snap_num)

    main(setup)
