 #!/usr/bin/env python3
import numpy as np
from scipy import integrate
import argparse
import os
import multiprocessing
from multiprocessing import Pool

# Constants (in uppercase)
PLANCK = 6.63e-34 # Planck constant in SI unit
SPEED_OF_LIGHT = 3e8 # Speed of light in meters per second
LAMBDA_MIN = 9.116e-8 # Minimum wavelength in micrometers (=13.6 eV)
LAMBDA_MAX = 2.0664e-7 # Maximum wavelength in micrometers (=6 eV)
G_0 = 1.6e-6 # ISRF strength in local solar neighborhood (Watt/meter^2)

class ISRFProcessor:
    def __init__(self, sed_file, cell_file, J_file, gas_file):
        self._lambda = self.get_lambda(sed_file)
        self._lambda_Habing = self._lambda[(self._lambda > LAMBDA_MIN) & (self._lambda < LAMBDA_MAX)]
        self._lambda_ion = self._lambda[self._lambda < LAMBDA_MAX]

    def get_lambda(self, path):
        _lambda = []
        try:
            with open(path) as fp:
                for line in fp:
                    if not line.startswith('#'):
                        data = line.split()
                        _lambda.append(float(data[0])*1e-6) # wavelength in unit of meter
            return np.array(_lambda)
        except IOError:
            print('SED file is unreadable - permission problem?')
            return np.array([])

    def read_cell_data(self, cell_file):
        xc = []
        yc = []
        zc = []
        try:
            with open(cell_file) as fp:
                for line in fp:
                    if not line.startswith('#'):
                        data = line.split()
                        xc.append(float(data[1]))
                        yc.append(float(data[2]))
                        zc.append(float(data[3]))
            return np.array(xc), np.array(yc), np.array(zc)
        except IOError:
            print('Cell file is missing - permission problem?')
            return np.array([]), np.array([]), np.array([])

    def read_gas_data(self, gas_file):
        x = []
        y = []
        z = []
        try:
            with open(gas_file) as fp:
                for line in fp:
                    if not line.startswith('#'):
                        data = line.split()
                        x.append(float(data[0]))
                        y.append(float(data[1]))
                        z.append(float(data[2]))
            print('In total there are %d particles found.' % len(x))
            return np.array(list(zip(range(len(x)), x, y, z)))
        except IOError:
            print('Gas file is missing - permission problem?')
            return np.array([[],[],[],[]])

    def calculate_ISRF(self, J_file):
        try:
            with open(J_file) as fp:
                G_cell = []
                Nion_cell = []
                for line in fp:
                    if not line.startswith('#'):
                        data = line.split()
                        data = np.array([float(item) for item in data])
                        data = data[1:]

                        y1 = data[(self._lambda > LAMBDA_MIN) & (self._lambda < LAMBDA_MAX)]
                        G = integrate.simps(y1/self._lambda_Habing, self._lambda_Habing) * (4.*np.pi) / G_0

                        y2 = data[self._lambda < LAMBDA_MAX]
                        Nion = integrate.simps(y2, self._lambda_ion) / (PLANCK*SPEED_OF_LIGHT**2) * (4.*np.pi)
                        Nion /= 1e6

                        G_cell.append(G)
                        Nion_cell.append(Nion)
            return G_cell, Nion_cell
        except IOError:
            print('!!!!!!!!!!!!!! ISRF file is unreadable - permission problem? !!!!!!!!!!!!!!!')
            return [], []

def assign_G(args):
    processor, position, iset = args
    xc, yc, zc = processor.xc, processor.yc, processor.zc
    G_cell = processor.G_cell
    Nion_cell = processor.Nion_cell

    rA = np.sqrt((xc - position[1])**2 + (yc - position[2])**2 + (zc - position[3])**2)

    path_to_output = './ISRF_%d.out' % iset
    with open(path_to_output,'a') as outfile:
        indice = np.where(rA == np.amin(rA))
        g = G_cell[indice[0][0]]
        n = Nion_cell[indice[0][0]]
        outfile.write('%d %d %.4f %.4g\n' % (position[0], indice[0][0], g, n))

if __name__=='__main__':
    # Get the number of available CPU cores
    mp = multiprocessing.cpu_count()
    print('You have {0:1d} CPUs'.format(mp))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Calculate the ISRF distribution of a simulated galaxy. (c) Lichen Liang 2021')
    parser.add_argument('-s', '--snapid', type=int, default=1200, help='Snapshot number, default "1200"')
    parser.add_argument('-i', '--haloid', type=int, default=9, help='Galaxy ID, default "9"')
    parser.add_argument('-N', '--number', type=int, default=72, help='Number of particles to be processed, default "72"')
    parser.add_argument('-C', '--set', type=int, default=0, help='Set number, "0"')
    args = parser.parse_args()
    ihalo = args.haloid
    isnap = args.snapid
    npar = args.number
    iset = args.set

    # Generate file paths for data
    skirt_dir = f'/home/m/murray/lichenli/lichenli/FIRE/FIREBox/snapshot_{isnap:03d}/{ihalo:05d}/'
    path_to_gas_file = f'{skirt_dir}gas_{isnap:03d}_{ihalo:03d}SRnm_faceon.txt'
    path_to_sed_file = f'{skirt_dir}HR_wd01mw_bpass_dz4_cmb_i0_sed.dat'
    path_to_cell_file = f'{skirt_dir}HR_wd01mw_bpass_dz4_cmb_grid_cellprops.dat'
    path_to_J_file = f'{skirt_dir}HR_wd01mw_bpass_dz4_cmb_grid_J.dat'

    # Initialize ISRFProcessor to process data
    processor = ISRFProcessor(path_to_sed_file, path_to_cell_file, path_to_J_file, path_to_gas_file)
    positions = processor.read_gas_data(path_to_gas_file)
    processor.xc, processor.yc, processor.zc = processor.read_cell_data(path_to_cell_file)
    processor.G_cell, processor.Nion_cell = processor.calculate_ISRF(path_to_J_file)

    # Print particle range for processing
    print('Particle IDs:', npar * iset, npar * (iset + 1))

    # Multiprocessing using a pool of workers
    pool = Pool(processes=mp)
    pool.map(assign_G, [(processor, position, iset) for position in positions[npar * iset:npar * (iset + 1)]])
