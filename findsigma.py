 #!/usr/bin/env python3
from scipy import spatial
import numpy as np
import argparse
import os
import multiprocessing
from multiprocessing import Pool

def read_gas_data(path_to_gas_file):
    """Read gas particle data from a file and extract relevant information."""
    global tree, vx, vy, vz
    x=[]
    y=[]
    z=[]
    h=[]
    vx=[]
    vy=[]
    vz=[]
    try:
        with open(path_to_gas_file) as fp:
            for line in fp:
                if not line.startswith('#'):
                    data = line.split()
                    x.append(float(data[0]))   # pc
                    y.append(float(data[1]))   # pc
                    z.append(float(data[2]))   # pc
                    h.append(float(data[3]))   # pc
                    vx.append(float(data[7]))  # km/s
                    vy.append(float(data[8]))  # km/s
                    vz.append(float(data[9]))  # km/s
    except IOError:
        print('!!!!!!!!!!!!!! Particle file is unreadable - permission problem? !!!!!!!!!!!!!!!')
    print('Number of gas particles that are processed in total:', len(x))

    tree = spatial.KDTree(np.array(list(zip(x, y, z))))
    positions = np.array(list(zip(range(len(x)), x, y, z, h)))
    return positions

def get_missing_data(path_to_mis_gas_file, all_positions):
    """Get positions of missing gas particles."""
    missing_ids=[]
    with open(path_to_mis_gas_file) as fp:
        for line in fp:
            missing_ids.append(int(line.split()[0]))

    positions = np.array([all_positions[id_] for id_ in missing_ids])
    print('%d missing particles.' % len(missing_ids))
    return positions

def find_sigma(position):
    """Calculate the velocity dispersion (sigma) for a particle based on its nearest neighbors."""
    global tree, vx, vy, vz, iset

    path_to_output = 'sigma_%d.out' % iset
    with open(path_to_output,'a') as outfile:
        index = tree.query_ball_point([position[1], position[2], position[3]], position[4])
        index = np.array(index)

        maxid = len(index)
        vx_kernel = np.array([vx[index[i]] for i in range(maxid)])
        vy_kernel = np.array([vy[index[i]] for i in range(maxid)])
        vz_kernel = np.array([vz[index[i]] for i in range(maxid)])

        sigma = (np.std(vx_kernel)**2 + np.std(vy_kernel)**2 + np.std(vz_kernel)**2)**0.5
        print('index = %d, sigma=%.3g' % (position[0], sigma), 'km/s')
        outfile.write(str(int(position[0])) + ' ' + '%.4g' % float(sigma) + '\n')

def main():

    mp = multiprocessing.cpu_count()
    print('You have {0:1d} CPUs'.format(mp))

    parser = argparse.ArgumentParser(description='(c) Lichen Liang 2020')
    parser.add_argument('simulation', type=str, help="simulation name (default 'FB15N1024')", default='FB15N1024')
    parser.add_argument('-N', '--number',type=int, help='number of particles to be processed (int -> default "72")', default=72)
    parser.add_argument('-C', '--set', type=int, help='set number (int -> default "0")', default=0)
    parser.add_argument('-s', '--snapid', type=int, help='snapshot number (int -> default "832")', default=832)
    parser.add_argument('-i', '--haloid', type=int, help='halo id (int -> default "10")', default=10)
    args = parser.parse_args()

    global iset
    sim = args.simulation
    npar = args.number
    iset = args.set
    isnap = args.snapid
    ihalo = args.haloid

    skirt_dir = '/home/m/murray/lichenli/lichenli/FIRE/FIREBox/snapshot_%03d/%05d/' % (isnap, ihalo)
    path_to_gas_file = skirt_dir + 'gas_%03d_%03dSRnm_faceon.txt' % (isnap, ihalo)
    path_to_mis_gas_file = skirt_dir + '%s_%05d_snap%03d_sigma_mis.txt' % (sim, ihalo, isnap)

    if os.path.exists(path_to_mis_gas_file):
        all_positions = read_gas_data(path_to_gas_file)
        positions = get_missing_data(path_to_mis_gas_file, all_positions)
    else:
        positions = read_gas_data(path_to_gas_file)

    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    pool = Pool(processes=mp)
    print('Particle IDs:', npar*iset, npar*(iset+1))
    pool.map(find_sigma, positions[npar * iset: npar * (iset + 1)])

if __name__ == '__main__':
    main()
