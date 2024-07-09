import argparse
from params import CompParams, PhysConst
import plotter
import math
import time
from DataHandler import DataHandler
from spectrum import ComputeKHCS


def parse_cla():
    parser = argparse.ArgumentParser(description='Calculate Kramers Heisenberg cross section for pumped and unpumed  water')
    parser.add_argument('-inpf', type=str, required=True, help='Input data file in hdf5 format')
    parser.add_argument('-omega0', type=float, default=535, help='Central frequency of probe pulse in eV')
    parser.add_argument('-FWHM', type=float, default=0.5, help='FWHM of probe pulse in eV')
    parser.add_argument('-outf', type=str, default='KH_CS.h5', help='Directory to save the output')
    parser.add_argument('-plot', action='store_true', help='flags if output data is plotted at the end')
    args = parser.parse_args()

    return args

def main(args):
    KH_Container = ComputeKHCS(args.omega0, args.FWHM, args.inpf, args.outf)
    ComputeKHCS.process_data(KH_Container)
    dfout1 = DataHandler.write_dfs(KH_Container.outf, KH_Container.omega_fl, KH_Container.KH_CS_inc, 
                                   KH_Container.KH_CS_coh, pumped=False)
    dfout2 = DataHandler.write_dfs(KH_Container.outf, KH_Container.omega_fl, KH_Container.KH_CSP_inc, 
                                   KH_Container.KH_CSP_coh, pumped=True)

    if args.plot:
       plotter.plot_spectrum(dfout1, dfout2)

if __name__ == '__main__':
    start = time.monotonic()
    args = parse_cla()
    main(args)
    end = time.monotonic()
    print("Time elapsed during the process:", end - start)

