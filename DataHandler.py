import pandas as pd
from params import PhysConst
#from spectrum import SpectrumCalculator
#from spectrum import ComputeKHCS

class DataHandler:
    @staticmethod
    def read_dfs(inpf, dframes):
        return [pd.read_hdf(inpf, dframe) for dframe in dframes]

    @staticmethod
    def write_dfs(outf, omega_fl, khcs_inc, khcs_coh, pumped):
        dfout = pd.DataFrame({'En': omega_fl*PhysConst.AU2EV.value, 'inc': khcs_inc, 'coh': khcs_coh})
        if pumped:
            keyval = 'pumped_KH'        
        else:
            keyval = 'unpumped_KH'
        dfout.to_hdf(outf, key=keyval, mode='a')
        return dfout
    