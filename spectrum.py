import numpy as np
from params import PhysConst, CompParams
import funcs
from DataHandler import DataHandler

class ComputeKHCS:
    def __init__(self, omega0, FWHM, inpf, outf):
        self.omega0 = omega0 / PhysConst.AU2EV.value
        self.sigma = FWHM / (2*np.sqrt(2*np.log(2))*PhysConst.AU2EV.value)

        self.omega_fl = np.linspace(CompParams.FL_LOW.value, 
                                    CompParams.FL_HIGH.value, 
                                    CompParams.FL_BINS.value)

        self.KH_CS_inc = np.zeros(len(self.omega_fl))
        self.KH_CS_coh = np.zeros(len(self.omega_fl))
        self.KH_CSP_inc = np.zeros(len(self.omega_fl))
        self.KH_CSP_coh = np.zeros(len(self.omega_fl))
        self.df_abs, self.df_absP, self.df_fl, self.df_flP = DataHandler.read_dfs(inpf, CompParams.DATAFRAMES.value)
        self.outf = outf

    def process_data(self):
        for Tframe in np.unique(self.df_abs.time.to_numpy()):
            for geom in np.unique(self.df_abs.geom.to_numpy()):
                #print(Tframe, geom)
                self.process_each_frame(Tframe, geom, pumped=False)
                self.process_each_frame(Tframe, geom, pumped=True)
        

    def process_each_frame(self, Tframe, geom, pumped):
        TDMAbs, AbsEn, TDMFlu, FlEn, EnShift = ComputeKHCS.get_matrices(self, Tframe, geom, pumped)
        if pumped:
            CSP_inc, CSP_coh = SpectrumCalculator.calc_khcs(TDMAbs, AbsEn, TDMFlu, FlEn, EnShift, 
                                                            self.omega0, self.sigma, self.omega_fl, pumped)
            self.KH_CSP_inc += CSP_inc / (len(np.unique(self.df_abs.time.to_numpy())) * len(np.unique(self.df_abs.geom.to_numpy())))
            self.KH_CSP_coh += np.real(CSP_coh) /(len(np.unique(self.df_abs.time.to_numpy())) * len(np.unique(self.df_abs.geom.to_numpy())))
        else:
            CS_inc, CS_coh = SpectrumCalculator.calc_khcs(TDMAbs, AbsEn, TDMFlu, FlEn, EnShift, 
                                                          self.omega0, self.sigma, self.omega_fl, pumped)
            self.KH_CS_inc += CS_inc / (len(np.unique(self.df_abs.time.to_numpy())) * len(np.unique(self.df_abs.geom.to_numpy())))
            self.KH_CS_coh += np.real(CS_coh) / (len(np.unique(self.df_abs.time.to_numpy())) * len(np.unique(self.df_abs.geom.to_numpy())))

    def get_matrices(self, Tframe, geom, pumped):    
        if pumped:
            df_abs     = self.df_absP
            df_fl      = self.df_flP
            NInterState = np.unique(df_fl.istate.to_numpy())
            NInitState = np.unique(df_abs.istate.to_numpy())
            AbsEn      = np.zeros((len(NInterState),len(NInitState)))
            TDMAbs     = np.zeros((len(NInterState), len(NInitState), 3))
            NFinalState= 1000
        else:
            df_abs     = self.df_abs
            df_fl      = self.df_fl
            NInterState = np.unique(df_fl.istate.to_numpy())
            AbsEn      = np.zeros(len(NInterState))
            TDMAbs     = np.zeros((len(NInterState), 3))
            NFinalState= 101
        
        FlEn           = np.zeros((len(NInterState),NFinalState))
        TDMFlu        = np.zeros((len(NInterState), NFinalState, 3))

        for state_indx in NInterState:
            FlEn [state_indx-1, :] = (df_fl[(df_fl.time == Tframe) & 
                                        (df_fl.geom == geom) & 
                                        (df_fl.istate == state_indx)]
                                        [['fl_En']].to_numpy())[0:NFinalState,0]
            TDMFlu[state_indx-1, :, :] = (df_fl[(df_fl.time == Tframe) & 
                                        (df_fl.geom == geom) & 
                                        (df_fl.istate == state_indx)]
                                        [['TDMx', 'TDMy', 'TDMz']].to_numpy())[0:NFinalState, :]

            if pumped:
                AbsEn[state_indx-1,:] = (df_abs[(df_abs.time == Tframe) & 
                                             (df_abs.geom == geom) & 
                                             (df_abs.fstate == state_indx)]
                                             [['abs_En']].to_numpy())[:,0]
                TDMAbs[state_indx-1, :, :] = (df_abs[(df_abs.time == Tframe) & 
                                             (df_abs.geom == geom) & 
                                             (df_abs.fstate == state_indx)]
                                             [['TDMx', 'TDMy', 'TDMz']].to_numpy())
                if state_indx == 1:
                    EnShift = (AbsEn[state_indx-1, :] / PhysConst.AU2EV.value)[:, np.newaxis] - \
                              (FlEn[state_indx-1, :] / PhysConst.AU2EV.value)[np.newaxis, :]
            else:
                AbsEn[state_indx-1] = ( df_abs[(df_abs.time == Tframe) & 
                                          (df_abs.geom == geom) & 
                                          (df_abs.fstate == state_indx)]
                                          [['abs_En']].to_numpy())
                TDMAbs[state_indx-1, :] = (df_abs[(df_abs.time == Tframe) & 
                                          (df_abs.geom == geom) & 
                                          (df_abs.fstate == state_indx)]
                                          [['TDMx', 'TDMy', 'TDMz']].to_numpy())
                if state_indx == 1:
                    EnShift = AbsEn[state_indx-1] / PhysConst.AU2EV.value - \
                              FlEn[state_indx-1, :] / PhysConst.AU2EV.value
                    
        return TDMAbs, AbsEn, TDMFlu, FlEn, EnShift


class SpectrumCalculator:
    @staticmethod
    def calc_khcs(TDMAbs, AbsEn, TDMFlu, FluEn, EnShift, omega0, sigma, omega_fl, pumped):
        if not pumped:
            return SpectrumCalculator._calc_khcs_unpumped(TDMAbs, AbsEn, TDMFlu, FluEn, 
                                                          EnShift, omega0, sigma, omega_fl)
        else:
            return SpectrumCalculator._calc_khcs_pumped(TDMAbs, AbsEn, TDMFlu, FluEn, 
                                                        EnShift, omega0, sigma, omega_fl)

    @staticmethod
    def _calc_khcs_unpumped(TDMAbs, AbsEn, TDMFlu, FlEn, EnShift, omega0, sigma, omega_fl):
        n_interm_state = AbsEn.shape[0]
        n_final_state  = FlEn.shape[1]
        kh_inc = np.zeros((len(omega_fl), n_final_state))
        kh_coh = np.zeros((len(omega_fl), n_final_state), dtype=np.complex128)
        for i in range(n_interm_state):
            denom = funcs.calc_denom(omega_fl, FlEn[i, :], PhysConst.GAMMA.value)
            inc_contrib = SpectrumCalculator._calc_inc_contrib(TDMAbs[i, :], TDMFlu[i, :, :], denom)
            coh_contrib = SpectrumCalculator._calc_coh_contrib_unpumped(TDMAbs, TDMFlu, FlEn, i, omega_fl)
            kh_inc += inc_contrib
            kh_coh += inc_contrib + coh_contrib
        return SpectrumCalculator._sum_khcs_unpumped(omega_fl, EnShift, omega0, sigma, kh_inc, kh_coh)

    @staticmethod
    def _calc_khcs_pumped(TDMAbs, AbsEn, TDMFlu, FlEn, EnShift, omega0, sigma, omega_fl):
        n_interm_state = AbsEn.shape[0]
        n_init_state   = AbsEn.shape[1]
        n_final_state  = FlEn.shape[1]
        kh_inc = np.zeros((len(omega_fl), n_init_state, n_final_state))
        kh_coh = np.zeros((len(omega_fl), n_init_state, n_final_state), dtype=np.complex128)
        for i in range(n_interm_state):
            denom = funcs.calc_denom(omega_fl, FlEn[i, :], PhysConst.GAMMA.value)
            for k in range(n_init_state):
                inc_contrib = SpectrumCalculator._calc_inc_contrib(TDMAbs[i, k, :], TDMFlu[i, :], denom)
                coh_contrib = SpectrumCalculator._calc_coh_contrib_pumped(TDMAbs, TDMFlu, FlEn, i, k, omega_fl)
                kh_inc[:, k, :] += inc_contrib
                kh_coh[:, k, :] += inc_contrib + coh_contrib
        return SpectrumCalculator._sum_khcs_pumped(omega_fl, EnShift, omega0, sigma, kh_inc, kh_coh)

    @staticmethod
    def _calc_inc_contrib(tdma, tdmf, denom):
        return funcs.calc_num_inc(tdma, tdmf, CompParams.BETA.value)[np.newaxis, :] / (np.real(denom) ** 2 + np.imag(denom) ** 2)

    @staticmethod
    def _calc_coh_contrib_unpumped(TDMAbs, TDMFlu, FlEn, i, omega_fl):
        n_interm_state = TDMAbs.shape[0]
        n_final_state  = TDMFlu.shape[1]
        coh_contrib = np.zeros((len(omega_fl), n_final_state), dtype=np.complex128)
        for j in range(n_interm_state):
            if j != i:
                coh_contrib += SpectrumCalculator._calc_coh_contrib_single(TDMAbs[i, :], TDMAbs[j, :], 
                                                                           TDMFlu[i, :, :], TDMFlu[j, :, :], 
                                                                           FlEn[i,:], FlEn[j,:], omega_fl)
        return coh_contrib

    @staticmethod
    def _calc_coh_contrib_pumped(TDMAbs, TDMFlu, FlEn, i, k, omega_fl):
        n_interm_state = TDMAbs.shape[0]
        n_final_state = FlEn.shape[1]
        coh_contrib = np.zeros((len(omega_fl), n_final_state), dtype=np.complex128)
        for j in range(n_interm_state):
            if j != i:
                coh_contrib += SpectrumCalculator._calc_coh_contrib_single(TDMAbs[i, k, :], TDMAbs[j, k, :], 
                                                                           TDMFlu[i, :, :], TDMFlu[j, :, :], 
                                                                           FlEn[i, :], FlEn[j, :], omega_fl)
        return coh_contrib

    @staticmethod
    def _calc_coh_contrib_single(tdma1, tdma2, tdmf1, tdmf2, FlEni, FlEnj, omega_fl):
        return ( funcs.calc_num_coh(tdma1, tdma2, tdmf1, tdmf2, CompParams.BETA.value)[np.newaxis, :] 
                / (funcs.calc_denom(omega_fl, FlEni, PhysConst.GAMMA.value) 
                   * np.conjugate(funcs.calc_denom(omega_fl, FlEnj, PhysConst.GAMMA.value)))
        )

    @staticmethod
    def _sum_khcs_unpumped(omega_fl, EnShift, omega0, sigma, kh_inc, kh_coh):
        factor = (omega_fl[:, np.newaxis] + EnShift) * omega_fl[:, np.newaxis] ** 3
        gaussian = funcs.profile_gaussian(omega_fl[:, np.newaxis] + EnShift, omega0, sigma)
        khcs_inc = np.sum(factor * kh_inc * gaussian, axis=1)
        khcs_coh = SpectrumCalculator._validate_and_sum_kh_coh(factor, kh_coh, gaussian)
        return khcs_inc, np.real(khcs_coh)
    
    @staticmethod
    def _sum_khcs_pumped(omega_fl, EnShift, omega0, sigma, kh_inc, kh_coh):
        factor = (omega_fl[:, np.newaxis, np.newaxis] + EnShift[np.newaxis, :, :]) * omega_fl[:, np.newaxis, np.newaxis] ** 3
        gaussian = funcs.profile_gaussian(omega_fl[:, np.newaxis, np.newaxis] + EnShift[np.newaxis, :, :], 
                                          omega0, sigma)
        khcs_inc = np.sum(factor * kh_inc * gaussian, axis=(1, 2))/kh_inc.shape[1]
        khcs_coh = np.sum(SpectrumCalculator._validate_and_sum_kh_coh(factor, kh_coh, gaussian), axis=-1)/kh_coh.shape[1]
        return khcs_inc, khcs_coh

    @staticmethod
    def _validate_and_sum_kh_coh(factor, kh_coh, gaussian):
        if np.any(np.imag(kh_coh) > 1.e-16):
            raise ValueError("KH_coh is not all real!")
        return np.sum(factor * kh_coh * gaussian, axis=-1)
    
