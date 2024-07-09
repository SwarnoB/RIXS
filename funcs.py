import numpy as np
import numba as nb
from params import CompParams, PhysConst

@nb.njit()
def profile_gaussian(x, x0, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

@nb.njit
def manual_dot_product(vec1, vec2):
    result = 0.0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return result

@nb.njit('float64(float64[::1], float64[::1], float64)')
def calc_inc_sum(tdma, tdmf, cos2beta):
    #term1 = np.linalg.norm(tdma)**2 * np.linalg.norm(tdmf)**2
    term1 = ( (tdma[0]*tdma[0] + tdma[1]*tdma[1] + tdma[2]*tdma[2]) 
             * (tdmf[0]*tdmf[0] + tdmf[1]*tdmf[1] + tdmf[2]*tdmf[2]) )
    dot = np.dot(tdma, tdmf)
    term2 = dot*dot
    return 1/30 * (term1*(3+cos2beta) + term2*(1-3*cos2beta))

@nb.njit('float64[:](float64[::1], float64[:,::1], float64)')
def calc_num_inc(tdma, tdmf, beta):
    cos2beta = np.cos(np.radians(2 * beta))
    num = np.zeros(tdmf.shape[0], dtype=np.float64)
    for i in range(tdmf.shape[0]):
        num[i] = calc_inc_sum(tdma, tdmf[i,:], cos2beta)
    return num


@nb.njit('float64(float64[::1], float64[::1], float64[::1], float64[::1], float64)')
def _calc_coh_sum(tdma1, tdma2, tdmf1, tdmf2, cos2beta):
    term1 = np.dot(tdma1,tdmf1) * np.dot(tdma2,tdmf2) 
    term2 = np.dot(tdma1,tdmf2) * np.dot(tdmf1,tdma2) 
    term3 = np.dot(tdma1,tdma2) * np.dot(tdmf1,tdmf2)
    return (1 / 30) * ((1/2) * (term1 + term2) * (1-3*cos2beta) + term3 * (3+cos2beta))

@nb.njit('float64[:](float64[::1], float64[::1], float64[:,::1], float64[:,::1], float64)')
def calc_num_coh(tdma1, tdma2, tdmf1, tdmf2, beta):
    cos2beta = np.cos(np.radians(2 * beta))
    num = np.zeros(tdmf1.shape[0], dtype=np.float64)
    for i in range(tdmf1.shape[0]):
        num[i] = _calc_coh_sum(tdma1, tdma2, tdmf1[i,:], tdmf2[i,:], cos2beta)
    return num

@nb.njit('complex128[:,:](float64[::1], float64[::1], float64)')
def calc_denom(omega_fl, fl_en, gamma):
    return omega_fl[:, np.newaxis] - (fl_en / PhysConst.AU2EV.value)[np.newaxis, :] + 1j * gamma / 2
