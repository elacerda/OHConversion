#!/usr/bin/python
import CALIFAUtils
from CALIFAUtils.lines import Lines
import numpy as np

def O3N2(Ha_obs, Hb_obs, O3_obs, N2_obs, mask_zones, tau_V = None, correct = False):
    Hb = np.ma.masked_array(Hb_obs, mask = mask_zones)
    O3 = np.ma.masked_array(O3_obs, mask = mask_zones)
    Ha = np.ma.masked_array(Ha_obs, mask = mask_zones)
    N2 = np.ma.masked_array(N2_obs, mask = mask_zones)
    if correct is True:
        tau_V_m = np.ma.masked_array(tau_V, mask = mask_zones)
        from pystarlight.util import redenninglaws
        q = redenninglaws.Cardelli_RedLaw([4861, 5007, 6563, 6583])
        Hb *= np.ma.exp(q[0] * tau_V_m) 
        O3 *= np.ma.exp(q[1] * tau_V_m) 
        Ha *= np.ma.exp(q[2] * tau_V_m) 
        N2 *= np.ma.exp(q[3] * tau_V_m)
    O3Hb = np.ma.log10(O3/Hb)
    N2Ha = np.ma.log10(N2/Ha)
    O3N2 = np.ma.log10(O3 * Ha / (N2 * Hb))
    return O3Hb, N2Ha, O3N2

dtCid = np.dtype([('Hb_obs', np.float),
                  ('O3_obs', np.float),
                  ('Ha_obs', np.float),
                  ('N2_obs', np.float),
                  ('SN_Hb_obs', np.float),
                  ('SN_O3_obs', np.float),
                  ('SN_Ha_obs', np.float),
                  ('SN_N2_obs', np.float),
                  ('AV', np.float)])

dtMari = np.dtype([('Zneb_mpa', np.float),
                   ('Ha_obs', np.float),
                   ('Hb_obs', np.float),
                   ('O3_obs', np.float),
                   ('N2_obs', np.float),
                   ('AV_lines' , np.float)])

txtCid = np.loadtxt('Line4EAD.txt', dtype = dtCid)
txtMari = np.loadtxt('Z_mpa_lines.txt', dtype = dtMari)

Zneb_mpa = np.ma.masked_array(txtMari['Zneb_mpa'], mask = ((txtMari['Zneb_mpa'] == -99.9) |  (txtMari['Zneb_mpa'] == -999.)), dtype = np.float)
Ha_obs = np.ma.masked_array(txtMari['Ha_obs'], mask = (txtMari['Ha_obs'] == -999.), dtype = np.float)
Hb_obs = np.ma.masked_array(txtMari['Hb_obs'], mask = (txtMari['Hb_obs'] == -999.), dtype = np.float)
O3_obs = np.ma.masked_array(txtMari['O3_obs'], mask = (txtMari['O3_obs'] == -999.), dtype = np.float)
N2_obs = np.ma.masked_array(txtMari['N2_obs'], mask = (txtMari['N2_obs'] == -999.), dtype = np.float)
SN_Ha_obs = np.ma.masked_array(txtCid['SN_Ha_obs'], mask = (txtCid['SN_Ha_obs'] == -999.), dtype = np.float)
SN_Hb_obs = np.ma.masked_array(txtCid['SN_Hb_obs'], mask = (txtCid['SN_Hb_obs'] == -999.), dtype = np.float)
SN_O3_obs = np.ma.masked_array(txtCid['SN_O3_obs'], mask = (txtCid['SN_O3_obs'] == -999.), dtype = np.float)
SN_N2_obs = np.ma.masked_array(txtCid['SN_N2_obs'], mask = (txtCid['SN_N2_obs'] == -999.), dtype = np.float)
AV_lines = np.ma.masked_array(txtMari['AV_lines'], mask = (txtMari['AV_lines'] == -999.), dtype = np.float)

m_not_OK = Ha_obs.mask | Hb_obs.mask | O3_obs.mask | N2_obs.mask
m_not_OK |= (SN_Ha_obs < 3)
m_not_OK |= (SN_Hb_obs < 3)
m_not_OK |= (SN_O3_obs < 3)
m_not_OK |= (SN_N2_obs < 3)

O3Hb, N2Ha, O3N2 = O3N2(Ha_obs, Hb_obs, O3_obs, N2_obs, m_not_OK, CALIFAUtils.GasProp().AVtoTau(AV_lines), correct = True)
l = Lines()
axis = [-2.5, 1.0, -1.6, 1.6]
