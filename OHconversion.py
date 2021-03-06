#!/usr/bin/python
from scipy.ndimage.filters import gaussian_filter1d
from CALIFAUtils.scripts import calc_running_stats
from CALIFAUtils.plots import plotOLSbisectorAxis
from matplotlib import pyplot as plt
from CALIFAUtils.lines import Lines
import matplotlib as mpl
import numpy as np
import CALIFAUtils
import seaborn.apionly as sns

#debug = True
debug = False

mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

def calc_O3N2(Hb_obs, O3_obs, Ha_obs, N2_obs, mask_zones, tau_V = None, correct = False):
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
  
if __name__ == '__main__':
    l = Lines(xn = 100)

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
    
    if debug is True:
        txtCid = np.loadtxt('Line4EAD_100.txt', dtype = dtCid)
        txtMari = np.loadtxt('Z_mpa_lines_100.txt', dtype = dtMari)
    else:
        txtCid = np.loadtxt('Line4EAD.txt', dtype = dtCid)
        txtMari = np.loadtxt('Z_mpa_lines.txt', dtype = dtMari)
    
    m_aux = (txtMari['Zneb_mpa'] == -99.9) | (txtMari['Zneb_mpa'] == -999.)
    Zneb_mpa = np.ma.masked_array(txtMari['Zneb_mpa'], mask = m_aux, dtype = np.float)
    m_aux = txtMari['Ha_obs'] == -999.
    Ha_obs = np.ma.masked_array(txtMari['Ha_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtMari['Hb_obs'] == -999.
    Hb_obs = np.ma.masked_array(txtMari['Hb_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtMari['O3_obs'] == -999.
    O3_obs = np.ma.masked_array(txtMari['O3_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtMari['N2_obs'] == -999.
    N2_obs = np.ma.masked_array(txtMari['N2_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtCid['SN_Ha_obs'] == -999.
    SN_Ha_obs = np.ma.masked_array(txtCid['SN_Ha_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtCid['SN_Hb_obs'] == -999.
    SN_Hb_obs = np.ma.masked_array(txtCid['SN_Hb_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtCid['SN_O3_obs'] == -999.
    SN_O3_obs = np.ma.masked_array(txtCid['SN_O3_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtCid['SN_N2_obs'] == -999.
    SN_N2_obs = np.ma.masked_array(txtCid['SN_N2_obs'], mask = m_aux, dtype = np.float)
    m_aux = txtMari['AV_lines'] == -999.
    AV_lines = np.ma.masked_array(txtMari['AV_lines'], mask = m_aux, dtype = np.float)
    
    m_gal_not_OK = np.bitwise_or(Ha_obs.mask, Hb_obs.mask)
    m_gal_not_OK = np.bitwise_or(m_gal_not_OK, O3_obs.mask)
    m_gal_not_OK = np.bitwise_or(m_gal_not_OK, N2_obs.mask)
    m_gal_not_OK = np.bitwise_or(m_gal_not_OK, np.ma.less(SN_Ha_obs, 3))
    m_gal_not_OK = np.bitwise_or(m_gal_not_OK, np.ma.less(SN_Hb_obs, 3))
    m_gal_not_OK = np.bitwise_or(m_gal_not_OK, np.ma.less(SN_O3_obs, 3))
    m_gal_not_OK = np.bitwise_or(m_gal_not_OK, np.ma.less(SN_N2_obs, 3))
    
    tau_V_lines = AV_lines * 1. / (2.5 * np.log10(np.exp(1.)))
    
    logO3Hb, logN2Ha, logO3N2 = calc_O3N2(Hb_obs, O3_obs, 
                                          Ha_obs, N2_obs, 
                                          m_gal_not_OK, tau_V_lines, 
                                          correct = True)
    
    logOH_M13 = 8.533 - 0.214 * logO3N2
    
    xm, ym = CALIFAUtils.ma_mask_xyz(x = logOH_M13, y = Zneb_mpa)
    
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # f = plt.figure()
    # f.set_dpi(100)
    # f.set_size_inches(10, 8)    
    # ax = f.gca()
    # ax.scatter(logN2Ha, logO3Hb, marker = '.', s = 1, c = '0.7', edgecolor = 'none', alpha = 0.6, label = '')
    # #m_aux = l.maskAbovelinebpt('K01', logN2Ha, logO3Hb)
    # ax.scatter(logN2Ha[~(Zneb_mpa.mask | m_aux)], logO3Hb[~(Zneb_mpa.mask | m_aux)], marker = '.', s = 5, c = 'b', edgecolor = 'none', alpha = 0.6, label = '')
    # for line in l.linesbpt:
    #     ax.plot(l.x[line], l.y[line], label = line)
    # #axis = [-2.5, 1.0, -1.6, 1.6]
    # ax.set_xlim(-2.5, 1.0)
    # ax.set_ylim(-1.6, 1.6) 
    # ax.set_xlabel(r'$\log\ ([NII]\lambda 6584 / H\alpha)$')    
    # ax.set_ylabel(r'$\log\ ([OIII]\lambda 5007 / H\beta)$')
    # plt.legend(loc = 'best')
    # plt.grid()
    # f.savefig('BPT.png')
    # plt.close(f)
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    
    #ax.autoscale(False)
    H, xedges, yedges = np.histogram2d(xm.compressed(), ym.compressed(), bins = (40,40))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    #f = plt.figure()
    #f.set_dpi(100)
    #f.set_size_inches(10, 8)
    #ax = f.gca()
    #with sns.axes_style("white"):
    sns.jointplot(xm.compressed(), ym.compressed(), kind="hex")
    f = plt.gcf()
    f.set_dpi(100)
    f.set_size_inches(10, 8)
    ax = f.axes[0]
    #print f.axes    
    #ax = f.axes
    
    im = ax.imshow(H, cmap=plt.cm.Blues, aspect = 'auto', interpolation='none', origin='low', extent = extent)
    #ax.scatter(xm, ym, marker = 'o', c = 'g', s = 5, edgecolor = 'none', alpha = 0.1, label = '')
    #ax.contour(H,extent=extent,linewidths=1, interpolation='nearest', origin = 'lower', levels=[16, 50, 84], cmap = plt.cm.YlGn)
    #ax.hist2d(xm.compressed(), ym.compressed(), bins = (10,10))
    
    nBox = 1000
    #nBox = len(xm.compressed()) / 50.
    #slope, intercept, sigma_slope, sigma_intercep = OLS_bisector(logOH_M13, Zneb_mpa)
    pos_x = 0.99
    pos_y = 0.
    kwargs_ols = dict(x_rms = xm, y_rms = ym, pos_x = pos_x, pos_y = pos_y, fs = 10, c = '#7B8DBF', rms = True, label = 'OLS', text = True)
    a, b, sa, sb = plotOLSbisectorAxis(ax, xm, ym, **kwargs_ols)
      
    dxBox = (xm.max() - xm.min()) / (nBox - 1.)
    kwargs_rs = dict(dxBox = dxBox, xbinIni = xm.min(), xbinFin = xm.max(), xbinStep = dxBox)    
    xbinCenter, xMedian, xMean, xStd, yMedian, \
        yMean, yStd, nInBin, xPrc, yPrc = calc_running_stats(xm, ym, **kwargs_rs)
    #ax.plot(xMedian, yMedian, '-.b', label = 'Median in x')
    sig = 40 / np.sqrt(8. * np.log(2))
    xM = np.ma.masked_array(xMedian)
    yM = np.ma.masked_array(yMedian)
    m_gs = np.isnan(xM) | np.isnan(yM) 
    xS = gaussian_filter1d(xM[~m_gs], sig)
    yS = gaussian_filter1d(yM[~m_gs], sig)
    ax.plot(xS, yS, c = 'r', ls = '--', label = 'smoothed median in x')
    pos_x = 0.99
    pos_y = 0.04
    kwargs_ols = dict(x_rms = xm, y_rms = ym, pos_x = pos_x, pos_y = pos_y, fs = 10, c = '#BA3E04', rms = True, label = 'OLS(tend xy)', text = True)
    a_tend, b_tend, sa_tend, sb_tend = plotOLSbisectorAxis(ax, xS, yS, **kwargs_ols)
      
    p = []
    for i in range(3):
        order = i + 1
        p.append(np.polyfit(xS, yS, order))
        linename = r'fit poly %d' % order
        l.addLine(linename, np.polyval, p[i], np.linspace(7.5, 9.0, l.xn + 1))
        rms = (ym.compressed() - np.polyval(p[i], xm.compressed())).std()
        print order, p[i], rms 
        ax.plot(l.x[linename], l.y[linename], label = '%s (rms: %.3f)'% (linename, rms))
  
    dxBox = (ym.max() - ym.min()) / (nBox - 1.)
    kwargs_rs = dict(dxBox = dxBox, xbinIni = ym.min(), xbinFin = ym.max(), xbinStep = dxBox)    
    xbinCenter, xMedian, xMean, xStd, yMedian, \
        yMean, yStd, nInBin, xPrc, yPrc = calc_running_stats(ym, xm, **kwargs_rs)
    #ax.plot(yMedian, xMedian, '-.r', label = 'Median in y')
    sig = 40 / np.sqrt(8. * np.log(2))
    xM = np.ma.masked_array(xMedian)
    yM = np.ma.masked_array(yMedian)
    m_gs = np.isnan(xM) | np.isnan(yM) 
    xS = gaussian_filter1d(xM[~m_gs], sig)
    yS = gaussian_filter1d(yM[~m_gs], sig)
    ax.plot(yS, xS, ls = '--', label = 'smoothed median in y')
  
    #axis = [7.5, 9.0, 7.5, 9.5]
    ax.set_xlim(7.5, 9.0)
    ax.set_ylim(7.5, 9.5) 
    ax.set_xlabel(r'$12\ +\ \log (O/H)$')    
    ax.set_ylabel(r'MPA-JHU')
    ax.legend(loc = 'upper left', fontsize = 14)
    ax.grid()
    f.tight_layout()
    f.savefig('logOH_ZnebMPA.png')
    plt.close(f)

    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # f = plt.figure() 
    # f.set_dpi(100)
    # f.set_size_inches(10, 8)    
    # ax = f.gca()
    # ax.scatter(logOH_M13, np.polyval(p[0], logOH_M13), marker = '.', s = 1, c = '0.5', edgecolor = 'none', alpha = 0.8, label = '')
    # ax.set_xlim(7.5, 9.0)
    # ax.set_ylim(7.5, 9.5) 
    # ax.set_xlabel(r'$12\ +\ \log (O/H)$')    
    # ax.set_ylabel(r'$12\ +\ \log (O/H)$ converted to MPA-JHU')    
    # plt.legend(loc = 'best')
    # plt.grid()
    # f.savefig('logOHCALIFAtoMPA.png')
    # plt.close(f)
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    