#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:48:45 2024

@author: pdavies
"""

import pandas as pd
import numpy as np
import CRNS_lib as CN_lib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import Darin_beta as DB
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
from scipy.ndimage import uniform_filter1d

path = '/home/pdavies/Documents/KNUST/PhD_work/Manuscript/Roland_Manuscript_2/COSMOS_Europe_Data/'
plot_paths = '/home/pdavies/Documents/KNUST/PhD_work/Manuscript/Roland_Manuscript_2/US_barometric_coeff/Current_analysis/Plots/'
site_info = pd.read_csv('Study_sites.csv')
Us_sites = site_info.iloc[12:,:]
Eu_sites = site_info.iloc[:12,:]

Jung = pd.read_csv('JUNG_Data_UTC.TXT',sep=',')
Jung['Date'] = pd.to_datetime(Jung.Date)
Jung = Jung.set_index('Date')
add_d = pd.read_csv('add.csv')
add_d.columns = ['ID', 'Site Name', 'Lat', 'Lon', 'Altitude', 'GV']
site_info = pd.concat([site_info,add_d],axis=0)
sites = site_info.ID.to_list()
#sites.reverse()


def pro(dat):
    dat['DateTime_utc'] = dat['DateTime_utc'] .map(lambda x: x.replace('.','-'))
    dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'], dayfirst=True)
    dat['DateTime_utc'] = dat['DateTime_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
    dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] )
    dat = dat.set_index('DateTime_utc')
    return dat

figu,axs = plt.subplots(6,5,figsize=(24,20),dpi=100)
axs = axs.flatten()
#fig2,axs2 = plt.subplots(4,3,figsize=(14,8),dpi=150)
#axs2 = axs2.flatten()

fig3,axs3 = plt.subplots(7,5,figsize=(27,13),dpi=100)
axs3 = axs3.flatten()
#add = ['11','3','23','78','73','CAC001','CRC001','AGCK003','OLC001','ALC002']

Results,avg_results, mon_results, Mon_contrib,cor_var= {},{},{},{},{}
for n, stn in enumerate(sites[:]):

    info = site_info[site_info.ID==stn]
    GV = info.GV.values
    if len(stn) > 2:
        name = info['Site Name'].values[0][:-4]
        #fig,ax = plt.subplots(1,1,figsize=(12,5),dpi=150)
        num = stn
        DF= CN_lib.load_EU_data(path,stn)
        dat1 = pro(pd.read_csv(path+'processed_crns_data_and_diagnostics/'+stn+'.csv'))
       # dat2 = pro(pd.read_csv(path+'calibration_raw_data/'+stn[:3]+'D'+stn[3:]+'.csv'))
        b = (dat1['NeutronCount_Epithermal_MovAvg24h_corrected']).resample('D').mean()
    else:
        name = info['Site Name'].values[0]
        num=str(stn).zfill(3)
        DF = CN_lib.load_US_CRNS_data(num)
        DF = CN_lib.remove_some_time_stamps(stn,DF)
        
    DF['MOD_cleaned'] = DF.MOD
    # apply max min threshold
    DF.loc[DF['MOD_cleaned'] < 50, 'MOD_cleaned'] = np.nan
    DF.loc[DF['MOD_cleaned'] >6000, 'MOD_cleaned'] = np.nan
    #Rol.loc[Rol[N_col+'_cleaned'] > max_thres, N_col+'_cleaned'] = np.nan
    
   # DF = DF[(DF.RH>=0)&(DF.RH<=100)]
    #DF = DF[(DF.MOD>40)&(DF.MOD<6000)]
    #RB added next line
   # DF = DF[(DF.Pressure>=700)&(DF.Pressure<=2000)]
    DF['MOD_24'] = DF.MOD.rolling(24, center=True, min_periods=12).median()
    DF['MOD_unc'] = DF.MOD.rolling(24, center=True, min_periods=12).sum()**.5
    DF['MOD_min'] = DF['MOD_24'] - DF['MOD_unc']
    DF['MOD_max'] = DF['MOD_24'] + DF['MOD_unc']

    # apply uncertainty thresholds
    # apply uncertainty thresholds

    DF.loc[DF['MOD_min'] > DF.MOD,'MOD_cleaned'] = np.nan
    DF.loc[DF['MOD_max'] < DF.MOD,'MOD_cleaned'] = np.nan
    DF = DF.resample('24H').mean()
    #DF = DF.rolling(window=24,min_periods=8).mean()
    DF = DF.join(Jung)
    DF['Abs_h'] = CN_lib.Absolute_conv(DF.RH, DF.TEMP)
    #ho = DF[DF.index == dat2.index[0]].Abs_h.values[0]

    P_cor = CN_lib.pressure_correction(DF.Pressure,0.0076) 
    H_cor = CN_lib.humidity_correction(DF.Abs_h)
    I_cor = CN_lib.incoming_correction(DF['RCORR_E'].mean(), DF['RCORR_E'],GV)
    DF['Np'] = DF.MOD*P_cor
    DF['Nh'] = DF.MOD*H_cor
    DF['Nhi'] = DF.MOD_cleaned*(H_cor*I_cor)
    DF['Nph'] = DF.MOD*P_cor*H_cor
    DF['Nphi'] = DF.MOD_cleaned*(P_cor*I_cor*H_cor)
    
    A_cor = DF['Nphi'].rolling(window=24,min_periods=10).median()#DF['Nphi'].resample('D').mean()
    #C_un = DF.MOD.resample('D').mean()
    if n <25:
        DF['MOD_difference'] = abs(DF['MOD_cleaned'].pct_change(periods=-1)) # difference in neutron count
        DF = DF[DF['MOD_difference'] <= 0.1]
    else:
        DF['MOD_difference_f1'] = abs(DF.MOD_cleaned.pct_change(periods=-1)) # difference in neutron count
        DF['MOD_difference_b1'] = abs(DF.MOD_cleaned.pct_change())
        
        DF['MOD_difference_f2'] = abs(DF.MOD_cleaned.pct_change(periods=-2)) # difference in neutron count
        DF['MOD_difference_b2'] = abs(DF.MOD_cleaned.pct_change(2))       
        #print(DF[['MOD_difference_b','MOD_cleaned','MOD_difference']])
 
        #actual_L = len(DF)
        DF = DF[(DF['MOD_difference_f1'] <= 0.10)&(DF['MOD_difference_b1'] <= 0.10)&(DF['MOD_difference_f2'] <= 0.10)&(DF['MOD_difference_b2'] <= 0.05)]
        #print(((actual_L-len(DF))/actual_L)*100)


    DF=DF.dropna(subset=['RH'])
    DF=DF.dropna(subset=['Nhi'])
    DF=DF.dropna(subset=['Pressure'])
    
    DF_analysis = DF[['Pressure','Nhi']].resample('24H').median()
    
    #Calculate ln for MOD and the difference. Do not average by ln

    DF_analysis['ln_MOD'] =  np.log(DF_analysis['Nhi'])
    DF_analysis=DF_analysis[['Pressure','ln_MOD']]
    DF_analysis = DF_analysis.diff(periods=-1)
    DF_analysis.columns = ['P-Po','In(N-No)']
    DF_analysis = DF_analysis.dropna(subset=['P-Po'])
    DF_analysis = DF_analysis.dropna(subset=['In(N-No)'])

    yr_list = DF_analysis.index.year.drop_duplicates().values
    count = 0
    
    out,temp_store = {},{}
    
    for yr in yr_list:
        temp_file,temp_stat = {}, {}
        for mnth in np.arange(1,13):
            try:
                tmp_dat = DF_analysis[(DF_analysis.index.year==yr)&(DF_analysis.index.month==mnth)]
                TR= (DF[(DF.index.year==yr)&(DF.index.month==mnth)])[['Abs_h','SOILM','Pressure']]
                SM,ABS = np.nanmean(TR.SOILM.values),np.nanmean(TR.Abs_h.values)
                tmp_dat = tmp_dat.dropna()
                P = np.nanmean(TR.Pressure)
                
                if len(tmp_dat) <15:
                    temp_store[str(yr)+'-'+str(mnth)] = [np.nan,np.nan,np.nan,np.nan]
                    continue
                else:
                    if stn in ['Wildacker','Iowa Validation Site']:
                        tmp_dat = tmp_dat[tmp_dat['P-Po'] >=-9.9]
                    slope_all, intercept_all, r_value_all, pv_all, std_all = stats.linregress(tmp_dat['P-Po'],
                                                                          tmp_dat['In(N-No)'])
                    temp_store[str(yr)+'-'+str(mnth)] = [round(slope_all,8),abs(round(r_value_all,2)),ABS,SM]
                    
                    if r_value_all <= -0.9:
                        # Darin's Method
                        
                        g, x, f_lat, f_bar, F_scale, beta_eff, beta = DB.beta_e(P,GV,info['Lat'].values[0])
                        
                        #print('Good')
                        if (slope_all >-0.02) & (slope_all<-0.0044):
                            #print(round(slope_all,5))
                            out[str(yr)+'-'+str(mnth)] = [str(mnth),round(slope_all,7),round(beta_eff[0],7),abs(round(r_value_all,2)),ABS,SM]
                            dt1 = DF_analysis[(DF_analysis.index.year == yr)&(DF_analysis.index.month == mnth)]
                            if stn in ['Wildacker','Iowa Validation Site']:
                                dt1 = dt1[dt1['P-Po'] >=-10]
                            else:
                                dt1 = dt1
                            
                        else:
                            continue
                        if count == 0:
                            at =dt1
                            #at1 =dt1b
                        else:
                            at = pd.concat([at,dt1],axis=0)
                        count = count+1
                    
                    else:
                        continue
                       # print('correlation < 0.9')
                        
            except:
                continue
    tp_file = pd.DataFrame(temp_store)

    out = pd.DataFrame(out,index=['Month','Beta_avg','Beta_Darin','r','Absolute_Humidity','SM']).T
    out.index = pd.to_datetime(out.index)
    #abs_df = out[['Absolute_Humidity','Beta_avg']].dropna()
    #sm_df = out[['SM','Beta_avg']].dropna()
    #r1,p1 = stats.pearsonr(abs_df.Beta_avg,abs_df.Absolute_Humidity)
    #r2,p2 = stats.pearsonr(sm_df.Beta_avg,sm_df.SM)
    #cor_var[name]= [r1,p1,r2,p2]
    out['Month'] = out.Month.astype(int)
    wint = out[(out.Month>=12) |(out.Month<3)]
    spring = out[(out.Month>=3) &(out.Month<6)]
    summer = out[(out.Month>=6) &(out.Month<9)]
    fall = out[(out.Month>=9) &(out.Month<12)]
    Mon_contrib[name] = [len(wint),len(spring),len(summer),len(fall)]
    print(name,len(wint),len(spring),len(summer),len(fall),len(out))
    a = pd.DataFrame(temp_store,index=['Beta','Corr','Absolute_Humidity','SM']).T
    a.index = pd.to_datetime(a.index)
    ab = a[(a.Corr>=0.9)&(a.Beta>-0.01)&(a.Beta<-0.001)]
    poor_r = a[(a.Corr<0.9)&(a.Beta>-0.01)&(a.Beta<0.001)]
    mon_results[num] = ab
    ct =np.nanmedian(out.Beta_avg)
    Beta_d = np.nanmedian(out.Beta_Darin)
    st =np.nanstd(ab.Beta)
    axs3[n].plot(poor_r.index, poor_r.Beta*100,marker='o',linestyle=' ', color = 'k',linewidth=1) 
    axs3[n].plot(ab.index, ab.Beta*100,marker='^', color = 'r',markersize=7,linestyle=' ')
    y_smooth = pd.DataFrame(ab.Beta*100,index=ab.index)
    y_smooth =y_smooth.rolling(window=3,min_periods=2).mean()
    axs3[n].plot(y_smooth.index, y_smooth.values, color = 'r',linestyle='-')
    axs3[n].set_title(info['Site Name'].values[0],fontsize=19)
    #axs1[countit-1].set_xticklabels(rotation=45)
    axs3[n].xaxis.set_tick_params(labelsize=16,rotation=20)
    axs3[3].yaxis.set_tick_params(labelsize=16)
    #loc = mticker.MultipleLocator(base=2.0) # this locato
   
    ####################################################################################################################
    #########                Regression Plots   #######################################################################
    ###################################################################################################################
    if name in ['Wildacker','Iowa Validation Site','Glensaugh']:
        at = at[at['P-Po']>-10]
        if name == 'Glensaugh':
            at = at[at['P-Po']<10]
    elif name == 'UMBS':
        at = at[at['P-Po']<10]
    else:
        at =at
    slope_all, intercept_all, r_value_all, pv_all, std_all = stats.linregress(at['P-Po'], at['In(N-No)'])
    sns.regplot(x='P-Po',y='In(N-No)',data=at,ci=None,ax=axs[n],scatter_kws={'alpha':0.7,'color':'b'},line_kws={'color':'r'})
    axs[n].text(0.4,0.9,r'$\beta_{avg}:$ '+str(round(ct,4)*100)[:5]+' %hPa',fontsize=16,transform=axs[n].transAxes)
    axs[n].text(0.4,0.8,r'$\beta_{reg}:$ '+str(round(slope_all,4)*100)[:5] +' %hPa',fontsize=16,transform=axs[n].transAxes)
    axs[n].text(0.4,0.7,r'$\beta_{Des}:$ '+str(round(Beta_d,4)*100)[:5] +' %hPa',fontsize=16,transform=axs[n].transAxes)
    avg_results[num] = [name,info['Lat'].values[0],info['Lon'].values[0],info.Altitude.values[0],
                                      GV[0],ct*100,st*100,slope_all*100,std_all*100,r_value_all,Beta_d*100]
    
    
    axs[n].set_title(info['Site Name'].values[0],fontsize=16)#+r' $\beta$: '+ str(round(ct,6)*-100)[:5]+' %hPa' ,fontsize=12)
    axs[n].set_ylim(at['In(N-No)'].min()-0.05,at['In(N-No)'].max()+0.05)
    #axs[n].set_xlim(at['P-Po'].min()-3,at['P-Po'].max()+3)
    axs[n].xaxis.set_tick_params(labelsize=18)
    axs[n].yaxis.set_tick_params(labelsize=18)

    if n in [0,5,10,15,20]:
        axs[n].set_ylabel('In(N-No)',fontsize=18)
    else:
        axs[n].set_ylabel(' ')
    
    if n >=25:
        axs[n].set_xlabel('P-Po',fontsize=18)
    else:
        axs[n].set_xlabel(' ')
        
    a = DF[['Nphi']].resample('D').mean()
    if a.max().values[0] >6000:
        r=DF
        #print(stn)
        #print(a[a.Nphi>6000])
    #b = dat.NeutronCount_Epithermal_Cum1h_corrected.resample('D').mean()
    
    
    #if len(stn) >2 :
        #C_un.plot(ax=axs2[n],color='r',label='Uncorrected')
        #A_cor.plot(ax=axs2[n],color='b',label='P. corrected')
        #b.plot(ax=axs2[n],color='k',label='C. corrected')
        
        
        #axs2[n].set_title(name)
        
        
   # else: 
    #    pass
#ax.legend()
figu.tight_layout()
#fig2.tight_layout()
fig3.tight_layout()
#fig2.savefig('verif.jpeg')
figu.savefig('Barometric_ccf.jpeg')
cols = ['Site Name','Lat','Lon','Altitude','GV','Beta_sp','sp_Error','Beta_reg','Error_reg','corr_reg','Beta_Darin']
RESULT = pd.DataFrame(avg_results,index = cols).T 
RESULT.to_csv('CRNS_beta.csv')
   
# import proplot as pplt
# import xarray as xr
# RC = xr.open_dataset('/home/pdavies/Documents/KNUST/PhD_work/Manuscript/Roland_Manuscript_2/US_barometric_coeff/Cut-off_Rigidity_2020.nc')
# RC = RC['Cut-off Ridity'].T

# Us_wc = RC.sel(lat=slice(20,55),lon=slice(-140,-50))
# Eu_wc = RC.sel(lat=slice(40,65),lon=slice(-20,30))
# US = RESULT.iloc[12:,:]
# EU = RESULT.iloc[:12,:]
# # Zooming in with cartopy degree-minute-second labels
# pplt.rc.reso = 'hi'
# fig = pplt.figure(refwidth=5)

# ax = fig.subplot(121, proj='pcarree')

# ax.format(lonlim=(-125, -65), latlim=(20, 55))
# cb = ax.scatter(x=US.Lon.astype(float),y=US.Lat.astype(float),c=US.Beta_sp.astype(float),vmin=-0.76,vmax=-0.54,s=90,
#                 cmap='jet',edgecolors='black', transform=pplt.PlateCarree())
# cv=ax.contour(Us_wc.lon,Us_wc.lat,Us_wc,colors=['grey'],transform=pplt.PlateCarree())
# ax.clabel(cv, colors = 'k',fontsize=15, inline=1)
# ax = fig.subplot(122, proj='cyl')
# ax.format(lonlim=(-10, 20), latlim=(40, 63))
# cb =ax.scatter(x=EU.Lon.astype(float),y=EU.Lat.astype(float),c=EU.Beta_sp.astype(float),vmin=-0.79,vmax=-0.64,s=90,
#                 cmap='jet',edgecolors='black', transform=pplt.PlateCarree())
# cv=ax.contour(Eu_wc.lon,Eu_wc.lat,Eu_wc,colors=['grey'],transform=pplt.PlateCarree())
# #ax.clabel(cv,inline=1,inline_spacing=0,fontsize=10,fmt='%1.0f',colors='b')
# ax.clabel(cv, colors = 'k',fontsize=15,inline=1)
# fig.format(
#     coast=True, labels=True,
#     borders=True, borderscolor='black',coastcolor='black',coastlinewidth=0.8,
#     abc='a.'
# )
# fig.colorbar( cb, loc='b', label='Barometric Coefficient [%hPa]',length=0.6,
#                 tickminor=False, extendsize='1.7em')


# pplt.rc.reset()
# #fig.savefig('Geobeta.jpeg',dpi=300)

# #####################################################################################################
# ############# Months contributions of beta   #################################
# Mon_con = pd.DataFrame(Mon_contrib,index=['Winter', 'Spring','Summer','Fall']).T
# fig, axs = pplt.subplots(nrows=2,ncols=2, refaspect=3, figwidth=12)
# axs.format(
#     xmargin=0, xlabel='Sites', ylabel='Correlation', grid=True,
#     suptitle='Humidity and Solar Activity Influence on Neutron Count',
# )
# for i,col in enumerate(['Winter', 'Spring','Summer','Fall']):
#     axs[i].bar(Mon_con[col], cycle='Blues', edgecolor='blue')
#     axs[i].format(xrotation=60,fontsize=15)
# axs[0].legend(loc='t')

# ############################################################################################
# ####################  Beta verse Absolute humidity and Soil moisture

# beta_cor = pd.DataFrame(cor_var,index=['Abs_cor','Abs_p','SM_cor','SM_p']).T  
# fig, axs = pplt.subplots(nrows=1, refaspect=3, figwidth=12)
# axs.format(
#     xmargin=0, xlabel='Sites', ylabel='Correlation', grid=True,
#     suptitle='Humidity and Solar Activity Influence on Beta',
# )

# axs.bar(beta_cor[['Abs_cor','SM_cor']], cycle='Blues', edgecolor='blue9'
#         )
        
# axs.format(xlocator=1, xminorlocator=0.5, ytickminor=False)
# #axs.format(title=var)

# axs.format(xrotation=60,fontsize=13)
# axs.legend([r'Absolute Humidity', r'SM$'],loc='t')
        
    
fig, axes =plt.subplots(2,1,figsize=(8,6),dpi=100,sharex=True)
ax = axes.flatten()

data = RESULT[['Lat', 'Lon', 'Altitude', 'GV', 'Beta_reg' ,'Beta_Darin']].astype(float)
data['Beta_reg'] = abs(data.Beta_reg)
data['Beta_Darin'] = abs(data.Beta_Darin)
sns.regplot(ax =ax[0],x='GV',y='Beta_reg',data=data,ci=None, scatter_kws={"color": "black"}, line_kws={"color": "red"})
cb = ax[0].scatter(data.GV, data.Beta_reg, s=50, c=data['Altitude'],cmap='terrain')
sns.regplot(ax =ax[1],x='GV',y='Beta_Darin',data=data,ci=None, scatter_kws={"color": "black"}, line_kws={"color": "red"})
cb = ax[1].scatter(data.GV, data.Beta_Darin, s=50, c=data['Altitude'],cmap='terrain')

for i, mt in enumerate(['Davies P','Desilet D']):
    
    ax[i].set_ylabel(r'$\beta$ coefficient [%hPa]',fontsize=10)
    ax[i].set_title(mt)
fig.tight_layout()
fig.colorbar(cb, cax = plt.axes([0.67, 0.937, 0.297, 0.02]),orientation='horizontal', label='Altitude')
