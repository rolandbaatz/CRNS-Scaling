
"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Authors:
Roland Baatz roland.baatz @ zalf.de
Patrick Davies patrickdavies94.pd@gmail.com

Maintainers and contact:
Currently maintained by the authors.

Copyright (C) Leibniz Centre for Agricultural Landscape Research (ZALF)
"""

from math import *
import numpy as np
import pandas as pd
import scipy.io as sio


def CV(x):
    return (x.std()/x.mean())*100

def Cal_func(Npih,No,bd):
    SWC = ((0.0808/((Npih/No)-0.372)-0.115)*bd)
    #SWC = (((bd*0.0808)/((Npih/No)'-0.372)) - 0.115*bd)'
    return SWC
def humidity_correction(Abs_h):
    Alpha = 0.0054
    cor_h = 1 +Alpha * (Abs_h - np.nanmean(Abs_h)) # humidity correction (Rosolem et al., 2013)
    return cor_h

def pressure_correction(P,beta):
    MA_L = 1/beta
    cor_p = np.exp((P - 1013.25)/MA_L)  # pressure correction (Desilets & Zreda, 2003)
    return cor_p


def Wd(d, r, bd, y):
    """Wd Weighting function to be applied on samples to calculate weighted impact of 
    soil samples based on depth.

    Parameters
    ----------
    d : float
        depth of sample (cm)
    r : float,int
        radial distance from sensor (m)
    bd : float
        bulk density (g/cm^3)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
            # Calculate the depth weighting for each layer
            df1['Wd'] = df1.apply(lambda row: Wd(
                row['DEPTH_AVG'], row['rscale'], bd, thetainitial), axis=1)
            df1['thetweight'] = df1['SWV'] * df1['Wd']


    """

    return(np.exp(-2*d/D86(r, bd, y)))


def D86(r, bd, y):
    """D86 Calculates the depth of sensor measurement (taken as the depth from which
    86% of neutrons originate)

    Parameters
    ----------
    r : float, int
        radial distance from sensor (m)
    bd : float
        bulk density (g/cm^3)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """

    return(1/bd*(8.321+0.14249*(0.96655+np.exp(-0.01*r))*(20+y)/(0.0429+y)))

def incoming_correction(N_ref,Inc,RCfactor):
    #cor_i = 1 + 1 * (150 / Inc - 1)
    fi = Inc/150
    Rccorr = -0.075*(RCfactor-4.49)+1
    ans = (fi-1)*Rccorr+1
    return ans**-1
    #return cor_i

def Absolute_conv(Rh,T):
    ab = (6.112 * np.exp((17.67*T)/(T+243.5)) * Rh * 2.1674)/(273.15 + T)
    return ab



def  Calculate_specific_month(DF_daily_joined,DF_hourly_joined):
    ## Read with hourly and daily array. 
    ## Now select bes suitable sets!
    years_with_data = list(set(DF_hourly_joined.index.year.to_list()))
    years_with_data.sort()
    #print(years_with_data)
    #years_with_data only used here.
    variance_by_month = {}
    print("Length variance_by_month=", str(len(variance_by_month)))
    #variance_by_month
    for yr in years_with_data:
        selected_year_tmp = DF_hourly_joined[DF_hourly_joined.index.year==yr]
        for mm in np.arange(1,13):
            selected_month_tmp = selected_year_tmp[selected_year_tmp.index.month == mm]
            #print(selected_month_tmp.columns)
            #print(selected_month_tmp.index.day)
            selected_month_tmp=selected_month_tmp.dropna(subset=['Pressure'])
            selected_month_tmp=selected_month_tmp.dropna(subset=['SOILM'])
            selected_month_tmp=selected_month_tmp.dropna(subset=['MOD'])
            daily_sample=selected_month_tmp.resample('24H').mean()
            #print(len(daily_sample), end=' ')
            if len(daily_sample) < 20:
                #here, first we checked ==0
                #print("Length does NOT suffice")
                #check for if at least 20 days with data are present within the month.
                pass
            else:
                #calculate variance
                #print("LEngth is fine")
                #print("selected_month_tmp")
                #print(selected_month_tmp)
                ans = CV(selected_month_tmp[['SOILM','Pressure']])
                variance_by_month[str(yr)+'-'+str(mm)+'-'+'12'] = ans.to_list()
    print("Length variance_by_month=", str(len(variance_by_month)))
    #print(variance_by_month)
    if len(variance_by_month)>=1:
        bw = (pd.DataFrame(variance_by_month).T).reset_index()
        bw.columns = ['Date','SOILM','Pressure']
        bw['Date'] = pd.to_datetime(bw.Date,format='%Y-%m-%d')
        bw = bw.set_index('Date')
        #bw = bw.dropna()
        #select where pressure is highest variable.
        aaaa = bw.sort_values(by='Pressure').iloc[-8:,:]
        aaaa.sort_values(by='SOILM')
        selected_yrs= aaaa.iloc[0,:].name.year
        selected_mths= aaaa.iloc[0,:].name.month
        #columns#['P-Po','In(N-No)'] each
        
        DF_hourly_joined_difference = DF_hourly_joined.diff(periods=-1)
        DF_hourly_joined_difference['ln_MOD'] = np.log(DF_hourly_joined_difference['MOD'])
        DF_hourly_joined_difference = DF_hourly_joined_difference[['Pressure','ln_MOD']]

        DF_hourly_joined_difference.columns = ['P-Po','In(N-No)']
        DF_hourly_joined_difference = DF_hourly_joined_difference.dropna(subset=['P-Po'])
        DF_hourly_joined_difference = DF_hourly_joined_difference.dropna(subset=['In(N-No)'])

        DF_hourly_joined_specific_mth = DF_hourly_joined_difference[(DF_hourly_joined_difference.index.year==selected_yrs) &
                                            (DF_hourly_joined_difference.index.month==selected_mths)]
        DF_hourly_joined_specific_mth= DF_hourly_joined_specific_mth.dropna()

        DF_daily_joined_specific_mth = DF_daily_joined[(DF_daily_joined.index.year==selected_yrs) &
                                            (DF_daily_joined.index.month==selected_mths)]
        DF_daily_joined_specific_mth= DF_daily_joined_specific_mth.dropna()
    else:
        #No months with more than 20 days of data exist:
        DF_daily_joined_specific_mth = pd.DataFrame(columns=['P-Po', 'In(N-No)'])
        DF_hourly_joined_specific_mth = pd.DataFrame(columns=['P-Po', 'In(N-No)'])
    return DF_hourly_joined_specific_mth, DF_daily_joined_specific_mth

def load_US_CRNS_data(num, us_path):
    p = us_path
    test = sio.loadmat(p+'COSMOS_'+num+'.mat',squeeze_me=True,struct_as_record=False)
    df = pd.DataFrame([test['Level1'].TIME,test['Level1'].PRESS,test['Level1'].RH,test['Level1'].TEM,test['Level1'].MOD]).T
    #Level 3 data
    SM_df = pd.DataFrame([test['Level3'].TIME,test['Level3'].SM12H,test['Level3'].SOILM]).T
    SM_df.columns = ['Date','SM12H','SOILM']
    SM_df['Date'] = pd.to_datetime(SM_df.Date-719529,unit='d').round('s')
    SM_df = SM_df.set_index('Date')
    SM_df = SM_df/100
    df.columns = ['Date','Pressure','RH','TEMP','MOD']
    df['Date'] = pd.to_datetime(df.Date-719529,unit='d').round('s')
    df = df.set_index('Date')
    df=df[(df.MOD>10)]
    DF = pd.concat([SM_df,df],axis=1)
    if num == '099':
        DF = DF.iloc[1:,:]
    return DF

def remove_some_time_stamps(ID,DF):
    ab = DF
    date_rm_after = pd.DataFrame({20:['2013-06-01'],49:['2014-08-01'],51:['2014-06-01'],
                     89:['2017-01-01'], 92:['2016-09-01'],1:['2012-05-01'],10:['2016-11-01'],84:['2015-12-01'],51:['2013-07-01'], 40:['2013-02'],14:['2019-01-01'],
                    33: ['2018-01-01'], 42: ['2018-01-01'],68: ['2013-05-01'], 110: ['2016-06-01'],53: ['2015-01-01']}).T #,47:['2017-01-01']

    #single_month = pd.DataFrame({31: [2013,3], 46: [2018,1], 56 :[2014,10],
                    #            58: [2015,4], 80: [2012,1], 82: [2015,10]}).T#, 51:[2016,6]
    single_month = pd.DataFrame({0:[0]}).T
    date_rm_before = pd.DataFrame({8:['2010-05-01'],11:['2010-10-01'],19: ['2012-06-01'],24: ['2017-01-01'], 46: ['2014-01-01'], 105: ['2015-01-01']}).T

    date_sel_range = pd.DataFrame({0:[0]}).T

    '''date_sel_range = pd.DataFrame({44:['2017-12-01','2018-02-28'],52: ['2015-06-01','2015-07-30'],
                     53: ['2013-06-01','2013-07-3'], 54:['2013-06-01','2013-07-30'],
                     57:['2015-06-01','2015-07-30'], 66: ['2012-12-01','2013-01-30'],
                     65:['2013-06-01','2013-07-30'], 
                     67: ['2018-12-01','2019-01-30'], 71:['2014-08-01','2014-09-30'],
                     73:['2016-01-01','2016-02-28'],76:['2013-01-01','2013-02-28'],
                     79:['2017-09-01','2017-09-30'],82:['2015-09-01','2015-10-30'],
                     87:['2015-09-01','2015-10-30'],93:['2015-09-01','2015-10-30'],
                    98:['2015-09-01','2015-10-30'],99:['2015-09-01','2015-10-30'],
                    101:['2015-09-01','2015-10-30'], 47: ['2014-01-01','2014-04-01']
                    ,111:['2019-07-01','2019-08-15'],12:['2013-06-01','2013-06-30'],
                    922: ['2017-07-01','2017-08-15'], 92:['2015-09-01','2015-10-30'],
                    108:['2019-01-01','2019-02-15'],13:['2013-06-01','2013-06-30'],
                    15: ['2013-06-01','2013-06-30'], 18: ['2013-06-01','2013-06-30'],
                    21: ['2013-06-01','2013-06-30'], 26: ['2013-06-01','2013-06-30']}).T #56:['2015-06-01','2015-07-30'],,102:['2019-04-01','2019-04-30']'''

    dates_ignore = pd.DataFrame({9:['2010-07-30','2010-08-20'],60:['2013-02-01','2013-06-01'],
                                16:['2015-01-01','2016-06-30'],20: ['2013-08-01','2013-12-30'],25:['2012-07-15','2013-05-01']
                                }).T#
    Remove_site = [27,68,105,110,102]

    #date_rm_after.iloc[:,0] = pd.to_datetime(date_rm_after.iloc[:,0], format='%Y-%m-%d')
    #date_rm_before.iloc[:,0] = pd.to_datetime(date_rm_before.iloc[:,0], format='%Y-%m-%d')
    #date_sel_range.iloc[:,0] = pd.to_datetime(date_sel_range.iloc[:,0], format='%Y-%m-%d')
    #dates_ignore.iloc[:,0] = pd.to_datetime(dates_ignore.iloc[:,0], format='%Y-%m-%d')
    if ID in date_rm_after.index:
        rm_date = date_rm_after.loc[ID,0]
        DF = DF[DF.index < rm_date]
        #axlist[i].set_title(ID+'')

    if ID in date_rm_before.index:
        rm_date = date_rm_before.loc[ID,0]
        DF = DF[DF.index > rm_date]


    if ID in date_sel_range.index:
        sel_range = date_sel_range.loc[ID,:]
        strt,end = sel_range.iloc[0], sel_range.iloc[1] 
        print(strt,end)
        DF = ab.loc[(ab.index >= strt) & (ab.index <= end)]
    
    if ID in single_month.index:
        rf = single_month.loc[ID,:]
        yr,m = rf.iloc[0], rf.iloc[1]
        DF = DF[(DF.index.year == yr) & (DF.index.month==m)]
    
    #if ID in dates_ignore.index:
    if ID in dates_ignore.index:
        sel_range = dates_ignore.loc[ID,:]
        strt,end = sel_range.iloc[0], sel_range.iloc[1]            
        DF = DF.loc[(DF.index < strt) | (DF.index > end)]
        if ID == 25:
            DF = DF.loc[(DF.index < '2012-07-15') | (DF.index > '2013-05-01')] 
            DF = DF.loc[(DF.index < '2011-07-15') | (DF.index > '2011-09-15')]
            
    #daily = DF.resample('D').mean()
    if ID == 19:
        DF = DF[DF.index > '2013-08-01']
    if ID == 73:
        DF = DF[DF.index < '2016-05-15']
    if ID == 80:
        DF = DF[DF.index > '2011-06-15']
    if ID == 82:
        DF = DF[DF.index < '2016-12-01']
    return DF

def load_EU_data(path,file):
    #print("Reading: ", file)
    ###Read in file and set timing
    dat = pd.read_csv(path+'raw_crns_and_meteorological_data/'+file+'.csv', na_values=['noData'])
    if file in ['ALC001','ALC002']:
        fr = dat['DateTime_utc'][dat['DateTime_utc'].str.contains(":")]
        for num in fr.index:
            dat.iloc[num,0] = fr[num].replace('.','-')
        fr = dat['DateTime_utc'][~dat['DateTime_utc'].str.contains(":")].astype(float)
        for num in fr.index:
            dat.iloc[num,0] = pd.to_datetime(fr[num],unit='d',origin='1900-01-01').round('s')
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'],dayfirst=True)
        dat['DateTime_utc'] = dat['DateTime_utc'].dt.strftime('%Y-%m-%d %H')
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] )
    elif file in ['AGCK003','FEC001','FSC001','SHC001','WAC001']:
        dat['DateTime_utc'] = pd.to_datetime(dat['DateTime_utc'], format='%Y-%m-%d %H:%M:%S%z', utc=True)#rm: utc=True
        #dat['DateTime_utc'] = dat['DateTime_utc'].dt.tz_localize('UTC')
        dat['DateTime_utc'] = dat['DateTime_utc'].dt.tz_localize(None)

        #blarm utc=True
    else:
        dat['DateTime_utc'] = dat['DateTime_utc'] .map(lambda x: x.replace('.','-'))
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] ,dayfirst=True)
        dat['DateTime_utc'] = dat['DateTime_utc'].dt.strftime('%Y-%m-%d %H')
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] )
    dat = dat.set_index('DateTime_utc')

    SM_df = pd.read_csv(path+'processed_crns_data_and_diagnostics/'+file+'.csv')
    if file in ['AGCK003']:
        SM_df['DateTime_utc'] = pd.to_datetime(SM_df['# Time'], format='%Y-%m-%d %H:%M:%S%z',utc=True)
        SM_df['DateTime_utc'] = SM_df['DateTime_utc'].dt.tz_localize(None)#rm: utc=True
    else:
        SM_df.DateTime_utc = SM_df['DateTime_utc'].map(lambda x: x[:-9])
        SM_df.DateTime_utc = pd.to_datetime(SM_df['DateTime_utc'] )
        SM_df['DateTime_utc'] = SM_df['DateTime_utc'].dt.strftime('%Y-%m-%d %H')
        SM_df['DateTime_utc']  = pd.to_datetime(SM_df['DateTime_utc'] )
    SM_df = SM_df.set_index('DateTime_utc')
    #print(dat.columns)
    SM_df = SM_df['SoilMoisture_volumetric_MovAvg24h']
    SM_df.name = 'SOILM'
    colName1 = dat.columns[dat.columns.str.contains(pat = 'NeutronCount_Epithermal_Cum1')].values[0]
    dat.index.name = 'Date'
    colName2 = dat.columns[dat.columns.str.contains(pat = 'AirPressure')].values[0]
    colName3 = dat.columns[dat.columns.str.contains(pat = 'AirTemperature')].values[0]
    colName4 = dat.columns[dat.columns.str.contains(pat = 'AirHumidity')].values[0]
        
    # Check if 'NeutronCount_Slow_Cum1h' column exists
    if 'NeutronCount_Slow_Cum1h' in dat.columns:
        colName5 = dat.columns[dat.columns.str.contains(pat = 'NeutronCount_Slow_Cum1h')].values[0] 
        df = dat[[colName1, colName2, colName3, colName4, colName5]]
        df.columns = ['MOD', 'Pressure', 'TEMP', 'RH', 'NeutronCount_Slow_Cum1h']
    else:
        dat['NeutronCount_Slow_Cum1h'] = pd.NA
        df = dat[[colName1, colName2, colName3, colName4, 'NeutronCount_Slow_Cum1h']]
        df.columns = ['MOD', 'Pressure', 'TEMP', 'RH', 'NeutronCount_Slow_Cum1h']
    #df = dat[[colName1,colName2,colName3,colName4]]
    #df.columns = ['MOD','Pressure','TEMP','RH']
    # Fill NA values with a placeholder (e.g., np.nan) before converting to float
    df = df.replace('noData', np.nan)
    df = df.replace('NoData', np.nan)
    df = df.fillna(np.nan).astype(float)
    df = df.astype(float)
    DF = df.join(SM_df)
    #
    #if file in ['ALC001']:
    #    DF = DF.loc[(DF.index < '2012-07-15') | (DF.index > '2013-05-01')]

    return DF


def load_EU_data_b(path,file):
    #print("Reading: ", file)
    dat = pd.read_csv(path+'raw_crns_and_meteorological_data/'+file+'.csv', na_values=['noData'])
    if file in ['ALC001','ALC002']:
        fr = dat['DateTime_utc'][dat['DateTime_utc'].str.contains(":")]
        for num in fr.index:
            dat.iloc[num,0] = fr[num].replace('.','-')
        fr = dat['DateTime_utc'][~dat['DateTime_utc'].str.contains(":")].astype(float)
        for num in fr.index:
            dat.iloc[num,0] = pd.to_datetime(fr[num],unit='d',origin='1900-01-01').round('s')
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'],dayfirst=True)
        dat['DateTime_utc'] = dat['DateTime_utc'].dt.strftime('%Y-%m-%d %H')
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] )
    elif file in ['AGCK003','FEC001','FSC001','SHC001','WAC001']:
        dat['DateTime_utc'] = pd.to_datetime(dat['DateTime_utc'], format='%Y-%m-%d %H:%M:%S%z', utc=True)#rm: utc=True
        #dat['DateTime_utc'] = dat['DateTime_utc'].dt.tz_localize('UTC')
        dat['DateTime_utc'] = dat['DateTime_utc'].dt.tz_localize(None)

        #blarm utc=True
    else:
        dat['DateTime_utc'] = dat['DateTime_utc'] .map(lambda x: x.replace('.','-'))
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] ,dayfirst=True)
        dat['DateTime_utc'] = dat['DateTime_utc'].dt.strftime('%Y-%m-%d %H')
        dat['DateTime_utc']  = pd.to_datetime(dat['DateTime_utc'] )
    dat = dat.set_index('DateTime_utc')

    SM_df = pd.read_csv(path+'processed_crns_data_and_diagnostics/'+file+'.csv')
    if file in ['AGCK003']:
        SM_df['DateTime_utc'] = pd.to_datetime(SM_df['# Time'], format='%Y-%m-%d %H:%M:%S%z',utc=True)
        SM_df['DateTime_utc'] = SM_df['DateTime_utc'].dt.tz_localize(None)#rm: utc=True
    else:
        SM_df.DateTime_utc = SM_df['DateTime_utc'].map(lambda x: x[:-9])
        SM_df.DateTime_utc = pd.to_datetime(SM_df['DateTime_utc'] )
        SM_df['DateTime_utc'] = SM_df['DateTime_utc'].dt.strftime('%Y-%m-%d %H')
        SM_df['DateTime_utc']  = pd.to_datetime(SM_df['DateTime_utc'] )
    SM_df = SM_df.set_index('DateTime_utc')

    SM_df = SM_df['SoilMoisture_volumetric_MovAvg24h']
    SM_df.name = 'SOILM'
    colName1 = dat.columns[dat.columns.str.contains(pat = 'NeutronCount_Epithermal_Cum1')].values[0]
    dat.index.name = 'Date'
    colName2 = dat.columns[dat.columns.str.contains(pat = 'AirPressure')].values[0]
    colName3 = dat.columns[dat.columns.str.contains(pat = 'AirTemperature')].values[0]
    colName4 = dat.columns[dat.columns.str.contains(pat = 'AirHumidity')].values[0]
    df = dat[[colName1,colName2,colName3,colName4]]
    df.columns = ['MOD','Pressure','TEMP','RH']
    df = df.replace('noData', np.nan)
    df = df.replace('NoData', np.nan)
    df = df.astype(float)
    DF = df.join(SM_df)
    #
    #if file in ['ALC001']:
    #    DF = DF.loc[(DF.index < '2012-07-15') | (DF.index > '2013-05-01')]

    return DF



def beta_e(p,r_c,lat):
    """ 
    Calculate correction barometric pressure coefficient according to McJannet and Desilets (2024)
    Args:
        p (float): mean air pressure of the site
        r_c (float): cutoff rigidity of the site
        lat (float): latitude of the site
    Returns:
        several site specific variables, aslo beta and effective beta
    """
    g_ref = 9.860665
    rho_rck = 2670
    
    z = -0.00000448211*p**3 + 0.0160234*p**2 - 27.0977*p+ 15666.1
    # latitude correction
    g_lat = 978032.7*(1 + 0.0053024*(sin(radians(lat))**2)-0.0000058*(sin(radians(2*lat)))**2)
    
    # free air correction
    del_free_air=-0.3087691*z
    
    # Bouguer correction
    del_boug=rho_rck*z*0.00004193
    
    g_corr = (g_lat + del_free_air + del_boug)/100000
    
    # based on reference pressure
    x0 =  1033 # depth, g cm-2
    
    # final gravity and depth
    g = g_corr/10
    x = p/g 
    # --- latitude scaling ---

    # parameters
    alpha_lat = 9.694
    k_lat = 0.9954
    
    #calculation
    f_lat = 1/(1-exp(-alpha_lat*r_c**(-k_lat)))
    
    # --- elevation scaling ---
    
    # parameters 
    n_1 = 0.01231386
    alpha_1 = 0.0554611
    k_1 = 0.6012159
    b0 = 4.74235E-06
    b1 = -9.66624E-07
    b2 = 1.42783E-09
    b3 = -3.70478E-09
    b4 = 1.27739E-09
    b5 = 3.58814E-11
    b6 = -3.146E-15
    b7 = -3.5528E-13
    b8 = -4.29191E-14
    
    # calculate beta at a point
    term1 = n_1*(1+exp(-alpha_1*r_c**k_1))**-1
    term2 = (b0+b1*r_c+b2*r_c**2)*(x**1)
    term3 = (b3+b4*r_c+b5*r_c**2)*(x**2)
    term4 = (b6+b7*r_c+b8*r_c**2)*(x**3)
    
    beta = -(term1 + term2 + term3 + term4)
    
    # calculate effective beta
    term1 = n_1*(1+exp(-alpha_1*r_c**k_1))**-1*(x-x0)
    term2 = 0.5*(b0+b1*r_c+b2*r_c**2)*(x**2-x0**2)
    term3 = 0.3333*(b3+b4*r_c+b5*r_c**2)*(x**3-x0**3)
    term4 = 0.25*(b6+b7*r_c+b8*r_c**2)*(x**4-x0**4)
    
    beta_eff = (term1 + term2 + term3 + term4)/(x0-x)
    
    f_bar = exp((x0-x)*beta_eff)
    
    F_scale = f_lat * f_bar
    
    
    #Print output
    return g, x, f_lat, f_bar, F_scale, beta_eff, beta
print('Not a file to run, only methods included here.')