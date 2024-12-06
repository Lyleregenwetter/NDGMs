# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:29:09 2022

@author: Lyle
"""


def getInvalid(df):
    df=df.copy()
    
    if not df.columns[0].startswith(" "):
        df.columns = [" " + string for string in list(df.columns)]
        
    invalid_bikes=[]
    for idx in df.index:    
        invalid=False
        if df.at[idx, " CS Length"]<=0:
            invalid=True
        if df.at[idx, " Stack"]<=0:
            invalid=True
        if df.at[idx, " Stack"]>=10:
            invalid=True
        if df.at[idx, " ST Length"]<=0:
            invalid=True
        if df.at[idx, " HT Length"]<=0:
            invalid=True
        if df.at[idx, " DT Length"]<=0:
            invalid=True
        if df.at[idx, " BB Length"]<=0:
            invalid=True
        if df.at[idx, " HT Angle"]<=0:
            invalid=True
        if df.at[idx, " SS E"]<=0:
            invalid=True
        if df.at[idx, " SS Z"]<=0:
            invalid=True
        if df.at[idx, " SSB Offset"]<=0:
            invalid=True
        if df.at[idx, " CSB Offset"]<=0:
            invalid=True
        if df.at[idx, " ST Angle"]<=0:
            invalid=True
        if df.at[idx, " BB OD"]<=0:
            invalid=True
        if df.at[idx, " TT OD"]<=0:
            invalid=True
        if df.at[idx, " HT OD"]<=0:
            invalid=True
        if df.at[idx, " DT OD"]<=0:
            invalid=True
        if df.at[idx, " CS OD"]<=0:
            invalid=True
        if df.at[idx, " SS OD"]<=0:
            invalid=True
        if df.at[idx, " ST OD"]<=0:
            invalid=True
        if df.at[idx, " HT UX"]<=0:
            invalid=True
        if df.at[idx, " HT LX"]<=0:
            invalid=True
        if df.at[idx, " ST UX"]<=0:
            invalid=True
        if df.at[idx, " ST Angle"]<=0:
            invalid=True
        if df.at[idx, " ST Angle"]>=180:
            invalid=True
        if df.at[idx, " HT Angle"]>=180:
            invalid=True
        if df.at[idx, " HT Angle"]<=0:
            invalid=True
        if df.at[idx, " CS F"]<=0:
            invalid=True
        if df.at[idx, " Dropout Offset"]<=0:
            invalid=True
        if " SSB_Include" in df.columns:
            if df.at[idx, " SSB_Include"]==1 and df.at[idx, " SSB OD"]<=0:
                invalid=True
            if df.at[idx, " CSB_Include"]==1 and df.at[idx, " CSB OD"]<=0:
                invalid=True
    #         print(idx)
    #         invalid=True
    #     if (df.at[idx, " ST Length"]+df.at[idx, " ST UX"])*np.sin(df.at[idx, " ST Angle"]/180*np.pi)>=1700:
    #         print(idx)
    #         invalid=True
        if (df.at[idx, " CS F"]>df.at[idx, " BB Length"])/2:
            invalid=True
        if df.at[idx, " SS OD"] + df.at[idx, " ST OD"] < 2*df.at[idx, " SS Z"]:
            invalid=True
        for tube in [" DT", " ST", " HT", " SS", " CS", " BB", " TT"]:
            if df.at[idx, tube + " OD"] < df.at[idx, tube + " Thickness"]:
                invalid=True
            if df.at[idx, tube + " Thickness"] < 0:
                invalid=True
    
            
        if invalid==True:
            invalid_bikes.append(idx)
        # if addspace==True:
        #     df.columns = [string[1:] for string in list(df.columns)]
    perc=1-len(invalid_bikes)/float(len(df.index))
    return invalid_bikes, perc