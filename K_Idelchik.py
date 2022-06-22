rho = 997
g = 9.81

def Kxin(k,wb,wc):
    
    # alpha = 90°, hb/hc = 1, up to wb/wc ~2
    
    if wb/wc <= 0.8:
        Ap = 1.
    else:
        Ap = 0.9
    
    zeta_cb = k*Ap*(0.34+(wb/wc)**2)
    
    return k*rho*zeta_cb

def Kyin(k,ws,wc): # p 282 Handbook of Hydraulic Resistance
    # assert ws/wc <= 1
    zeta_cs = 0.4*(1-(ws/wc)**2)

    return k*rho*zeta_cs

def Kxout(k,Qb,Qc,Fb,Fc): # P 266 Handbook of Hydraulic Resistance
    Fratio = Fb/Fc
    if Fratio >= 0 and Fc <= 0.2:
        A = 1.
    else: # linear regression from table 7-4
        A = -0.2256*Fratio+0.8299
    
    zeta_cb = A*(1+((Qb*Fc)/(Qc*Fb))**2-2*(1-(Qb/Qc)))

    return k*rho*zeta_cb

def Kyout(k,Qb,Qc):
    zeta_cs = 1.55*(Qb/Qc)-(Qb/Qc)**2

    return k*rho*zeta_cs 

#  possibilité de calculer un Kyout facteur de ws**2 et non de wc**2