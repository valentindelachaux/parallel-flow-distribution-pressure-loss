


import numpy as np

# set parallel parameters
Np = 18  # number of hydraulic paths (horizontal pipes)

# set pipe parameters
DCi = 0.0329 * np.ones(Np)  # [m] diameter of inlet manifold
DCo = 0.0329 * np.ones(Np)  # [m] diameter of outlet manifold
DR = 0.0073 * np.ones(Np)  # [m] diameter of fictitious row pipe
ACi = DCi ** 2 * np.pi / 4  # [m^2] flow area of main pipes supplying each row
ACo = DCo ** 2 * np.pi / 4  # [m^2] flow area of main pipes returning from each row
AR = DR ** 2 * np.pi / 4  # [m^2] flow area of horizontal pipe
LC = np.array([0.165] + [0.1215] * (Np - 1))  # [m] distance between horizontal pipes
LR = 5.8 * np.ones(Np)  # [m] length of horizontal pipes
ep = 0  # [m] surface roughness
frfac_correlation = 1  # 1=Blasius, 2=Colebrook, 3=Haaland. 4=Joseph&Jang

if Np != len(DCo) and Np != len(DCi) and Np != len(DR) and Np != len(LR):
    raise ValueError("Np is different from the specified number of rows!")

# initialization
it = 1
itmax = 40  # max number of iterations
itend = itmax
tolFlow = 1e-5  # tolerance for flow convergence
tolDp = 1e-4  # tolerance for pressure convergence
tolLastTee = 1e-4  # tolerance for mass conservation in the last tee junction
ReR = np.zeros(Np)  # [-] Reynolds number, R postscript refers to absorber pipe
fR = np.zeros(Np)  # [-] Darcy friction factor
YR = np.zeros(Np)  # [1/kg.m] resistance coefficient
ReCi = np.zeros(Np)  # C postscript refers to manifold segments between pipes
ReCo = np.zeros(Np)  # i/o refers to inlet/outlet manifold
fCi = np.zeros(Np)
fCo = np.zeros(Np)
YTinSi = np.zeros(Np)
YTinSt = np.zeros(Np)
ZTinSi = np.zeros(Np)
ZTinSt = np.zeros(Np)
YToutSi = np.zeros(Np)
YToutSt = np.zeros(Np)
ZToutSi = np.zeros(Np)
ZToutSiL = np.zeros(Np)
ZToutSt = np.zeros(Np)
YCi = np.zeros(Np)
YCo = np.zeros(Np)
YTot = np.zeros(Np)

# fluid properties
x = 40  # [%] glycol content (=0 for water)
Tin = 20  # [degC] fluid (inlet) temperature
rhoIn = densityGlyMixAndWat(x, Tin)  # [kg/m3] fluid density at inlet
nuIn = viscosityGlyMixAndWat(x, Tin)  # [m^2/s] fluid kin. viscosity at inlet
Vtot_m3h = 2.4  # [m3/h] total.

Vtot = Vtot_m3h / 3600  # [m3/s] total volume rate in input
Mtot = Vtot * rhoIn     # [kg/s] total mass rate in input

# set reasonable initial guess for flow rates
MoldR = Mtot / Np * np.ones(Np)  # [kg/s] mass flow rate per row (uniform distribution assumed)
MoldC = Mtot * np.ones(Np)     # MoldC(1)=Mtot (other cells will be overwritten)
for jj in range(1, Np):
    MoldC[jj] = MoldC[jj - 1] - MoldR[jj - 1]  # mass flow in the connecting pipes 
MnewC = MoldC.copy()  # creation of variable MnewC, so that the first cell is Mtot

# set right hand side (RHS) column vector
BB = np.concatenate(([Mtot], np.zeros(Np - 1)))  # RHS vector

while it <= itmax:
    plt.figure(1)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 20))))
    plt.figure(1)
    plt.plot(range(1, Np + 1), MoldR / Mtot * 100)
    plt.xlabel('Row #')
    plt.ylabel('% of total flow')
    plt.grid(True)
    plt.gca().set_prop_cycle(None)

    VoldR = MoldR / rhoIn  # [m3/s] volume flow rate in absorber pipes
    VoldC = MoldC / rhoIn  # [m3/s] volume flow rate in manifold segments
    
    for jj in range(Np):
        ReR[jj] = (DR[jj] / (nuIn * AR[jj])) * abs(MoldR[jj] / rhoIn)  # Re in abs. pipes [-]
        fR[jj] = FrictionFactorFunc(ReR[jj], DR[jj], ep, frfac_correlation)  # fr.f. in abs. pipes [-]
        YR[jj] = fR[jj] * LR[jj] / DR[jj] * 0.5 / rhoIn / (AR[jj] ** 2)  # resistance coefficients [1/kg.m]
        ReCi[jj] = (DCi[jj] / (nuIn * ACi[jj])) * abs(MoldC[jj] / rhoIn)  # Re in manifold in [-]
        ReCo[jj] = (DCo[jj] / (nuIn * ACo[jj])) * abs(MoldC[jj] / rhoIn)  # Re in manifold out [-]
        fCi[jj] = FrictionFactorFunc(ReCi[jj], DCi[jj], ep, frfac_correlation)  # fr.f. in manifold in [-]
        fCo[jj] = FrictionFactorFunc(ReCo[jj], DCo[jj], ep, frfac_correlation)  # fr.f. in manifold out [-]
        YTinSi[jj], ZTinSi[jj] = TeeDivSide(VoldR[jj], VoldC[jj], AR[jj], ACi[jj], ACi[jj], rhoIn, ReCi[jj])  # diverter=inlet (side)
        YTinSt[jj], ZTinSt[jj] = TeeDivSt(VoldR[jj], VoldC[jj], AR[jj], ACi[jj], ACi[jj], rhoIn, ReCi[jj])  # diverter=inlet (straight)
        YToutSi[jj], ZToutSi[jj], ZToutSiL[jj] = TeeConvSide(VoldR[jj], VoldC[jj], AR[jj], ACi[jj], ACi[jj], rhoIn, ReCo[jj])  # converter=out (side)
        YToutSt[jj], ZToutSt[jj] = TeeConvSt(VoldR[jj], VoldC[jj], AR[jj], ACi[jj], ACi[jj], rhoIn, ReCo[jj], ZToutSiL[jj])  # converter=out (straight)
        YCi[jj] = fCi[jj] * LC[jj] / DCi[jj] * 0.5 / rhoIn / (ACi[jj] ** 2)  # res.coef. in [1/kg.m]
        YCo[jj] = fCo[jj] * LC[jj] / DCo[jj] * 0.5 / rhoIn / (ACo[jj] ** 2)  # res.coef. out [1/kg.m]
        YTot[jj] = YR[jj]  # YT initialized as YR, as it includes at least the row Dp
        for ii in range(jj):
            YTot[jj] += YCi[ii] * MoldC[ii]**2 / MoldR[jj]**2 + YCo[ii] * MoldC[ii]**2 / MoldR[jj]**2  # supply and return pipes
            if ii < jj:
                YTot[jj] += YTinSt[ii] * MoldC[ii]**2 / MoldR[jj]**2  # T-in (div) straight, normalized for MC
                YTot[jj] += YToutSt[ii] * MoldC[ii]**2 / MoldR[jj]**2  # T-out (con) straight, normalized for MC
        YTot[jj] += YTinSi[jj] * MoldC[jj]**2 / MoldR[jj]**2  # add side passage of the inlet-tee
        YTot[jj] += YToutSi[jj] * MoldC[jj]**2 / MoldR[jj]**2  # add side passage of the outlet-tee

    AA = np.diag(-YTot * MoldR) + np.diag(YTot[:-1] * MoldR[:-1], -1)
    AA[0, :] = 1  # coefficient matrix
    MnewR = np.linalg.solve(AA, BB)  # find solution vector
    for jj in range(1, Np):
        MnewC[jj] = MnewC[jj - 1] - MnewR[jj - 1]

    # check convergence
    Dp = np.concatenate((MnewR[:-1] * np.diag(AA, -1), MnewR[-1] * (-AA[-1, -1])))
    emaxR = np.max(np.abs(MnewR - MoldR) / MnewR)
    emaxC = np.max(np.abs(MnewC - MoldC) / MnewC)
    emaxFlow = max(emaxC, emaxR)
    emaxDp = (np.max(Dp) - np.min(Dp)) / np.mean(Dp)

    print(f'\nIteration = {it:3d}      max error = {emaxFlow:8.2e}')
    print('MnewR [kg/s]  MoldR [kg/s]  MnewC [kg/s]  MoldC [kg/s]')
    for j in range(len(MoldR)):
        print(f'{MnewR[j]:8.2g}    {MoldR[j]:11.2g}  {MnewC[j]:11.2g}  {MoldC[j]:11.2g}')

    if emaxFlow < tolFlow and emaxDp < tolDp:
        itend = it
        break
    else:
        it += 1
        MoldR = (MnewR + MoldR) / 2
        for jj in range(1, Np):
            MoldC[jj] = MoldC[jj - 1] - MoldR[jj - 1]

# check on respect of mass conservation in last tee junction
if abs(MoldR[-1] - MoldC[-1]) / abs(MoldC[-1]) > tolLastTee:
    raise ValueError('Mass conservation is violated in the last tee junction')

# print max relative error and iteration count
print(f'\nNumber of iterations to convergence = {itend:3d}')
print(f'Max relative error at convergence =  {emaxFlow:8.2e}')

VR = MnewR / rhoIn  # [m3/s] volume flow rate in different rows
vR = VR / AR  # [m/s] average flow velocity in rows
VC = MnewC / rhoIn  # [m3/s] volume flow rate in connecting pipes
vCi = VC / ACi  # [m/s] average flow velocity conn. pipes (inlet)
vCo = VC / ACo  # [m/s] average flow velocity conn. pipes (outlet)
DpR = MnewR**2 * YR  # [Pa] pressure drop in row only
DpCi = MnewC**2 * YCi  # [Pa] Pressure drop in conn. pipe segment
DpCo = MnewC**2 * YCo  # [Pa] Pressure drop in conn. pipe segment

# print summary results
print('\nParallelFlow.py: Summary Results\n')
print('   Calculated Parameters for each Row:')
print('Pipe Mass Rate Flow Rate Velocity  Reynolds Frict.Loss')
print('      [kg/s]    [m^3/h]   [m/s]     [-]       [Pa]')
for n in range(len(MnewR)):
    print(f'{n+1:2d}  {MnewR[n]:7.2f}   {VR[n]*3600:7.3f}  {vR[n]:7.2f}  {ReR[n]:8.0f} {DpR[n]:9.0f}')
print('\n')
print('Fluid Properties:')
print(f' density [kg/m^3]:          {rhoIn:10.3e}')
print(f' kinem. viscosity [m^2/s]:  {nuIn:10.3e}')
print(f' inlet temperature [degC]:  {Tin:10.1f}')
print(f' glycol content [%%]:        {x:10.1f}')
print(f' total flow rate [kg/s]:    {Mtot:10.4f}')
print(f' total flow rate [m^3/h]:    {Vtot*3600:10.4f}')
print(f' Total Calculated Flow Rate [kg/s]:               {np.sum(MnewR):7.4f}')
print(f' Calculated DeltaP across Branched Section [kPa]: {Dp[0]/1000:7.2f}')
