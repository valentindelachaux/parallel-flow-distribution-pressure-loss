import os
import sys
import math
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import openpyxl as opxl
from openpyxl.utils.dataframe import dataframe_to_rows

from CoolProp.CoolProp import PropsSI
import fluids as fds

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'RD-systems-and-test-benches')))

import utils.data_processing as dp

def find_fluid(fluid):

    if type(fluid) == str:
        return fluid
    
    elif type(fluid) == dict:
        fluid_name = fluid.get('name','')
        glycol_rate = fluid.get('glycol_rate',0)

        if fluid_name == 'MPG' or fluid_name == 'MEG':
            fluid = f'INCOMP::{fluid_name}[{glycol_rate}]'
        else:
            fluid = fluid_name

        return fluid
    
    else:
        raise TypeError("Fluid must be a string or a dictionary")


class hx_harp:

    def __init__(self,par):
        
        self.config = par["config"]

        if self.config == "hx":

            self.N_panel = par["N_panel"]
            self.N_EP = par["N_EP"]
            self.N_riser_per_panel = par["N_riser_per_panel"]
            self.N_riser_per_EP = int(self.N_riser_per_panel/self.N_EP)

            self.inter_riser = par["inter_riser"]
            self.inter_EP = par["inter_EP"]
            self.inter_panel = par["inter_panel"]

            self.update_geometry()

        else:

            self.N_panel = par["N_panel"]
            self.inter_panel = par["inter_panel"]
            self.N = self.N_panel
            self.Ly = (self.N_panel-1)*[self.inter_panel]

        self.regular_PL_calculation_method = par.get("regular_PL_calculation_method", "friction_factor")
        self.a_x = par["a_x"]
        self.b_x = par["b_x"]

        self.riser = duct(par["shape_riser"],par["D_riser"],par["h_riser"],par["w_riser"],par["L_riser"])
        self.man_in = duct(par["shape_man"],par["D_in"],par["h_man"],par["w_man"],par["L_man"])
        self.man_out = duct(par["shape_man"],par["D_out"],par["h_man"],par["w_man"],par["L_man"])
        self.man = duct(par["shape_man"],par["D_man"],par["h_man"],par["w_man"],par["L_man"])

        self.type = par["type"]
        if self.type == "Z":
            self.ref = 0
        elif self.type == "U":
            self.ref = self.N_panel*self.N_riser_per_panel - 1

        self.method = par["method"]
    
        self.theta = par["theta"]
        self.roughness = par["roughness"]

        self.specific_inter_panel = par.get("specific_inter_panel",0)
        self.inter_panel_coeff = par.get("inter_panel_coeff",1)

        self.coeff_Kxin = par["coeff_Kxin"]
        self.coeff_Kxout = par["coeff_Kxout"]
        self.coeff_Kyin = par["coeff_Kyin"]
        self.coeff_Kyout = par["coeff_Kyout"]

    def compute_metrics(self):
        self.M = (self.N*math.pi*(self.riser.D/2)**2)/self.man.A
        self.E = self.man.L/self.man.D
    
        self.R_D = self.riser.D/self.man.D
        self.A_R = self.N*self.R_D**2

    def make_dict(self):
        dict_riser = dict(("{}_{}".format(k,"riser"),v) for k,v in vars(self.riser).items())
        dict_man = dict(("{}_{}".format(k,"man"),v) for k,v in vars(self.man).items())
        dict_man_in = dict(("{}_{}".format(k,"in"),v) for k,v in vars(self.man_in).items())
        dict_man_out = dict(("{}_{}".format(k,"out"),v) for k,v in vars(self.man_out).items())
        par = {**vars(self),**dict_riser,**dict_man,**dict_man_in,**dict_man_out}
        par.pop("riser")
        par.pop("man")
        par.pop("man_in")
        par.pop("man_out")

        return par
    
    def compute_flow(self, Vdot, fluid_dict, p, T):

        self.man_in.compute_flow(Vdot/2, fluid_dict, p, T)
        self.man_out.compute_flow(Vdot/2, fluid_dict, p, T)
        self.riser.compute_flow(Vdot/self.N, fluid_dict, p, T)

    def change_coeff(self,coeff_Kxin,coeff_Kxout,coeff_Kyin,coeff_Kyout):
        self.coeff_Kxin = coeff_Kxin
        self.coeff_Kxout = coeff_Kxout
        self.coeff_Kyin = coeff_Kyin
        self.coeff_Kyout = coeff_Kyout

    def change_man_diameter(self, D):
        self.man.D = D
        self.man_in.D = D
        self.man_out.D = D

        self.man.A = math.pi*(D/2)**2
        self.man_in.A = math.pi*(D/2)**2
        self.man_out.A = math.pi*(D/2)**2

    def change_riser_diameter(self, D):
        self.riser.D = D
        self.riser.A = math.pi*(D/2)**2

    def change_riser_width_and_update_N(self, w):
        h_riser_origin = self.riser.h
        L_riser_origin = self.riser.L
        shape_riser_origin = self.riser.shape
        assert shape_riser_origin == "rectangular", "Riser shape must be rectangular"

        self.riser = duct(shape_riser_origin,math.nan, h_riser_origin, w, L_riser_origin)
        assert self.N_EP == 1, "Number of EP must be 1"
        self.N_riser_per_panel = int( (self.man.L - self.inter_riser) / (w + self.inter_riser) )

        self.update_geometry()

    def change_riser_height(self, h):
        w_riser_origin = self.riser.w
        L_riser_origin = self.riser.L
        shape_riser_origin = self.riser.shape
        assert shape_riser_origin == "rectangular", "Riser shape must be rectangular"

        self.riser = duct(shape_riser_origin, math.nan, h, w_riser_origin, L_riser_origin)

    def change_manifold_length_only(self, L):

        if self.riser.shape == "tubular":
            width = self.riser.D
        elif self.riser.shape == "rectangular":
            width = self.riser.w
        else:
            raise ValueError("Riser shape must be either 'tubular' or 'rectangular'")

        self.man.L = L
        self.man_in.L = L
        self.man_out.L = L

        self.inter_riser = (self.man.L - self.N*width) / (self.N+1)

        self.update_geometry()

    def change_N_total(self, N):

        self.N_riser_per_panel = N
        self_N_riser_per_EP = int(self.N_riser_per_panel/self.N_EP)

        self.update_geometry()

    def update_geometry(self):

        self.N = self.N_riser_per_panel*self.N_panel

        if self.N_EP == 1:
            if self.N_panel == 1:
                self.Ly = (self.N_riser_per_panel-1)*[self.inter_riser]
            else:
                petit = (self.N_riser_per_panel-1)*[self.inter_riser]
                long = [self.inter_panel]
                self.Ly = (self.N_panel-1)*(petit+long)+petit 
        else:
            if self.N_panel == 1:
                petit = (self.N_riser_per_EP-1)*[self.inter_riser]
                long = [self.inter_EP]
                self.Ly = (self.N_EP-1)*(petit+long)+petit # m
            else:
                petit = (self.N_riser_per_EP-1)*[self.inter_riser]
                long = [self.inter_EP]
                Ly = (self.N_EP-1)*(petit+long)+petit
                long = [self.inter_panel]
                self.Ly = (self.N_panel-1)*(Ly+long)+Ly

class system_harp:

    def __init__(self,par):

        self.shape_man = par["shape_man"]
        if self.shape_man == "tubular":
            self.D_in = par["D_in"]
            self.D_out = par["D_out"]
            self.Ain = math.pi*(self.D_in/2)**2
            self.Aout = math.pi*(self.D_out/2)**2
        elif self.shape_man == "rectangular":
            self.h_man = par["h_man"]
            self.w_man = par["w_man"]
            self.D_in = 2*(self.h_man*self.w_man)/(self.h_man+self.w_man)
            self.Ain = self.h_man * self.w_man
            self.Aout = self.Ain

        self.type = par["type"]
        if self.type == "Z":
            self.ref = 0
        elif self.type == "U":
            self.ref = self.N_panel - 1
    
        self.theta = par["theta"]
        self.roughness = par["roughness"]

        self.coeff_Kxin = par["coeff_Kxin"]
        self.coeff_Kxout = par["coeff_Kxout"]
        self.coeff_Kyin = par["coeff_Kyin"]
        self.coeff_Kyout = par["coeff_Kyout"]

    def make_dict(self):
        dict_riser = dict(("{}_{}".format(k,"riser"),v) for k,v in vars(self.riser).items())
        dict_man = dict(("{}_{}".format(k,"man"),v) for k,v in vars(self.man).items())
        par = {**vars(self),**dict_riser,**dict_man}
        par.pop("riser")
        par.pop("man")

class duct:

    def __init__(self,shape,D=None,h=None,w=None,L=None,k=None):
        self.shape = shape

        if self.shape == "tubular":
            self.D = D
            self.A = math.pi*(self.D/2)**2
        elif self.shape == "rectangular":
            self.h = h
            self.w = w
            self.D = 2*self.h*self.w/(self.h+self.w)
            self.A = self.h*self.w

        self.L = L
        if k != None:
            self.k = k
        else:
            self.k = 0.001*1e-3

    def change_D(self, D):
        self.D = D
        self.A = math.pi*(self.D/2)**2

    def compute_flow(self, Vdot, fluid_dict, p, T):

        fluid = find_fluid(fluid_dict)

        Dv = Vdot/(3.6*1E6) # m3/s
        V = Dv/self.A

        rho = PropsSI('D', 'P', p, 'T', T, fluid) # kg/m3
        eta = PropsSI('V', 'P', p, 'T', T, fluid) # kg/m3

        Re = fds.core.Reynolds(V,self.D,rho,mu=eta) # viscosité dynamique mu ou eta)
        f = fds.friction.friction_factor(Re = Re, eD=self.k/self.D)
        K = fds.K_from_f(f,self.L,self.D)
        dP = fds.dP_from_K(K,rho,V=V)/1000

        self.Vdot = Vdot
        self.Dv = Dv
        self.Re = Re
        self.V = V

        self.rho = rho
        self.eta = eta

        self.mdot = self.Dv*self.rho
        
        self.f = f
        self.K = K
        self.dP = dP

    def regular_PL(self, Vdot, fluid_dict, p, T):
        """Computes regular pressure losses for the duct in kPa"""

        self.compute_flow(Vdot,fluid_dict,p,T)

        return self.dP
    
class bend:

    def __init__(self, D, angle, rc=None, bend_diameters=None, roughness=0.0, L_unimpeded=None):

        self.D = D
        self.A = math.pi*(self.D/2)**2
        self.angle = angle
        self.rc = rc
        self.bend_diameters = bend_diameters
        self.roughness = roughness
        self.L_unimpeded = L_unimpeded

    def singular_PL(self,Vdot,fluid,glycol_rate,p,T):

        
        Dv = Vdot/(3.6*1E6)
        V = Dv/self.A

        p = dp.check_unit_p(p)
        T = dp.check_unit_T(T)

        rho = PropsSI('D', 'P', p, 'T', T, f'INCOMP::{fluid}[{glycol_rate}]') # kg/m3
        eta = PropsSI('V', 'P', p, 'T', T, f'INCOMP::{fluid}[{glycol_rate}]') # kg/m3

        Re = fds.core.Reynolds(V,self.D,rho,mu=eta) # viscosité dynamique mu ou eta)
        K = fds.fittings.bend_rounded(self.D, self.angle, rc=self.rc, bend_diameters=self.bend_diameters, Re=Re, roughness=self.roughness, L_unimpeded=self.L_unimpeded)
        dP = fds.dP_from_K(K,rho,V=V)/1000

        return dP