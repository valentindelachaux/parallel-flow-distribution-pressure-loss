import os
import math
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import openpyxl as opxl
from openpyxl.utils.dataframe import dataframe_to_rows

from CoolProp.CoolProp import PropsSI
import fluids as fds

import data_processing as dp

class hx_harp:

    def __init__(self,par):
        self.sch = par["sch"]

        if self.sch == "exchanger":

            self.N_panel = par["N_panel"]
            self.N_EP = par["N_EP"]
            self.N_riser_per_panel = par["N_riser_per_panel"]
            self.N_per_EP = int(self.N_riser_per_panel/self.N_EP)


            self.inter_riser = par["inter_riser"]
            self.inter_EP = par["inter_EP"]
            self.inter_panel = par["inter_panel"]

            self.N = self.N_riser_per_panel*self.N_panel

            if self.N_EP == 1:
                self.Ly = (self.N_riser_per_panel-1)*[self.inter_riser]
            else:
                if self.N_panel == 1:
                    petit = (self.N_per_EP-1)*[self.inter_riser]
                    long = [self.inter_EP]
                    self.Ly = (self.N_EP-1)*(petit+long)+petit # m
                else:
                    petit = (self.N_per_EP-1)*[self.inter_riser]
                    long = [self.inter_EP]
                    Ly = (self.N_EP-1)*(petit+long)+petit
                    long = [self.inter_panel]
                    self.Ly = (self.N_panel-1)*(Ly+long)+Ly

        else:

            self.N_panel = par["N_panel"]
            self.a_x = par["a_x"]
            self.b_x = par["b_x"]
            self.inter_panel = par["inter_panel"]

            self.N = self.N_panel

            self.Ly = (self.N_panel-1)*[self.inter_panel]


        self.riser = duct(par["shape_riser"],par["D_riser"],par["h_riser"],par["w_riser"],par["L_riser"])
        self.man = duct(par["shape_man"],par["D_man"],par["h_man"],par["w_man"],par["L_man"])

        # self.shape_man = par["shape_man"]
        # if self.shape_man == "tubular":
        #     self.Din = par["Din"]
        #     self.Dout = par["Dout"]
        #     self.Ain = math.pi*(self.Din/2)**2
        #     self.Aout = math.pi*(self.Dout/2)**2
        # elif self.shape_man == "rectangular":
        #     self.h_man = par["h_man"]
        #     self.w_man = par["w_man"]
        #     self.Din = 2*(self.h_man*self.w_man)/(self.h_man+self.w_man)
        #     self.Ain = self.h_man * self.w_man
        #     self.Aout = self.Ain

        self.type = par["type"]
        if self.type == "Z":
            self.ref = 0
        elif self.type == "U":
            self.ref = self.N_panel*self.N_riser_per_panel - 1

        self.method = par["method"]
    
        self.theta = par["theta"]
        self.roughness = par["roughness"]

        self.coeff_Kxin = par["coeff_Kxin"]
        self.coeff_Kxout = par["coeff_Kxout"]
        self.coeff_Kyin = par["coeff_Kyin"]
        self.coeff_Kyout = par["coeff_Kyout"]

    def compute_metrics(self):
        self.M = (self.N*math.pi*self.riser.D)/self.man.A
        self.E = self.man.L/self.man.D

    def make_dict(self):
        dict_riser = dict(("{}_{}".format(k,"riser"),v) for k,v in vars(self.riser).items())
        dict_man = dict(("{}_{}".format(k,"man"),v) for k,v in vars(self.man).items())
        par = {**vars(self),**dict_riser,**dict_man}
        par.pop("riser")
        par.pop("man")

        return par

    def change_coeff(self,coeff_Kxin,coeff_Kxout,coeff_Kyin,coeff_Kyout):
        self.coeff_Kxin = coeff_Kxin
        self.coeff_Kxout = coeff_Kxout
        self.coeff_Kyin = coeff_Kyin
        self.coeff_Kyout = coeff_Kyout


class system_harp:

    def __init__(self,par):

        self.shape_man = par["shape_man"]
        if self.shape_man == "tubular":
            self.Din = par["Din"]
            self.Dout = par["Dout"]
            self.Ain = math.pi*(self.Din/2)**2
            self.Aout = math.pi*(self.Dout/2)**2
        elif self.shape_man == "rectangular":
            self.h_man = par["h_man"]
            self.w_man = par["w_man"]
            self.Din = 2*(self.h_man*self.w_man)/(self.h_man+self.w_man)
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

    def __init__(self,shape,D,h,w,L):
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

    def regular_PL(self,Vdot,fluid,glycol_rate,p,T):
        """Computes regular pressure losses for the duct in kPa"""

        Dv = Vdot/(3.6*1E6)
        V = Dv/self.A

        p = dp.check_unit_p(p)
        T = dp.check_unit_T(T)

        rho = PropsSI('D', 'P', p, 'T', T, f'INCOMP::{fluid}[{glycol_rate}]') # kg/m3
        eta = PropsSI('V', 'P', p, 'T', T, f'INCOMP::{fluid}[{glycol_rate}]') # kg/m3

        Re = fds.core.Reynolds(V,self.D,rho,mu=eta) # viscosité dynamique mu ou eta)
        f = fds.friction.friction_factor(Re = Re,eD=0.001/self.D)
        K = fds.K_from_f(f,self.L,self.D)
        dP = fds.dP_from_K(K,rho,V=V)/1000

        return dP