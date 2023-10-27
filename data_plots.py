import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def PL_percent(PL):
    PL_per = pd.DataFrame((list(zip(PL["Total PL"], 100*PL["SPL entrance"]/PL["Total PL"], 100*PL["RPL manifold"]/PL["Total PL"], 100*PL["RPL riser"]/PL["Total PL"], 100*PL["SPL tee"]/PL["Total PL"]))), columns = ["Total PL", "SPL entrance", "RPL manifold", "RPL riser", "SPL tee"])
    PL_per.reset_index().plot(x='index', y= ["SPL entrance", "RPL manifold", "RPL riser", "SPL tee"], style='o', xlabel='N° riser', ylabel='% head loss contribution')

def PL_hist(PL):
    PL.plot.bar(y=['SPL entrance', 'RPL manifold', 'RPL riser', 'SPL tee'], stacked = True, xlabel='N° riser', ylabel='Pressure loss')