#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:39:23 2022

@author: jwebb2020
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from LocalResources import dataWrangling as dw


def hrByhr_ttest(a,b):
    ttest_all = []
    for i in range(len(a[0])):
        a_curr = dw.extract_column(a, i)
        b_curr = dw.extract_column(b,i)
        ttest = st.ttest_ind(a_curr,b_curr, equal_var = False)
        ttest = ttest[1]
        ttest_all.append(ttest)
    return ttest_all