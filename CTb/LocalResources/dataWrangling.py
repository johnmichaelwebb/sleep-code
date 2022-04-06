#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:22:15 2022

@author: jwebb2020
"""
import numpy as np

def get_tot_per_brain(a):
    new = []
    for i in range(int(len(a)/3)):
        curr = []
        for j in range(3):
            curr.append(a[i*3+j])
        new.append(np.sum(curr))
    return new

def create_perc(a, b, c):
    a_perc = []
    b_perc = []
    c_perc = []
    for i in range(len(a)):
        SUM = a[i] + b[i] +c[i]
        a_perc.append(a[i]/SUM*100)
        b_perc.append(b[i]/SUM*100)
        c_perc.append(c[i]/SUM*100)
    return a_perc, b_perc, c_perc  