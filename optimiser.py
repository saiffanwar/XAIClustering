''' This file contains the Cluster Optimiser class which is used to optimise the number of medoids when clustering the LLR models to approximate a more complex model.'''
import numpy as np
import pandas as pd


class MedoidOptimiser():

    def __init__(self, x,y):


