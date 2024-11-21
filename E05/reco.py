import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Hit:

    def __init__(self, m,s,t):
        """
        costruttore di Id Modulo (m);
        Id Sensore(s);
       Time Stamp rivelazione(t).
        inizializzazione degli attributi
        """
        self.m=m
        self.s=s
        self.t=t

    """
    queste cose si usano per definire comparatori per oggetti. permettono il confrontro tra due oggetti
    
    """
    def __str__(self):
        return "Hit(m='{:}', s={:},t={:})".format(self.m, self.s, self.t)

    def __eq__(self, other):
        return  self.m == other.m and self.t == other.t


    def __lt__(self, other):
        return self.m < other.m  and self.t<other.t

    def __gt__(self, other):
        return self.m>other.m and self.t>other.t


    
