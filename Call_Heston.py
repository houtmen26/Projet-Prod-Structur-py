from Modele Heston import param
import pandas as pd
import numpy as np

### L'objectif ici  étant de pricer un call avec le modele de Heston

# On commence par recuperer les parametre calibré par notre modele puis on va simuler S et V en meme temps.

# Une fois cela effectué on pourra par MC calculter S_T donc S[-1] et farie (S_T-K) et faire la moyenne dess simulation,
# il faudra faire des test pour que ca dure un peu moins de 1 minutes sinon ca sera trop long (call spread, strip strap ect prendront
# trop de temps)



def Call_heston(S0,v0,k,rho,neta,theta,K,r,Nmc,N):
    S = [0]*N
    V = [0] * N

    for i in range(Nmc):
        for n in range(N):

        pass
    pass
pass