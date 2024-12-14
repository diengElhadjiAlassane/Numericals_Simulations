"""
Resolution du probleme de Cauchy
 d_t W(x,t) +  d_x F(W(x,t)) = 0 sur [x_min, x_max]
avec comme condition initial:  W(x,t=0)=g(x), 
pour t>0 et x in [0;L] adpater à l'équation de Saint-Venant 
définit par le système suivant: 
   (1) d_t h + d_x (hu) = 0
   (2)  d_t (hu) + d_x(hu^2 + 0.5gh^2) = -gh d_x Z

h= hauteur d'eau 
q = debit de l'eau 
u= le vitesse de l'eau 
z= fonction de topographie du sol     

Auteur : El Hadji Alassane DIENG
Date: 2024
"""

# Importation des module necessaire 
from numpy import * 
import matplotlib.pyplot as plt

################################################################
# Les procédure à utiliser 
def calcul_v(h,hu):
    if h > 1.e-6: 
        return hu/h
    else: 
        return 1.e-6


def pasdetemps(Nx,dx,cfl,W,z): 
    aux = 0.0
    for i in arange(Nx-1):
        hl = W[0,i]
        hR = W[0,i+1] 
        ul = calcul_v(W[0,i],W[1,i])
        ur = calcul_v(W[0,i+1],W[1,i+1])
        
        cl = sqrt(g*hl)
        cr = sqrt(g*hr)
        aux = max(abs(ur)+cr,abs(ul)+cl,aux)
    return dx/aux*cfl


def flux_lf(W,Nx,g,z,lambd): # calcul du flux de lax frederichs 
    res_flux = zeros((2,Nx))
    for i in arange(0,Nx-1):
        hl = W[0,i]
        hr = W[0,i+1]
        
        ul = calcul_v(hl,W[1,i])
        ur = calcul_v(hr,W[1,i])
        
        pil = hl*ul**2 + g*0.5*hl**2
        pir = hr*ur**2 + g*0.5*hr**2
        
        res_flux[0,i] = 0.5*(hl*ul+hr*ur) - (0.5*(hr-hl))/lambd
        res_flux[1,i] = 0.5*(pil + pir) -(0.5*(hr*ur-hl*ul))/lambd
        
    return res_flux




################################################################
# Programme principal: Main 
# Paramètre du probléme 
Nx= 100   # nombre de point du maillage 
Tf = 0.1 # temps final 
x_min = 0. # borne inf de l'intervalle 
x_max = 1.  # borne sup de l'intervalle 
g= 10. # coefficient de gravitation 
itermax = 500 # interation maximal  
cfl = 0.5 # cofficient cfl 

# données de simulation pour le problème de cauchy 
hl = 1.0
hr= 0.5
ur = 0.
ul = 0. 

# allocation des variables de bases 
W = zeros((2,Nx), dtype=float)       # W = (h,q=hu) 
flux = zeros((2,Nx), dtype=float)    # fonction flux numérique 
z = zeros(Nx, dtype=float)           # la topographie 
 

dx = (x_max-x_min)/(Nx-1) # pas en espace 
x = linspace(x_min,x_max,Nx)

# initialisation 
for i in arange(0,Nx): 
    if x[i] < (x_max+x_min)/2.: 
        W[0,i] =  1.  #max(0.0, hl - z[i])
        W[1, i] =  1.e-6 #max(0.0, hl - z[i])*ul
    else:
        W[0,i] = 0.5 # max(0.0, hr - z[i])
        W[1, i] = 1.e-6 # max(0.0, hr - z[i])*ur

niter = 0
t = 0.0 

    
while ((niter < itermax) and ( t < Tf)):
    dt = pasdetemps(Nx,dx,cfl,W,z)
    if (t+dt > Tf):
        dt = Tf-t
    t = t + dt
    niter = niter +1

    lambd = dt/dx
    flux = flux_lf(W,Nx,g,z,2.*lambd)
   # print(flux)
    for i in arange(1,Nx-1): 
        W[0,i] = W[0,i]  - lambd*(flux[0,i] - flux[0,i-1])
        W[1,i] = W[1,i]  - lambd*(flux[1,i] - flux[1,i-1])

    # condition aux limites 
    W[:,0] = W[:,1]
    W[:,Nx-1] = W[:,Nx-2] 
    
    for i in arange(0,Nx):
        if (W[0,i] < 1.e-6):
            W[0,i] = 1.e-6
            W[1,i] = 1.e-6
            

# Représentation graphique de la solution
plt.plot(x, W[0,:]+z, label= "free surface h+z", lw=3)
#plt.plot(x,z, label="Topographie")
plt.xlabel('x')
plt.ylabel('h+z')
plt.legend()
plt.show()
    





