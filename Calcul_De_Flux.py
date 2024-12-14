# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:38:05 2024

@author: El Hadji Alassane DIENG
"""

#Resolution du probleme de Cauchy
# d_t u(x,t) +  d_x q(u(x,t)) = 0
# avec comme condition initial:  u(x,t=0)=g(x),
# pour t>0 et x in [0;L] avec conditions au bord de Newmann
# et q(u) = u^2/2 où L est la longueur du domaine


# Importation des module necessaire
from numpy import *
import matplotlib.pyplot as plt


# définition fonctions à utilisés
def flux(u): # flux de Burger : q(u) = u^2/2
    return (0.5*u**2)

def fluxprime(u): #Derivee du flux de Burgers : q'(u)=u
    return u

##########################################################################
def g_simple(a,b):
    return 0.5*(flux(a)+flux(b))

def calcul_flux_simple(u):
    global alpha, Nx
    u0 = uu
    #print(u0[0])
    for i in arange(1,Nx-1):
        u0[i] = u[i] - alpha*( g_simple(u[i],u[i+1])- g_simple(u[i-1],u[i]) )
    # condition au bord
    u0[0]= u0[1]
    u0[Nx-1]= u0[Nx-2]

    return u0



###########################################################################
def signe(x):
    if x > 0:
        return 1
    elif x <0:
        return -1
    else:
        return 0


def g_MR(a,b):
    if (a!=b):
        aux = (signe(flux(a)-flux(b)))*(signe(a-b))
    else:
        aux = signe(fluxprime(a))
    if aux==-1:
        return flux(b)
    else:
        return flux(a)

def calcul_Murman_Roe(u):
     global alpha, Nx
     u0 = u
     #print(u0[0])
     for i in arange(1,Nx-1):
         u0[i] = u[i] - alpha*( g_LW(u[i],u[i+1])- g_LW(u[i-1],u[i]) )
    # condition au bord
     u0[0]= u0[1]
     u0[Nx-1]= u0[Nx-2]
     return u0



###########################################################################
def g_LF(a,b): # Flux numerique de Lax-Friedrichs
    global alpha
    return 0.5*(flux(a)+flux(b)+(a-b)/alpha)

def calcul_lax_Friedrichs(u):
    global alpha, Nx
    u0 = u
    #print(u0[0])
    for i in arange(1,Nx-1):
        u0[i] = u[i] - alpha*( g_LF(u[i],u[i+1])- g_LF(u[i-1],u[i]) )
    # condition au bord
    u0[0]= u0[1]
    u0[Nx-1]= u0[Nx-2]

    return u0
###########################################################################

def g_LW(a, b): #c'est flux intermediaire entre deux point adjooint
    global alpha
    return 0.5*(flux(a)+flux(b))-0.5*alpha*(flux(b)-flux(a))*fluxprime(0.5*(a+b))

def calcul_Lax_Wendroff(u):
    global alpha, Nx
    u0=u
    for i in arange(1, Nx-1):
        u0[i]=u[i]-alpha*(g_LW(u[i],u[i+1])-g_LW(u[i-1],u[i]))
    u0[0] = u0[1]
    u0[Nx-1] = u0[Nx-2]
    return u0

###########################################################################

def g_R(a,b): # Flux numerique de Rusanov
    C=max(abs(fluxprime(a)),abs(fluxprime(b)))
    return 0.5*(flux(a)+flux(b)+C*(a-b))

def calcul_Rusanov(u):
    global alpha, Nx
    u0 = u
    #print(u0[0])
    for i in arange(1,Nx-1):
        u0[i]=u[i]-alpha*(g_R(u[i],u[i+1])-g_R(u[i-1],u[i]))
    # condition au bord
    u0[0]= u0[1]
    u0[Nx-1]= u0[Nx-2]

    return u0


# fonction calculant les différentes solutions exactes:
def calcul_sol_exact(CI,x,t):
    N= shape(x)[0]
    u = zeros(N)
    if CI==1: # choc qui avance selon RH
        uL= 1.0
        uR= 0.0
        qL =  0.5*uL**2 # q(uL)
        qR = 0.5*uR**2 # q(uR)
        sigma = (qR-qL)/(uR-uL)
        for i in range(N):
            if (x[i]< sigma*t+1.5):
                u[i] = uL
            else:
                u[i] = uR
    elif CI==2: # detente
        uL= 0.0
        uR= 1.0
        dqL = uL # q'(uL)
        dqR = uR # q'(uR)
        for i in range(N):
            if (x[i] <= dqL*t+1.5):
                u[i] = uL
            elif (x[i] >= dqR*t+1.5):
                u[i] = uR
            else:
                u[i] = (x[i]-1.5)/t
    elif CI==3:
        for i in range(N):
            if x[i] < 1.0:
                u[i] = 0.0
            elif x[i] < min(1+t,1+sqrt(2*t)):
                u[i] = (x[i]-t)/t
            elif ((x[i]<t*0.5+2) & (t<2)):
                u[i] = 1.0
            else:
                u[i] = 0.0
    else:
        u= sin(pi*x)
    return u



#Programme principale : Main

# Les paramètres pour la simulation
CI = int(input("Donner le numéro de cas test à simuler:"))
Nx = 500 # nombre de point du domaine
L= 12.0  # longueur du domaine
Tf = 1 # temps final d'arret de la simulation
cfl = 0.5# coefficient cfl

# Discrétisation de l'espace
dx = L/(Nx-1) # pas en space
x = linspace(0,L,Nx)

# condition initial du problème
u_init = calcul_sol_exact(CI, x,0)

u = u_init.copy()
u1 = u_init.copy()
u2 = u_init.copy()
u3 = u_init.copy()
t= 0.0 # temps initial

# schéma
while (t< Tf ):
    maxu = (abs(max(u)))
    dt = (cfl*dx/maxu) # calcul du dt par rapport au cfl
    alpha = (dt/dx)
    t = t+ dt
    # methode à utiliser
    #u = calcul_flux_simple(u)
    u = calcul_lax_Friedrichs(u)
    u1 = calcul_Lax_Wendroff(u)
    u2 = calcul_Rusanov(u)
    u3 = calcul_Murman_Roe(u)

# Représentation graphique de la solution
u_ex = calcul_sol_exact(CI, x,t)
plt.plot(x, u, label= "Solution approx LF")
plt.plot(x, u1, "r--", label= "Solution approx LW")
plt.plot(x, u2, "y:o", label= "Solution approx R")
plt.plot(x, u3, "b:+", label= "Solution approx MR")
plt.plot(x,u_init, label="Solution init")
plt.plot(x,u_ex, label="Solution exact" )
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()