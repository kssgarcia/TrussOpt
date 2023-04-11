#%% imports
import numpy as np
from truss_opt import *
#%% Difine mesh
length = 6
height = 3
nx = 11
ny = 9
nodes, elements, nels, x, y = grid_truss(length, height, nx, ny)

mask_loads = (x==length/2) & (y==0)
loads = np.zeros((mask_loads.sum(), 3))
loads[:, 0] = nodes[mask_loads, 0]
loads[:, 2] = -1.0

maskBC_1 = (x == -length/2) & (y==height/2)
maskBC_2 = (x == -length/2) & (y==-height/2)
nodes[maskBC_1, 3] = -1
nodes[maskBC_1, 4] = -1
nodes[maskBC_2, 3] = -1
nodes[maskBC_2, 4] = -1
BC = nodes[(nodes[:,-2] == -1) & (nodes[:,-1] == -1), 0]

#areas = np.random.uniform(low=0.1, high=1.0, size=nels)
mats = np.ones((nels, 2))
areas = np.ones(nels)
mats[:, 1] = areas
matsI = mats.copy()
#%% Optimization
niter = 20
RR = 0.01
ER = 0.001
V_opt = int(nels * 0.50)
ELS = None

mats2 = mats.copy()
for i in range(40):
    if not is_equilibrium(nodes, elements, mats, loads) or (mats2[:,1]>1e-8).sum() < V_opt: 
        print('Not equilibrium')
        break
    mats2 = mats.copy()
    disp, stress = fem_sol(nodes, elements, mats, loads)
    RR_el = np.abs(stress)/np.abs(stress.max())
    mask_del = RR_el < RR

    mask_els = protect_els(elements, loads, BC, mask_del)
    mask_del *= mask_els
    mats[mask_del, 1] = 1e-8
    RR += ER
print(RR)
#%% Plotting
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title('Original truss')
plot_truss(nodes, elements, matsI, loads)
plt.subplot(122)
plt.title('Optimize truss')
plot_truss(nodes, elements, mats2, loads)
plt.show()