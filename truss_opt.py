import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from solidspy.preprocesor import rect_grid
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol

plt.style.use("ggplot")
plt.rcParams["grid.linestyle"] = "dashed"

def is_equilibrium(nodes, elements, mats, loads):
    """
    Check if the system is in equilibrium
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
        
    Returns
    -------
    equil : bool
        Variable True when the system is in equilibrium and False when it doesn't
    """   
    equil = True
    DME, IBC , neq = ass.DME(nodes[:,-2:], elements)
    stiff, _ = ass.assembler(elements, mats, nodes[:,:-2], neq, DME)
    load_vec = ass.loadasem(loads, IBC, neq)
    disp = sol.static_sol(stiff, load_vec)
    if not(np.allclose(stiff.dot(disp)/stiff.max(), load_vec/stiff.max())):
        equil = False

    return equil

def fem_sol(nodes, elements, mats, loads):
    """
    Compute the FEM solution for a given problem.

    Parameters
    ----------
    nodes : array
        Array with nodes
    elements : array
        Array with element information.
    mats : array
        Array with material elements. We need a material profile
        for each element for the optimization process.
    loads : array
        Array with loads.

    Returns
    -------
    disp_comp : array
        Displacement for each node.
    """
    DME, IBC , neq = ass.DME(nodes[:,-2:], elements)
    stiff, _ = ass.assembler(elements, mats, nodes[:,:-2], neq, DME)
    load_vec = ass.loadasem(loads, IBC, neq)
    disp = sol.static_sol(stiff, load_vec)
    disp_comp = pos.complete_disp(IBC, nodes, disp)
    stress_nodes = pos.stress_truss(nodes, elements, mats, disp_comp)
    return disp_comp, stress_nodes


def weight(areas, nodes, elements):
    """Compute the weigth of the truss"""
    ini = elements[:, 3]
    end = elements[:, 4]
    lengths = np.linalg.norm(nodes[end, 1:3] - nodes[ini, 1:3], axis=1)
    return np.sum(areas * lengths)


def compliance(areas, nodes, elements, loads, mats):
    """Compute the compliance of the truss"""
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    forces = np.zeros_like(disp)
    forces[loads[:, 0].astype(int), 0] = loads[:, 1]
    forces[loads[:, 0].astype(int), 1] = loads[:, 2]
    return np.sum(forces*disp)


def stress_cons(areas, nodes, elements, mats, loads, stresses, comp):
    """Return the stress constraints"""
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    cons = np.asarray(stresses) -\
        pos.stress_truss(nodes, elements, mats2, disp)
    return cons[comp]


def stress_bnd(areas, nodes, elements, mats, loads, stresses):
    """Bounds on the stress for each member"""
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    return np.asarray(stresses) -\
        pos.stress_truss(nodes, elements, mats2, disp)


def grid_truss(length, height, nx, ny):
    """
    Generate a grid made of vertical, horizontal and diagonal
    members
    """
    nels = (nx - 1)*ny +  (ny - 1)*nx + 2*(nx - 1)*(ny - 1)
    x, y, _ = rect_grid(length, height, nx - 1, ny - 1)
    nodes = np.zeros((nx*ny, 5))
    nodes[:, 0] = range(nx*ny)
    nodes[:, 1] = x
    nodes[:, 2] = y
    elements = np.zeros((nels, 5), dtype=int)
    elements[:, 0] = range(nels)
    elements[:, 1] = 6
    elements[:, 2] = range(nels)
    hor_bars =  [[cont, cont + 1] for cont in range(nx*ny - 1)
                 if (cont + 1)%nx != 0]
    vert_bars =  [[cont, cont + nx] for cont in range(nx*(ny - 1))]
    diag1_bars =  [[cont, cont + nx + 1] for cont in range(nx*(ny - 1))
                   if  (cont + 1)%nx != 0]
    diag2_bars =  [[cont, cont + nx - 1] for cont in range(nx*(ny - 1))
                   if  cont%nx != 0]
    bars = hor_bars + vert_bars + diag1_bars + diag2_bars
    elements[:len(bars), 3:] = bars
    return nodes, elements, nels, x, y


def plot_truss(nodes, elements, mats, stresses, mask_del=None, tol=1e-5):
    """
    Plot a truss and encodes the stresses in a colormap
    """
    mask = (mats[:,1]==1e-8)
    if mask.sum() > 0:
        stresses[mask] = 0

    max_stress = max(-stresses.min(), stresses.max())
    scaled_stress = 0.5*(stresses + max_stress)/max_stress
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    max_val = 4
    min_val = 0.5
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    for el in elements:
        if areas[el[2]] > tol and mask_del[el[0]]:
            ini, end = el[3:]
            color = plt.cm.seismic(scaled_stress[el[0]])
            plt.plot([nodes[ini, 1], nodes[end, 1]],
                     [nodes[ini, 2], nodes[end, 2]],
                     color=color, lw=widths[el[2]])
    plt.axis("image")

def protect_els(els, loads, BC, mask_del):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes
    mask_del : ndarray 
        Mask array with the elements that must be deleted.
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = mask_del.copy()
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -2:] == p)[:,0]
        mask_els[protect_index] = False

        for protect in protect_index:
            a_ = np.argwhere(els[np.logical_not(mask_els), -2] == els[protect, -2])[:,0]
            b_ = np.argwhere(els[np.logical_not(mask_els), -1] == els[protect, -1])[:,0]
            if len(a_)==1 or len(b_)==1:
                mask_els[protect] = True

    return mask_els

def del_node(nodes, els):
    """
    Retricts nodes dof that aren't been used.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements

    Returns
    -------
    """   
    n_nodes = nodes.shape[0]
    for n in range(n_nodes):
        if n not in els[:, -4:]:
            nodes[n, -2:] = -1


def plot_truss1(nodes, elements, mats, stresses, mask_del=None, tol=1e-5):
    """
    Plot a truss and encodes the stresses in a colormap
    """
    mask = (mats[:,1]==1e-8)
    if mask.sum() > 0:
        stresses[mask] = 0

    max_stress = max(-stresses.min(), stresses.max())
    scaled_stress = 0.5*(stresses + max_stress)/max_stress
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    max_val = 4
    min_val = 0.5
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    for el in elements:
        ini, end = el[3:]
        plt.plot([nodes[ini, 1], nodes[end, 1]],
                    [nodes[ini, 2], nodes[end, 2]],
                    color=(1.0, 0.0, 0.0, 1.0), lw=widths[el[2]])
    plt.axis("image")