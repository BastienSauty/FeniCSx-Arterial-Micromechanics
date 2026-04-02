from dolfinx.plot import vtk_mesh
from dolfinx import fem, plot
import ufl
import numpy as np

### Plotting functions
import pyvista as pv

pv.global_theme.color = 'white'
pv.global_theme.background = 'white'
#pv.global_theme.show_edges = True
pv.global_theme.font.color = 'black'


def init_vtk_mesh(domain, u):
    """
    Create an object that contain everything needed to create a vtk mesh in the plot functions
    contain the initial vtk mesh
    AND total displacement u
    """
    V_elem = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
    V = fem.FunctionSpace(domain, V_elem)

    cells, types, x = plot.vtk_mesh(V)
    
    class vtk_mesh_class:
        def __init__(self, cells, types, x, u):
            self.x = x
            self.cells = cells
            self.types = types
            self.u = u
            
    return(vtk_mesh_class(cells, types, x, u))
    
    
def plot_scalar(f, vtk_mesh, warp_by_u=False, axis="3D", title='scalar'):
    # vtk_mesh is the INITIAL mesh in our problem, created with the function init_vtk_mesh
    # warp_by_u to warp the field plotted by the total displacement u
    
    grid = pv.UnstructuredGrid(vtk_mesh.cells, vtk_mesh.types, vtk_mesh.x)
    # grid.point_data["u"] = vtk_mesh.u.x.array

    
    p = pv.Plotter()
    p.add_text(title, font_size=14, color="black", position="upper_edge")
    
    if warp_by_u:
        p.add_mesh(grid, style="wireframe", color="k")
        
        vals = np.zeros((vtk_mesh.x.shape[0], 3))
        vals[:,:len(vtk_mesh.u)] = vtk_mesh.u.x.array.reshape((vtk_mesh.x.shape[0], len(vtk_mesh.u)))
        grid["u"] = vals
        
        warped = grid.warp_by_vector("u", factor=1.)
               
        warped.cell_data[title] = f.vector.array
        warped.set_active_scalars(title)
        
        p.add_mesh(warped, show_edges=False, show_scalar_bar=True)
        
    else:
        grid.cell_data[title] = f.vector.array
        p.add_mesh(grid, show_edges=True, show_scalar_bar=True)
        
    if axis=="xy":
        p.view_xy()
    elif axis=="yz":
        p.view_yz()
    elif axis=="xz":
        p.view_xz()
    pv.start_xvfb()
    if pv.OFF_SCREEN:
        p.screenshot("2D_function_warp.png", transparent_background=transparent,
                              window_size=[figsize, figsize])
    else:
        p.show()

def plot_scalar_double(f, g, vtk_mesh, title_f='scalar', title_g='scalar'):
    # vtk_mesh is the INITIAL mesh in our problem, created with the function init_vtk_mesh
    # DO NOT WARP BY DISP -> plot is in initial conf
    
    grid_f = pv.UnstructuredGrid(vtk_mesh.cells, vtk_mesh.types, vtk_mesh.x)
    grid_g = pv.UnstructuredGrid(vtk_mesh.cells, vtk_mesh.types, vtk_mesh.x)
    
    grid_f.cell_data[title_f] = f.vector.array
    grid_g.cell_data[title_g] = g.vector.array
    
    subplotter = pv.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text(title_f, font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid_f, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text(title_g, font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid_g, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()
    pv.start_xvfb()
    if pv.OFF_SCREEN:
        subplotter.screenshot("2D_function_warp.png", transparent_background=transparent,
                              window_size=[figsize, figsize])
    else:
        subplotter.show()
        
        
def plot_scalar_doublea(u, v, V, title_u, title_v):
    # To visualize the function u, we create a VTK-compatible grid to
    # values of u to
    cells, types, x = plot.create_vtk_mesh(V)
    grid_u = pv.UnstructuredGrid(cells, types, x)
    grid_u.point_data[title_u] = u.x.array
    
    grid_v = pv.UnstructuredGrid(cells, types, x)
    grid_v.point_data[title_v] = v.x.array

    # The function "u" is set as the active scalar for the mesh, and
    # warp in z-direction is set
    grid_u.set_active_scalars(title_u)
    grid_v.set_active_scalars(title_v)
    #warped = grid_u.warp_by_scalar()

    # A plotting window is created with two sub-plots, one of the scalar
    # values and the other of the mesh is warped by the scalar values in
    # z-direction
    subplotter = pv.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text(title_u, font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid_u, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text(title_v, font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid_v, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()
    if pv.OFF_SCREEN:
        subplotter.screenshot("2D_function_warp.png", transparent_background=transparent,
                              window_size=[figsize, figsize])
    else:
        subplotter.show()
        
