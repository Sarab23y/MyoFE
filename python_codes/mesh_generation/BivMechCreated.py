# -*- coding: utf-8 -*-
"""
Created OCT 20 2024

@author: Sara
"""



import sys
import os
import vtk_py as vtk_py
import dolfin as dolfin
import numpy as np
from mpi4py import MPI as pyMPI

# Parameters
epicutfilename = "91690_EP2_2.stl"    # Epicardial surface
LVendocutfilename = "91690_LV2_2.stl"  # Left Ventricle (LV) endocardial surface
RVendocutfilename = "91690_RV2_2.stl"  # Right Ventricle (RV) endocardial surface
meshsize = 0.6                         # Mesh size
casename = "BiV"                       # Case name for file outputs
epiid = 1                              # Epicardium material ID
lvid = 2                               # Left Ventricle material ID
rvid = 3                               # Right Ventricle material ID
isLV = False                           # Whether to create LV-only or BiV mesh
quad_deg = 4                           # Quadrature degree for FEM integration
RVangle = [60, -60]                    # Fiber angles for RV
LVangle = [60, -60]                    # Fiber angles for LV
Septangle = [60, -60]                  # Fiber angles for septum
isrotatept = False                     # Rotation flag for fibers
isreturn = True                        # Flag to return fiber fields
meshname = "91690"                     # Mesh name for output files
outdir = "./"                          # Output directory


def BiV_Mesh(vtk_file_str='', output_file_str='', quad_deg=4, 
             LVangle=[60, -60], RVangle=[60, -60], Septangle=[60, -60], 
             epicutfilename='', LVendocutfilename='', RVendocutfilename='',
             epiid=1, lvid=2, rvid=3, meshsize=0.6):

    # Create VTK mesh for BiV model (Epicardium, LV and RV endocardium)
    ugrid = vtk_py.create_BiVmesh(epicutfilename, LVendocutfilename, RVendocutfilename, 
                                  casename, meshsize=meshsize)

    # Extract mesh and facets for FEniCS
    fenics_mesh_ref, fenics_facet_ref, fenics_edge_ref = vtk_py.extractFeNiCsBiVFacet(ugrid, tol=1e-1)

    # Read STL files for LV, RV, and Epicardium for region marking
    pdata_endLV = vtk_py.readSTL(LVendocutfilename, verbose=False)
    pdata_endRV = vtk_py.readSTL(RVendocutfilename, verbose=False)
    pdata_epi = vtk_py.readSTL(epicutfilename, verbose=False)

    # Assign material IDs to regions (LV, RV, Epicardium)
    matid = vtk_py.addRegionsToBiV(ugrid, pdata_endLV, pdata_endRV, pdata_epi)
    cnt = 0
    for cell in dolfin.cells(fenics_mesh_ref):
        idx = int(ugrid.GetCellData().GetArray("region_id").GetTuple(cnt)[0])
        fenics_mesh_ref.domains().set_marker((cell.index(), idx), 3)
        cnt += 1
    matid = dolfin.MeshFunction("size_t", fenics_mesh_ref, 3, fenics_mesh_ref.domains())

    # Translate mesh so that z = 0 (align with xy-plane)
    comm = fenics_mesh_ref.mpi_comm().tompi4py()
    ztop = comm.allreduce(np.amax(fenics_mesh_ref.coordinates()[:, 2]), op=pyMPI.MAX)
    ztrans = dolfin.Expression(("0.0", "0.0", str(-ztop)), degree=1)

    if dolfin.dolfin_version() != "1.6.0":
        dolfin.ALE.move(fenics_mesh_ref, ztrans)
    else:
        fenics_mesh_ref.move(ztrans)

    # Save mesh and facets to file
    dolfin.File(output_file_str + "BiVFacet.pvd") << fenics_facet_ref
    dolfin.File(output_file_str + "matid.pvd") << matid

    # Set fiber orientation for LV, RV, and septum regions
    fiber_angle_param = {
        "mesh": fenics_mesh_ref,
        "facetboundaries": fenics_facet_ref,
        "LV_fiber_angle": LVangle,
        "LV_sheet_angle": [0.1, -0.1],
        "Septum_fiber_angle": Septangle,
        "Septum_sheet_angle": [0.1, -0.1],
        "RV_fiber_angle": RVangle,
        "RV_sheet_angle": [0.1, -0.1],
        "LV_matid": lvid,
        "Septum_matid": 1,  # Assuming septum has matid = 1
        "RV_matid": rvid,
        "matid": matid,
        "isrotatept": isrotatept,
        "isreturn": isreturn,
        "outfilename": meshname,
        "outdirectory": output_file_str,
        "epiid": epiid,
        "rvid": rvid,
        "lvid": lvid,
        "degree": quad_deg,
    }

    ef, es, en = vtk_py.SetBiVFiber_Quad_PyQ(fiber_angle_param)

    # Write mesh and fiber orientations to HDF5 file
    f = dolfin.HDF5File(fenics_mesh_ref.mpi_comm(), output_file_str + meshname + ".hdf5", "w")
    f.write(fenics_mesh_ref, meshname)
    f.close()

    f = dolfin.HDF5File(fenics_mesh_ref.mpi_comm(), output_file_str + meshname + ".hdf5", "a")
    f.write(fenics_facet_ref, meshname + "/" + "facetboundaries")
    f.write(fenics_edge_ref, meshname + "/" + "edgeboundaries")
    f.write(matid, meshname + "/" + "matid")
    f.write(ef, meshname + "/" + "eF")
    f.write(es, meshname + "/" + "eS")
    f.write(en, meshname + "/" + "eN")
    f.close()


if __name__ == "__main__":
    # Define the output folder for saving the mesh and results
    output_folder = 'output_files/BiV_model/'
    os.makedirs(output_folder, exist_ok=True)

    # Call the function to create the BiV mesh
    BiV_Mesh(vtk_file_str="91690.vtk",
             output_file_str=output_folder,
             quad_deg=4,
             LVangle=[60, -60],
             RVangle=[60, -60],
             Septangle=[60, -60],
             epicutfilename=epicutfilename,
             LVendocutfilename=LVendocutfilename,
             RVendocutfilename=RVendocutfilename,
             epiid=epiid, lvid=lvid, rvid=rvid,
             meshsize=0.6)

    print("Biventricular mesh created")
