# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:15:59 2022

@author: Hossein
"""

from pyclbr import Function
import numpy as np
import json
import dolfin as dolfin
import os
from ..dependencies.forms import Forms
from ..dependencies.nsolver import NSolver
from ..dependencies.assign_heterogeneous_params import assign_heterogeneous_params 
import vtk_py
from mpi4py import MPI as pyMPI

class MeshClass():

    def __init__(self, parent_parameters,
                 predefined_mesh=None,
                 predefined_functions=None):

        self.parent_parameters = parent_parameters
        self.hs = self.parent_parameters.hs
        mesh_struct = parent_parameters.instruction_data['mesh']

        if self.parent_parameters.comm.Get_size() > 1:
            parameters['mesh_partitioner'] = 'SCOTCH'

        self.model = dict()
        self.data = dict()

        # Check if a predefined mesh is given, else create a new mesh
        if not predefined_mesh:
            # New BiV mesh creation and facet extraction logic
            ugrid = vtk_py.create_BiVmesh(
                mesh_struct['epicutfilename'],
                mesh_struct['LVendocutfilename'],
                mesh_struct['RVendocutfilename'],
                mesh_struct['casename'],
                meshsize=mesh_struct.get('meshsize', 0.6)
            )

            # Extract FEniCS mesh and facets
            self.model['mesh'], fenics_facet_ref, fenics_edge_ref = vtk_py.extractFeNiCsBiVFacet(
                ugrid, tol=1e-1
            )

            # Assign material IDs
            pdata_endLV = vtk_py.readSTL(mesh_struct['LVendocutfilename'], verbose=False)
            pdata_endRV = vtk_py.readSTL(mesh_struct['RVendocutfilename'], verbose=False)
            pdata_epi = vtk_py.readSTL(mesh_struct['epicutfilename'], verbose=False)
            matid = vtk_py.addRegionsToBiV(ugrid, pdata_endLV, pdata_endRV, pdata_epi)
            cnt = 0
            for cell in dolfin.cells(self.model['mesh']):
                idx = int(ugrid.GetCellData().GetArray("region_id").GetTuple(cnt)[0])
                self.model['mesh'].domains().set_marker((cell.index(), idx), 3)
                cnt += 1

            matid = dolfin.MeshFunction("size_t", self.model['mesh'], 3, self.model['mesh'].domains())

            # Translate mesh so that z = 0
            comm = self.model['mesh'].mpi_comm().tompi4py()
            ztop = comm.allreduce(np.amax(self.model['mesh'].coordinates()[:, 2]), op=pyMPI.MAX)
            ztrans = dolfin.Expression(("0.0", "0.0", str(-ztop)), degree=1)
            dolfin.ALE.move(self.model['mesh'], ztrans)

            # Save mesh and facets to HDF5
            hdf5_path = os.path.join(os.getcwd(), mesh_struct['BiV_Mesh'] + ".hdf5")
            f = dolfin.HDF5File(self.model['mesh'].mpi_comm(), hdf5_path, "w")
            f.write(self.model['mesh'], mesh_struct['BiV_Mesh'])
            f.write(fenics_facet_ref, mesh_struct['BiV_Mesh'] + "/" + "facetboundaries")
            f.write(fenics_edge_ref, mesh_struct['BiV_Mesh'] + "/" + "edgeboundaries")
            f.write(matid, mesh_struct['BiV_Mesh'] + "/" + "matid")
            f.close()

        else:
            self.model['mesh'] = predefined_mesh

        # Initialize communicator and functions
        self.comm = self.model['mesh'].mpi_comm()
        self.model['function_spaces'] = self.initialize_function_spaces(mesh_struct)
        self.model['functions'] = self.initialize_functions(mesh_struct, predefined_functions)
        self.model['boundary_conditions'] = self.initialize_boundary_conditions()

        # Set up weak form and solver parameters
        self.model['Ftotal'], self.model['Jac'], \
            self.model['uflforms'], self.model['solver_params'] = self.create_weak_form()

        # Now, setup the fibers
        fiber_angle_param = {
            "mesh": self.model['mesh'],
            "facetboundaries": self.model['functions']['facetboundaries'],
            "LV_fiber_angle": mesh_struct.get("LVangle", [60, -60]),
            "LV_sheet_angle": [0.1, -0.1],
            "Septum_fiber_angle": mesh_struct.get("Septangle", [60, -60]),
            "Septum_sheet_angle": [0.1, -0.1],
            "RV_fiber_angle": mesh_struct.get("RVangle", [60, -60]),
            "RV_sheet_angle": [0.1, -0.1],
            "LV_matid": 0,
            "Septum_matid": 1,
            "RV_matid": 2,
            "matid": matid,
            "isrotatept": False,
            "isreturn": True,
            "outfilename": mesh_struct['BiV_Mesh'],
            "outdirectory": mesh_struct['outdir'],
            "epiid": mesh_struct['epiid'],
            "rvid": mesh_struct['rvid'],
            "lvid": mesh_struct['lvid'],
            "degree": mesh_struct.get("quad_deg", 4),
        }

        ef, es, en = vtk_py.SetBiVFiber_Quad_PyQ(fiber_angle_param)

        # Save fiber orientations
        f = dolfin.HDF5File(self.model['mesh'].mpi_comm(), hdf5_path, "a")
        f.write(ef, mesh_struct['BiV_Mesh'] + "/" + "eF")
        f.write(es, mesh_struct['BiV_Mesh'] + "/" + "eS")
        f.write(en, mesh_struct['BiV_Mesh'] + "/" + "eN")
        f.close()
