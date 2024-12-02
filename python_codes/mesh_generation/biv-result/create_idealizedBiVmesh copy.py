import os
import glob
import vtk
import dolfin
from mpi4py import MPI as pyMPI
from vtk.util import numpy_support
#from tqdm import tqdm  # For progress visualization

# Import vtk_py module (assumes the functions are available)
import vtk_py
import sys
sys.path.append(r"C:\Users\sba431\biv-result\vtk_py")


def main():
    
    gmsh_path = r"C:\Users\sba431\Github\MyoFE\python_codes\mesh_generation\biv-result\gmsh-4.13.1-Windows64\gmsh.exe"  # Path to Gmsh executable
    directory = r"C:\Users\sba431\Github\MyoFE\python_codes\mesh_generation\biv-result"
 # Directory to save all results
    meshname = "biv_idealized3_generalized"  # Base name for mesh files

    # Paths for geo and vtk files
    geo_file = os.path.join(directory, meshname + ".geo")
    vtk_file = os.path.join(directory, meshname + ".vtk")

    # ===================================
    # MPI Setup 
    # ===================================
    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()

    # =====================
    # Step 1: Generate Mesh
    # =====================
    print("Generating mesh from {}...".format(geo_file))
    if not os.path.exists(geo_file):
        print("Debug: Full path '{}' does not exist.".format(geo_file))
        raise IOError("{} not found. Ensure the .geo file exists.".format(geo_file))

    gmsh_command = '"{}" -3 "{}" -o "{}"'.format(gmsh_path, geo_file, vtk_file)
    os.system(gmsh_command)

    if not os.path.exists(vtk_file):
        raise RuntimeError("Mesh generation failed. {} not created.".format(vtk_file))

    print("Mesh generated: {}".format(vtk_file))

    # ===========================
    # Step 2: Process the Mesh
    # ===========================
    print("Processing mesh: {}...".format(vtk_file))

    # Read in the VTK mesh
    ugrid = vtk_py.readUGrid(vtk_file)

    # Rotate the mesh (align z=0 at base)
    print("Rotating mesh...")
    ugrid_rot = vtk_py.rotateUGrid(ugrid, rx=0.0, ry=-90.0, rz=0.0, sx=1.0, sy=1.0, sz=1.0)

    # Reduce point precision
    print("Reducing point precision...")
    newpts = vtk.vtkPoints()
    for p in tqdm(range(ugrid_rot.GetNumberOfPoints()), desc="Updating Points"):
        pt = [round(ugrid_rot.GetPoints().GetPoint(p)[k], 5) for k in range(3)]
        newpts.InsertNextPoint(pt)
    ugrid_rot.SetPoints(newpts)

    # Save the rotated mesh
    rotated_file = os.path.join(directory, meshname + "_rot.vtk")
    vtk_py.writeUGrid(ugrid_rot, rotated_file)
    print("Rotated mesh saved: {}".format(rotated_file))

    # ==========================
    # Step 3: Extract FEniCS Mesh
    # ==========================
    print("Extracting FEniCS mesh...")
    fenics_mesh, fenics_facet, fenics_edge = vtk_py.extractFeNiCsBiVFacet(ugrid_rot)

    # Save FEniCS meshes
    dolfin.File(os.path.join(directory, "bivmesh.pvd")) << fenics_mesh
    dolfin.File(os.path.join(directory, "bivfacet.pvd")) << fenics_facet
    dolfin.File(os.path.join(directory, "bivedge.pvd")) << fenics_edge
    print("FEniCS mesh files saved.")

    # ============================
    # Step 4: Calculate Volumes
    # ============================
    print("Calculating LV and RV cavity volumes...")
    X = dolfin.SpatialCoordinate(fenics_mesh)
    N = dolfin.FacetNormal(fenics_mesh)
    ds = dolfin.ds(subdomain_data=fenics_facet)

    lv_vol_form = -dolfin.Constant(1.0 / 3.0) * dolfin.inner(N, X) * ds(2)
    rv_vol_form = -dolfin.Constant(1.0 / 3.0) * dolfin.inner(N, X) * ds(3)

    lv_vol = dolfin.assemble(lv_vol_form, form_compiler_parameters={"representation": "uflacs"})
    rv_vol = dolfin.assemble(rv_vol_form, form_compiler_parameters={"representation": "uflacs"})

    if rank == 0:
        print("LV cavity volume: {:.2f} ml".format(lv_vol))
        print("RV cavity volume: {:.2f} ml".format(rv_vol))

    # ============================
    # Step 5: Set Material Regions
    # ============================
    print("Assigning material regions...")
    LVendo, RVendo, Epi = GetSurfaces(directory, "bivfacet000000.vtu", "f", False)
    ugrid = vtk_py.readXMLUGrid(os.path.join(directory, "bivmesh000000.vtu"))
    vtk_py.addRegionsToBiV(ugrid, LVendo, RVendo, Epi)

    matid = dolfin.MeshFunction("size_t", fenics_mesh, 3, fenics_mesh.domains())
    matid_vtk = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray("region_id"))
    matid.array()[:] = matid_vtk
    dolfin.File(os.path.join(directory, "matid.pvd")) << matid

    print("Material regions assigned and saved.")

    # ============================
    # Step 6: Save to HDF5
    # ============================
    print("Saving data to HDF5 file...")
    hdf5_file = os.path.join(directory, meshname + ".hdf5")
    with dolfin.HDF5File(fenics_mesh.mpi_comm(), hdf5_file, "w") as f:
        f.write(fenics_mesh, meshname)
        f.write(fenics_facet, meshname + "/facetboundaries")
        f.write(fenics_edge, meshname + "/edgeboundaries")
        f.write(matid, meshname + "/matid")

    print("HDF5 file saved: {}".format(hdf5_file))

    # Optional: Duplicate for refinement
    refined_file = os.path.join(directory, meshname + "_refine.hdf5")
    os.system('copy "{}" "{}"'.format(hdf5_file, refined_file))  # Windows uses copy
    print("Refined HDF5 file saved: {}".format(refined_file))


# ===========================================
# Helper Function: GetSurfaces
# ===========================================
def GetSurfaces(directory, filebasename, fieldvariable, isparallel):
    filenames = glob.glob(os.path.join(directory, filebasename + "*"))
    filenames.sort()
    filenames = [filename for filename in filenames if filename[-3:] != "pvd"]

    if filenames[0][-4:] == "pvtu" and isparallel:
        ugrid = vtk_py.readXMLPUGrid(filenames[0])
    elif filenames[0][-4:] == ".vtu" and not isparallel:
        ugrid = vtk_py.readXMLUGrid(filenames[0])

    Epi = vtk_py.extractUGridBasedOnThreshold(ugrid, fieldvariable, 1)
    Epi = vtk_py.convertUGridtoPdata(Epi)
    LVendo = vtk_py.extractUGridBasedOnThreshold(ugrid, fieldvariable, 2)
    LVendo = vtk_py.convertUGridtoPdata(LVendo)
    RVendo = vtk_py.extractUGridBasedOnThreshold(ugrid, fieldvariable, 3)
    RVendo = vtk_py.convertUGridtoPdata(RVendo)

    return LVendo, RVendo, Epi


if __name__ == "__main__":
    main()

