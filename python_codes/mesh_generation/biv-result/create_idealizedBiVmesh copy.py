
sys.path.append(r"C:/Users/sba431/Github/MyoFE/python_codes/mesh_generation/biv-result/vtk_py")

import os
import vtk
import dolfin
from mpi4py import MPI as pyMPI
from vtk.util import numpy_support
from vtk_py.readUGrid import readUGrid
from vtk_py.rotateUGrid import rotateUGrid
from vtk_py.extractFeNiCsBiVFacet import extractFeNiCsBiVFacet
from vtk_py.writeUGrid import writeUGrid

def generate_mesh(geo_file, gmsh_path, output_vtk):
    """
    Generate mesh using GMSH from a .geo file.
    """
    if not os.path.exists(geo_file):
        raise IOError(f"{geo_file} not found. Ensure the .geo file exists.")

    gmsh_command = f'"{gmsh_path}" -3 "{geo_file}" -o "{output_vtk}"'
    os.system(gmsh_command)

    if not os.path.exists(output_vtk):
        raise RuntimeError(f"Mesh generation failed. {output_vtk} not created.")

    print(f"Mesh generated: {output_vtk}")

def process_mesh(input_vtk, output_rotated_vtk):
    """
    Process the generated VTK mesh, rotate it, and save the rotated version.
    """
    ugrid = readUGrid(input_vtk)
    ugrid_rot = rotateUGrid(ugrid, rx=0.0, ry=-90.0, rz=0.0, sx=1.0, sy=1.0, sz=1.0)

    # Reduce point precision
    newpts = vtk.vtkPoints()
    for p in range(ugrid_rot.GetNumberOfPoints()):
        pt = [round(ugrid_rot.GetPoints().GetPoint(p)[k], 5) for k in range(3)]
        newpts.InsertNextPoint(pt)
    ugrid_rot.SetPoints(newpts)

    writeUGrid(ugrid_rot, output_rotated_vtk)
    print(f"Rotated mesh saved: {output_rotated_vtk}")

def extract_fenics_mesh(rotated_vtk, output_dir, meshname):
    """
    Extract FEniCS-compatible mesh components.
    """
    ugrid = readUGrid(rotated_vtk)
    fenics_mesh, fenics_facet, fenics_edge = extractFeNiCsBiVFacet(ugrid)

    dolfin.File(os.path.join(output_dir, "bivmesh.pvd")) << fenics_mesh
    dolfin.File(os.path.join(output_dir, "bivfacet.pvd")) << fenics_facet
    dolfin.File(os.path.join(output_dir, "bivedge.pvd")) << fenics_edge
    print("FEniCS mesh files saved.")

    return fenics_mesh, fenics_facet, fenics_edge

def calculate_volumes(fenics_mesh, fenics_facet):
    """
    Calculate LV and RV cavity volumes.
    """
    X = dolfin.SpatialCoordinate(fenics_mesh)
    N = dolfin.FacetNormal(fenics_mesh)
    ds = dolfin.ds(subdomain_data=fenics_facet)

    lv_vol_form = -dolfin.Constant(1.0 / 3.0) * dolfin.inner(N, X) * ds(2)
    rv_vol_form = -dolfin.Constant(1.0 / 3.0) * dolfin.inner(N, X) * ds(3)

    lv_vol = dolfin.assemble(lv_vol_form, form_compiler_parameters={"representation": "uflacs"})
    rv_vol = dolfin.assemble(rv_vol_form, form_compiler_parameters={"representation": "uflacs"})

    print(f"LV cavity volume: {lv_vol:.2f} ml")
    print(f"RV cavity volume: {rv_vol:.2f} ml")

def save_to_hdf5(fenics_mesh, fenics_facet, fenics_edge, output_file, meshname):
    """
    Save FEniCS mesh and associated data to HDF5.
    """
    with dolfin.HDF5File(fenics_mesh.mpi_comm(), output_file, "w") as f:
        f.write(fenics_mesh, meshname)
        f.write(fenics_facet, f"{meshname}/facetboundaries")
        f.write(fenics_edge, f"{meshname}/edgeboundaries")
    print(f"HDF5 file saved: {output_file}")

def main():
    gmsh_path = r"C:/Users/sba431/Github/MyoFE/python_codes/mesh_generation/biv-result/gmsh-4.13.1-Windows64/gmsh.exe"
    directory = r"C:/Users/sba431/Github/MyoFE/python_codes/mesh_generation/biv-result"
    meshname = "biv_idealized3_generalized"

    geo_file = os.path.join(directory, meshname + ".geo")
    vtk_file = os.path.join(directory, meshname + ".vtk")
    rotated_file = os.path.join(directory, meshname + "_rot.vtk")
    hdf5_file = os.path.join(directory, meshname + ".hdf5")

    # Step 1: Generate Mesh
    print("Step 1: Generating mesh...")
    generate_mesh(geo_file, gmsh_path, vtk_file)

    # Step 2: Process the Mesh
    print("Step 2: Processing mesh...")
    process_mesh(vtk_file, rotated_file)

    # Step 3: Extract FEniCS Mesh
    print("Step 3: Extracting FEniCS mesh...")
    fenics_mesh, fenics_facet, fenics_edge = extract_fenics_mesh(rotated_file, directory, meshname)

    # Step 4: Calculate Volumes
    print("Step 4: Calculating volumes...")
    calculate_volumes(fenics_mesh, fenics_facet)

    # Step 5: Save to HDF5
    print("Step 5: Saving to HDF5...")
    save_to_hdf5(fenics_mesh, fenics_facet, fenics_edge, hdf5_file, meshname)

if __name__ == "__main__":
    main()
