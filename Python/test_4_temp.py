from dolfin import *
import dolfin
import os as os
import numpy as np
from petsc4py import PETSc
from forms import Forms
from nsolver import NSolver as NSolver
# LCL change
#from math import *
import math as math  #LCL
import matplotlib.pyplot as plt

no_of_int_points = 24
no_of_x_bins = 49
no_of_transitions = 4
n_array_length = 52 
hsl0 = 1000
time_steps = 400
step_size = 0.0005
Ca_flag = 4

parameters["form_compiler"]["quadrature_degree"]=2
parameters["form_compiler"]["representation"] = "quadrature"
#
#os.system("rm *.pvd")
#os.system("rm *.vtu")
# defining parts of the model where the boundary condition should be applied later
#  where x[0] = 0
class Left(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and abs(x[0]) < tol
#  where x[0] = 10
class Right(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and abs(x[0]-1.0) < tol
#  where x[2] = 0
class Lower(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and abs(x[2]) < tol
#  where x[1] = 0
class Front(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and abs(x[1]) < tol
#  where x[0], x[1] and x[2] = 0
class Fix(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and abs(x[0]) < tol and abs(x[1]) < tol and abs(x[2]) < tol
#
#
mesh = UnitCubeMesh(1,1,1)
#plot(mesh)
#plt.show()
f0 = Constant((1.0, 0.0, 0.0))
s0 = Constant((0.0, 1.0, 0.0))
n0 = Constant((0.0, 0.0, 1.0))
facetboundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
facetboundaries.set_all(0)
left = Left()
right = Right()
fix = Fix()
lower = Lower()
front = Front()
#
left.mark(facetboundaries, 1)
right.mark(facetboundaries, 2)
fix.mark(facetboundaries, 3)
lower.mark(facetboundaries, 4)
front.mark(facetboundaries, 5)
#
ds = dolfin.ds(subdomain_data = facetboundaries)
#
###############################################################################
#
#
isincomp = True#False
N = FacetNormal (mesh)
Press = Expression(("P"), P=0.0, degree=0)
Cparam = Constant(1.0e2)                                                        #??


TF = TensorFunctionSpace(mesh, 'DG', 1)

Velem = VectorElement("Lagrange", tetrahedron, 2, quad_scheme="default")
Velem._quad_scheme = 'default'
Qelem = FiniteElement("Lagrange", tetrahedron, 1, quad_scheme="default")
Qelem._quad_scheme = 'default'
Quadelem = FiniteElement("Quadrature", tetrahedron, degree=2, quad_scheme="default")
Quadelem._quad_scheme = 'default'

W = FunctionSpace(mesh, MixedElement([Velem,Qelem]))
Quad = FunctionSpace(mesh, Quadelem)

Quad_vectorized_Fspace = FunctionSpace(mesh, MixedElement(n_array_length*[Quadelem]))

# assigning BCs
u_D = Expression(("u_D"), u_D = 0.0, degree = 2)
bcleft= DirichletBC(W.sub(0).sub(0), Constant((0.0)), facetboundaries, 1)         # u1 = 0 on left face
bcright= DirichletBC(W.sub(0).sub(0), u_D, facetboundaries, 2)                    
bcfix = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), fix, method="pointwise") # at one vertex u = v = w = 0
bclower= DirichletBC(W.sub(0).sub(2), Constant((0.0)), facetboundaries, 4)        # u3 = 0 on lower face
bcfront= DirichletBC(W.sub(0).sub(1), Constant((0.0)), facetboundaries, 5)        # u2 = 0 on front face
bcs = [bcleft, bcright, bcfix, bclower, bcfront]

du,dp = TrialFunctions(W)
w = Function(W)
dw = TrialFunction(W)
(u,p) = split(w)
(v,q) = TestFunctions(W)
wtest = TestFunction(W)

params= {"mesh": mesh,
     "facetboundaries": facetboundaries,
     "facet_normal": N,
	 "mixedfunctionspace": W,
	 "mixedfunction": w,
     "displacement_variable": u,
     "pressure_variable": p,
	 "fiber": f0,
     "sheet": s0,
     "sheet-normal": n0,
     "C_param": Cparam,
	 "incompressible": isincomp,                                                   
	 "Kappa":Constant(1e5)}

uflforms = Forms(params)

Fmat = uflforms.Fmat()
Cmat = (Fmat.T*Fmat)
Emat = uflforms.Emat()
J = uflforms.J()

n = J*inv(Fmat.T)*N
dx = dolfin.dx(mesh,metadata = {"integration_order":2})

#Ematrix = project(Emat, TF)
Wp = uflforms.PassiveMatSEF()

stress = Function(Quad)

Pactive = stress * as_tensor(f0[i]*f0[j], (i,j))

# Automatic differentiation  #####################################################################################################
F1 = derivative(Wp, w, wtest)*dx
F2 = inner(Pactive, grad(v))*dx
Ftotal = F1 + F2

Jac1 = derivative(F1, w, dw)
Jac2 = derivative(F2, w, dw)
Jac = Jac1 + Jac2
##################################################################################################################################

# Contraction phase
header_file = open("./C++/hs.h","r")
code = header_file.read()
header_file.close()

ext_module = compile_extension_module(code=code, source_directory="C++", sources=["hs.cpp", "mf.cpp", "Ca.cpp", "base_parameters.cpp"],
     additional_system_headers=["petscvec.h"],
     include_dirs=[".", os.path.abspath("C++"),"/usr/include", "./C++"],
     library_dirs = ['/usr/lib/x86_64-linux-gnu'],
     libraries = ['libgsl.a'])

Myosim = ext_module.hs()

_FE_params = {"step_size": step_size};
Myosim.FE_params.update(_FE_params)

_Ca_params = {"Ca_flag": Ca_flag};
Myosim.Ca_params.update(_Ca_params)


y_vec = Function(Quad_vectorized_Fspace)

y_vec_old = np.load("/home/fenics/shared/test_4_temp/y_vec_old.npy")
delta_hsl = np.load("/home/fenics/shared/test_4_temp/delta_hsl.npy")
#delta_hsl = np.round(delta_hsl,10)
#hsl = np.load("/home/fenics/shared/test_4_temp/hsl.npy")
hsl = np.ones(24) * 1001.0
#p_f = np.load("/home/fenics/shared/test_4_temp/p_f.npy")
p_f = np.ones(24) * 0.723647
#cb_f = np.load("/home/fenics/shared/test_4_temp/cb_f.npy")
cb_f = np.ones(24) * 367.288


_Ca_params = {"time_point": 102};
Myosim.Ca_params.update(_Ca_params)

y_vec_new = Myosim.apply_time_step(y_vec_old, delta_hsl, hsl, p_f, cb_f)

stress.vector()[:] = Myosim.Get_cb_force_vec()

cb_f = stress.vector().get_local()[:]

u_D.u_D = 0.002

solve(Ftotal == 0, w, bcs, J = Jac, form_compiler_parameters={"representation":"uflacs"})

hsl = project(sqrt(dot(f0, Cmat*f0))*hsl0, Quad).vector().get_local()[:]

p_f = project(uflforms.Cauchy_fiber(), Quad).vector().get_local()[:]

print hsl
print delta_hsl
print cb_f
print p_f
print (y_vec_new[(no_of_int_points-3)*n_array_length:(no_of_int_points-2)*n_array_length]- y_vec_new[(no_of_int_points-2)*n_array_length:(no_of_int_points-1)*n_array_length])