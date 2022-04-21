# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:15:59 2022

@author: Hossein
"""

from ast import operator
from operator import methodcaller
import os
import json
import pandas as pd
import numpy as np
from dolfin import *

import time
from scipy.integrate import solve_ivp

from protocol import protocol as prot

from .dependencies.recode_dictionary import recode
from .mesh.mesh import MeshClass 
from .circulation.circulation import Circulation as circ
from .heart_rate.heart_rate import heart_rate as hr
from .dependencies.forms import Forms
from .output_handler.output_handler import output_handler as oh
from .baroreflex import baroreflex as br
from .half_sarcomere import half_sarcomere as hs 


from mpi4py import MPI



class LV_simulation():
    """Class for running a LV simulation using FEniCS"""

    def __init__(self,comm, instruction_data):


        # Check for model input first
        if not "model" in  instruction_data:
           return 
        self.instruction_data = instruction_data
        # Create a model dict for things that do not change during a simulation
        self.model = dict()
        # And a data dict for things that might
        self.data = dict()

        # Create the comminicator between cores
        self.comm = comm

        # Define half_sarcomere class to be used in initilizing 
        # function spaces, functions, and week form
        hs_struct = \
            instruction_data['model']['half_sarcomere']
        self.hs = hs.half_sarcomere(hs_struct)
        self.y_vec_length = len(self.hs.myof.y)

        # Initialize and define mesh objects (finite elements, 
        # function spaces, functions)
        mesh_struct = instruction_data['mesh']
        self.mesh = MeshClass(self)

        self.mesh_files = dict()

        self.y_vec = \
            self.mesh.model['functions']['y_vec'].vector().get_local()[:]

        
        """Lets handle dof mapping for quadrature points """
        self.dofmap_list = []
        self.dofmap = self.mesh.model['function_spaces']['quadrature_space'].dofmap().dofs()
        
        # Send dof mapping and list of coords to root core (i.e. 0)
        if self.comm.Get_rank() != 0:
            self.comm.send(self.dofmap,dest = 0, tag = 0)
        else: # Root core recieves  from other cores
            self.dofmap_list.append(self.dofmap)
            for i in range(1,self.comm.Get_size()):
                self.dofmap_list.append(self.comm.recv(source = i, tag = 0))
        # Now broadcast the list to all cores
        self.dofmap_list = \
            self.comm.bcast(self.dofmap_list)

        """ Create a data structure for holding """
        # half_sarcomere parameters spatially 
        # 4 comes from using degree 2
        #self.hs_params_mesh = dict()
        self.local_n_of_int_points = \
            4 * np.shape(self.mesh.model['mesh'].cells())[0]
        
        """ Calculate the total no of integration points"""
        # First on the root core
        self.global_n_of_int_points = \
            self.comm.reduce(self.local_n_of_int_points)
        # Then broadcast to all other cores
        self.global_n_of_int_points = \
            self.comm.bcast(self.global_n_of_int_points)

        """ Now generate a list (with len = total num of cores) """
        # that holds the num of integer points for each core
        self.int_points_per_core = \
                np.zeros(self.comm.Get_size())
        # Send local num of integer points to root core (i.e. 0)
        if self.comm.Get_rank() != 0:
            self.comm.send(self.local_n_of_int_points,dest = 0, tag = 1)
        else: # Root core recieves local num of int points from other cores
            self.int_points_per_core[0] = self.local_n_of_int_points
            for i in range(1,self.comm.Get_size()):
                self.int_points_per_core[i] = \
                    self.comm.recv(source = i, tag = 1)
        # Now broadcast the list to all cores
        self.int_points_per_core = \
            self.comm.bcast(self.int_points_per_core)

        if self.comm.Get_rank() == 1:
            print 'Total no if int points is %0.0f'\
                %self.global_n_of_int_points

        """ Generating arrays for holding half-sarcomere data"""
        # accross the mesh
        # Start with half-saromere length
        self.hs_length_list = self.mesh.hs_length_list
        # Delta half-sarcomere length
        self.delta_hs_length_list = np.zeros(self.local_n_of_int_points)
        # Active stress generated by cross-bridge cycling of myosin
        self.cb_stress_list = self.mesh.cb_stress_list
        # Passive stress in half-sarcomeres
        self.pass_stress_list = self.mesh.pass_stress_list

        """ Generating half-sarcomere object list"""
        self.hs_objs_list = []
        for i in np.arange(self.local_n_of_int_points):
            self.hs_objs_list.append(hs.half_sarcomere(hs_struct))
            #""" Assign the hs length according to what used in the mesh"""
            self.hs_objs_list[-1].data['hs_length'] = \
                self.hs_length_list[i]
        
        
        """ Handle the coordinates of quadrature (integer) points"""
        gdim = self.mesh.model['mesh'].geometry().dim()

        self.coord = self.mesh.model['function_spaces']['quadrature_space'].\
                tabulate_dof_coordinates().reshape((-1, gdim))
        if self.comm.Get_rank()!=0:
                self.comm.send(self.coord,dest=0,tag = 3)
        else:
            for i in range(1,self.comm.Get_size()):
                self.coord = \
                        np.append(self.coord,
                            self.comm.recv(source = i, tag = 3),axis = 0)
        
        self.coord = self.comm.bcast(self.coord)
    
        """Handle the coordinates """
        x_coord = []
        y_coord = []
        z_coord = []
        for i, c in enumerate(self.coord):
            x_coord.append(c[0])
            y_coord.append(c[1])
            z_coord.append(c[2])

        self.x_coord = np.array(x_coord)
        self.y_coord = np.array(y_coord)
        self.z_coord = np.array(z_coord)
        if self.comm.Get_rank() == 0:
            i = np.where(self.z_coord/self.z_coord.max()>0.5)
            ind_to_change = np.isin(self.dofmap,i)
            hs_list = np.array(self.hs_objs_list)
            #for i,j  in enumerate(hs_list[ind_to_change]):
            #   j.myof.data['k_1'] = 10
            #   print self.hs_objs_list[i].myof.data['k_1']



        rank_id = self.comm.Get_rank()
        print '%0.0f integer points have been assigned to core %0.0f'\
             %(self.local_n_of_int_points,rank_id)

        """ Create a circulatory system object"""
        circ_struct = instruction_data['model']['circulation']
        self.circ = circ(circ_struct,self.mesh)

        if self.comm.Get_rank() == 0:
            print self.circ.data['v']

        """ Create a heart-rate object"""
        hr_struct = instruction_data['heart_rate']
        self.hr = hr(hr_struct)
        self.data['heart_rate'] = \
            self.hr.return_heart_rate()
        
        """ Initialize simulation time and counter"""
        self.data['time'] = 0
        self.t_counter = 0

        """ If requried, create the baroreceptor"""
        self.data['baroreflex_active'] = 0
        self.data['baroreflex_setpoint'] = 0
        if ('baroreflex' in instruction_data['model']):
            self.br = br.baroreflex(instruction_data['model']['baroreflex'],
                                    self,
                                    self.circ.data['pressure_arteries'])
        else:
            self.br = []
        # If required, create the growth object
        self.gr = []
        # If required, create the vad object
        self.va = []


        
    def create_data_structure(self,no_of_data_points, frequency = 1):
        """ returns a data frame from the data dicts of each component """

        # First build up the field data
        # Prune some fields from the self_data
        sk = []
        for k in self.circ.data.keys():
            if (k not in ['p','v','s','compliance','resistance','f']):
                sk.append(k)

        data_fields = sk + \
            list(self.data.keys()) + \
            list(self.hr.data.keys()) 
            #list(self.circ.data.keys()) 
            #list(self.hs.data.keys()) + \
            #list(self.hs.memb.data.keys()) + \
            #list(self.hs.myof.data.keys()) + \
            #['write_mode']

        # Add in fields from optional modules
        if (self.br != []):
            data_fields = data_fields + list(self.br.data.keys())
        if (self.gr != [] ):
            data_fields = data_fields + list(self.gr.data.keys())
        if (self.va != []):
            data_fields = data_fields + list(self.va.data.keys())

        # Now start define the data holder
        rows = int(no_of_data_points/frequency) + 1 # 1 for time zero
        #sim_data = pd.DataFrame()
        #z = np.zeros(no_of_data_points)

        sim_data = dict()
        for f in data_fields:
            #print f
            sim_data[f] = np.zeros(rows)
            #s = pd.Series(data=z, name=f)
            #sim_data = pd.concat([sim_data, s], axis=1)

        return sim_data

    def create_data_structure_for_spatial_variables(self,no_of_data_points, 
                                                    num_of_int_points, 
                                                    spatial_data_fields = [],
                                                    in_average = False,
                                                    frequency = 1):
        """ return a data structure for each spatial variables, specially for MyoSim parameters"""

        print 'creating spatial sim data'
        
        rows = int(no_of_data_points/frequency) +1 # 1 for time zero
        i = np.zeros(rows)
        c = np.arange(num_of_int_points)

        data_field = []
        self.spatial_myof_data_fields = []
        self.spatial_memb_data_fields = []
        self.spatial_hs_data_fields = []
        if spatial_data_fields:
            # create data fileds based on what user has asked
            for sd in spatial_data_fields:
                if sd['level'][0] == 'myofilaments':
                    for f in sd['fields']:
                        self.spatial_myof_data_fields.append(f)
                if sd['level'][0] == 'membranes':
                    for f in sd['fields']:
                        self.spatial_memb_data_fields.append(f)
        else:
            # create default data fields
            self.spatial_hs_data_fields = list(self.hs.data.keys())
            self.spatial_myof_data_fields = list(self.hs.myof.data.keys())#['M_SRX','M_DRX','M_FG','n_off','n_on','n_overlap',
                                                #'n_bound']
            self.spatial_memb_data_fields = list(self.hs.memb.data.keys())#['Ca_cytosol','Ca_SR']

        data_field = self.spatial_hs_data_fields +\
                        self.spatial_myof_data_fields+\
                            self.spatial_memb_data_fields
        if in_average:
            spatial_data = pd.DataFrame()
            data_field.append('time')

            for f in data_field:
                s = pd.Series(data=np.zeros(rows), name=f)
                spatial_data = pd.concat([spatial_data, s], axis=1)
                #spatial_data[f]['time'] = pd.Series()
        else:
            spatial_data = dict()
            for f in data_field:
                spatial_data[f] = pd.DataFrame(0,index = i,columns=c)
                #spatial_data[f]['time'] = pd.Series(0)
        print 'spatial simulation data is created'
        
        return spatial_data

    def run_simulation(self,protocol_struct,output_struct=[]):

        self.prot = prot.protocol(protocol_struct)
        
        # First setup the protocol for creating output data holders
        spatial_data_fields = []
        self.spatial_data_to_mean = False
        self.dumping_data_frequency = 1
        if output_struct:
            if 'spatial_data_fileds' in output_struct:
                spatial_data_fields = output_struct['spatial_data_fileds']
            if 'dumping_spatial_in_average' in output_struct:
                if output_struct['dumping_spatial_in_average'][0] == True:
                    self.spatial_data_to_mean = True
            if 'frequency_n' in output_struct:
                self.dumping_data_frequency = \
                    output_struct['frequency_n'][0]


        # Define simulation data holders for storing 
        # 1-D variables (pressure, volume, etc.)
        if self.comm.Get_rank() == 0:
            self.sim_data = \
                    self.create_data_structure(self.prot.data['no_of_time_steps'],
                                                frequency = self.dumping_data_frequency )

        # Now define data holder for spatial variables.
        # Create local data holders for spatial varibles on each core
        self.local_spatial_sim_data = \
            self.create_data_structure_for_spatial_variables(self.prot.data['no_of_time_steps'],
                                                                self.local_n_of_int_points,
                                                                spatial_data_fields = spatial_data_fields,
                                                                in_average = self.spatial_data_to_mean,
                                                                frequency = self.dumping_data_frequency)
        # Create a global data holder for spatial variables 
        # on root core (i.e. 0)
        if self.comm.Get_rank() == 0:
            self.spatial_sim_data = \
                self.create_data_structure_for_spatial_variables(self.prot.data['no_of_time_steps'],
                                                                self.global_n_of_int_points,
                                                                spatial_data_fields = spatial_data_fields,
                                                                in_average = self.spatial_data_to_mean,
                                                                frequency = self.dumping_data_frequency)
        # Step through the simulation
        self.t_counter = 0
        self.write_counter = 0
        self.envelope_counter = 0

        # Initilize the output mesh files if any
        self.total_disp_file = [] 
        self.output_data_str = [] 
        self.mesh_obj_to_save = []
        if output_struct:
            if 'mesh_output_path' in output_struct:
                mesh_out_path = output_struct['mesh_output_path'][0]
                # Cehck the output path
                if self.comm.Get_rank() == 0:
                    self.check_output_directory_folder(path = mesh_out_path)
                
                if "mesh_object_to_save" in output_struct:
                    print 'mesh obj is defined'
                    self.mesh_obj_to_save = output_struct['mesh_object_to_save']
                    # start creating file for mesh objects
                    file_path = os.path.join(mesh_out_path,'solution.xdmf') 
                    self.solution_mesh = XDMFFile(mpi_comm_world(),file_path)
                    self.solution_mesh.parameters.update({"functions_share_mesh": True,
                                            "rewrite_function_mesh": False})
                
                    for m in self.mesh_obj_to_save:
                        
                        if m == 'displacement':
                            temp_obj = self.mesh.model['functions']['w'].sub(0)
                        if m == 'hs_length':
                            temp_obj = project(self.mesh.model['functions']['hsl'], 
                                                self.mesh.model['function_spaces']["scaler"])
                        if m == 'active_stress':
                            temp_obj = project(inner(self.mesh.model['functions']['f0'],
                                        self.mesh.model['functions']['Pactive']*
                                        self.mesh.model['functions']['f0']),
                                        self.mesh.model['function_spaces']["scaler"])
                        if m == 'fiber_direction':
                            temp_obj = project(self.mesh.model['functions']['f0'],
                                        self.mesh.model['function_spaces']['vector_f'])

                        temp_obj.rename(m,'')
                        self.solution_mesh.write(temp_obj,0)

            if 'output_data_path' in output_struct:
                self.output_data_str = output_struct['output_data_path'][0]
                if self.comm.Get_rank() == 0: 
                    self.check_output_directory_folder(path = self.output_data_str)

        for i in np.arange(self.prot.data['no_of_time_steps']+1):
            try:
                self.implement_time_step(self.prot.data['time_step'])
            except RuntimeError: 
                print "RuntimeError happend"
                self.handle_output(output_struct)
                return

        # Now build up global data holders for 
        # spatial variables if multiple cores have been used
        self.handle_output(output_struct)
       

    def implement_time_step(self, time_step):
        """ Implements time step """
        
        if self.comm.Get_rank() == 0:
            print '******** NEW TIME STEP ********'
            print (self.data['time'])

            if (self.t_counter % 10 == 0):
                print('Sim time (s): %.0f  %.0f%% complete' %
                    (self.data['time'],
                    100*self.t_counter/self.prot.data['no_of_time_steps']))

                vol, press, flow = self.return_system_values()
                
                print(json.dumps(vol, indent=4))
                print(json.dumps(press, indent=4))
                print(json.dumps(flow, indent=4))

        # Check for baroreflex and implement
        if (self.br):
            self.data['baroreflex_active'] = 0
            for b in self.prot.baro_activations:
                if ((self.t_counter >= b.data['t_start_ind']) and
                        (self.t_counter < b.data['t_stop_ind'])):
                    self.data['baroreflex_active'] = 1

            self.br.implement_time_step(self.circ.data['pressure_arteries'],
                                        time_step,
                                        reflex_active=
                                        self.data['baroreflex_active'])
        # check for any perturbation
        for p in self.prot.perturbations:
            if (self.t_counter >= p.data['t_start_ind'] and 
                self.t_counter < p.data['t_stop_ind']):
                if p.data['level'] == 'circulation':
                    self.circ.data[p.data['variable']] += \
                        p.data['increment']
                elif p.data['level'] == 'baroreflex':
                    self.br.data[p.data['variable']] += \
                        p.data['increment']
                elif p.data['level'] == 'myofilaments':
                    for j in range(self.local_n_of_int_points):
                        self.hs_objs_list[j].myof.data[p.data['variable']] +=\
                            p.data['increment']
                elif p.data['level'] == 'membranes':
                    for j in range(self.local_n_of_int_points):
                        self.hs_objs_list[j].memb.data[p.data['variable']] +=\
                            p.data['increment']

        # Proceed time
        (activation, new_beat) = \
            self.hr.implement_time_step(time_step)

        if self.comm.Get_rank() == 0:
            # Solve MyoSim ODEs across the mesh
            print 'Solving MyoSim ODEs across the mesh'
        start = time.time()
        for j in range(self.local_n_of_int_points):
        
            self.hs_objs_list[j].update_simulation(time_step, 
                                                self.delta_hs_length_list[j], 
                                                activation,
                                                self.cb_stress_list[j],
                                                self.pass_stress_list[j])
            self.hs_objs_list[j].update_data()
            
            
            
            if j%1000==0 and self.comm.Get_rank() == 0:
                print '%.0f%% of integer points are updated' % (100*j/self.local_n_of_int_points)
            self.y_vec[j*self.y_vec_length+np.arange(self.y_vec_length)]= \
                self.hs_objs_list[j].myof.y[:]
        end =time.time()

        if self.comm.Get_rank() == 0:
            print 'Required time for solving myosim was'
            t = end-start 
            print t

        # Now update fenics FE for population array (y_vec) and hs_length
        self.mesh.model['functions']['y_vec'].vector()[:] = self.y_vec
        self.mesh.model['functions']['hsl_old'].vector()[:] = self.hs_length_list

        # Update circulation and FE function for LV cavity volume
        self.circ.data['v'] = \
                self.circ.evolve_volume(time_step, self.circ.data['v'])

        # Update LV cavity volume fenics function        
        self.mesh.model['functions']['LVCavityvol'].vol = \
            self.circ.data['v'][-1]

        #Solve cardiac mechanics weak form
        #--------------------------------
        if self.comm.Get_rank() == 0:
            print 'solving weak form'
        Ftotal = self.mesh.model['Ftotal']
        w = self.mesh.model['functions']['w']
        bcs = self.mesh.model['boundary_conditions']
        Jac = self.mesh.model['Jac']

        solve(Ftotal == 0, w, bcs, J = Jac, form_compiler_parameters={"representation":"uflacs"})

        self.mesh.model['functions']['w'] = w
        # Start updating variables after solving the weak form 

        # First pressure in circulation
        for i in range(self.circ.model['no_of_compartments']-1):
            self.circ.data['p'][i] = (self.circ.data['v'][i] - self.circ.data['s'][i]) / \
                    self.circ.data['compliance'][i]
        # 0.0075 is for converting to mm Hg
        self.circ.data['p'][-1] = \
                0.0075*self.mesh.model['uflforms'].LVcavitypressure()

        # Then update FE function for cross-bridge stress, hs_length, and passive stress
        # across the mesh
        self.cb_stress_list = project(self.mesh.model['functions']['cb_stress'],
                                self.mesh.model['function_spaces']['quadrature_space']).vector().get_local()[:]

        self.mesh.model['functions']['hsl_old'].vector()[:] = \
            project(self.mesh.model['functions']['hsl'], self.mesh.model['function_spaces']["quadrature_space"]).vector().get_local()[:]

        self.mesh.model['functions']['pseudo_old'].vector()[:] = \
            project(self.mesh.model['functions']['pseudo_alpha'], self.mesh.model['function_spaces']["quadrature_space"]).vector().get_local()[:]

        new_hs_length_list = \
            project(self.mesh.model['functions']['hsl'], self.mesh.model['function_spaces']["quadrature_space"]).vector().get_local()[:]

        self.delta_hs_length_list = new_hs_length_list - self.hs_length_list
        self.hs_length_list = new_hs_length_list
        
        temp_DG = project(self.mesh.model['functions']['Sff'], 
                    FunctionSpace(self.mesh.model['mesh'], "DG", 1), 
                    form_compiler_parameters={"representation":"uflacs"})

        p_f = interpolate(temp_DG, self.mesh.model['function_spaces']["quadrature_space"])
        self.pass_stress_list = p_f.vector().get_local()[:]
        
        # Convert negative passive stress in half-sarcomeres to 0
        self.pass_stress_list[self.pass_stress_list<0] = 0
    
        self.comm.Barrier()
        # Update sim data for non-spatial variables on root core (i.e. 0)

        self.update_data(time_step)
        if self.t_counter%self.dumping_data_frequency == 0:
            print 'Dumping data ...'
            if self.comm.Get_rank() == 0:
                self.write_complete_data_to_sim_data()

            # Now update local spatial data for each core
            self.write_complete_data_to_spatial_sim_data(self.comm.Get_rank())

            self.write_counter = self.write_counter + 1

            # save data on mesh
            if self.mesh_obj_to_save:
                print 'Saving to 3d mesh'
                for m in self.mesh_obj_to_save:
                    if m == 'displacement':
                        temp_obj = self.mesh.model['functions']['w'].sub(0)
                    if m == 'hs_length':
                        temp_obj = project(self.mesh.model['functions']['hsl'], 
                                                self.mesh.model['function_spaces']["scaler"])
                    if m == 'active_stress':
                        temp_obj = project(inner(self.mesh.model['functions']['f0'],
                                        self.mesh.model['functions']['Pactive']*
                                        self.mesh.model['functions']['f0']),
                                        self.mesh.model['function_spaces']["scaler"])
                    if m == 'fiber_direction':
                            temp_obj = project(self.mesh.model['functions']['f0'],
                                        self.mesh.model['function_spaces']['vector_f'])

                    temp_obj.rename(m,'')
                    self.solution_mesh.write(temp_obj,self.data['time'])

        # Update the t counter for the next step
        self.t_counter = self.t_counter + 1
        self.data['time'] = self.data['time'] + time_step
   
    def update_data(self, time_step):
        """ Update data after a time step """

        # Update data for the heart-rate
        self.data['heart_rate'] = self.hr.return_heart_rate()

        
        self.circ.updata_data(time_step)

    def return_system_values(self, time_interval=0.01):
        d = dict()
        vol = dict()
        pres = dict()
        flow = dict()
        vol['volume_ventricle'] = self.circ.data['v'][-1]
        vol['volume_aorta'] = self.circ.data['v'][0]
        vol['volume_arteries'] = self.circ.data['v'][1]
        vol['volume_arterioles'] = self.circ.data['v'][2]
        vol['volume_capillaries'] = self.circ.data['v'][3]
        vol['volume_venules'] = self.circ.data['v'][4]
        vol['volume_veins'] = self.circ.data['v'][5]

        pres['pressure_ventricle'] = self.circ.data['p'][-1]
        pres['pressure_aorta'] = self.circ.data['p'][0]
        pres['pressure_arteries'] = self.circ.data['p'][1]
        pres['pressure_arterioles'] = self.circ.data['p'][2]
        pres['pressure_capillaries'] = self.circ.data['p'][3]
        pres['pressure_venules'] = self.circ.data['p'][4]
        pres['pressure_veins'] = self.circ.data['p'][5]

        flow['flow_ventricle_to_aorta'] = self.circ.data['f'][0]
        flow['flow_aorta_to_arteries'] = self.circ.data['f'][1]
        flow['flow_arteries_to_arterioles'] = self.circ.data['f'][2]
        flow['flow_arterioles_to_capillaries'] = self.circ.data['f'][3]
        flow['flow_capillaries_to_venules'] = self.circ.data['f'][4]
        flow['flow_venules_to_veins'] = self.circ.data['f'][5]
        flow['flow_veins_to_ventricle'] = self.circ.data['f'][6]
        

        """if (self.data['time'] > time_interval):
            self.temp_data = \
                self.sim_data[self.sim_data['time'].between(
                    self.data['time']-time_interval, self.data['time'])]

            d['volume_ventricle_max'] = \
                self.temp_data['volume_ventricle'].max()
            d['stroke_volume'] = d['volume_ventricle_max'] - \
                self.temp_data['volume_ventricle'].min()
            d['pressure_ventricle'] = self.temp_data['pressure_ventricle'].mean()
            #d['ejection_fraction'] = self.temp_data['ejection_fraction'].mean()
            d['heart_rate'] = self.data['heart_rate']
            d['cardiac_output'] = d['stroke_volume'] * d['heart_rate']"""
           
            
        return vol, pres,flow

    def write_complete_data_to_sim_data(self):
        """ Writes full data to data frame """


        for f in list(self.data.keys()):
            self.sim_data[f][self.write_counter] = self.data[f]
        for f in list(self.circ.data.keys()):
            if (f not in ['p', 'v', 's', 'compliance', 'resistance',
                            'inertance', 'f']):
                self.sim_data[f][self.write_counter] = self.circ.data[f]
        for f in list(self.hr.data.keys()):
            self.sim_data[f][self.write_counter] = self.hr.data[f]

        if (self.br):
            for f in list(self.br.data.keys()):
                self.sim_data[f][self.write_counter] = self.br.data[f]
        if (self.gr):
            for f in list(self.gr.data.keys()):
                self.sim_data[f][self.write_counter] = self.gr.data[f]
    
        self.sim_data['write_mode'] = 1
        

    def write_complete_data_to_spatial_sim_data(self,rank):

        print 'Writing spatial variables on core id: %0.0f' %rank

        if self.spatial_data_to_mean:
            self.local_spatial_sim_data.at[self.write_counter,'time'] = \
                self.data['time']
            for f in list(self.spatial_hs_data_fields):
                data_field = []
                for h in self.hs_objs_list:
                    data_field.append(h.data[f]) 
                self.local_spatial_sim_data.at[self.write_counter,f] = np.mean(data_field)

            for f in list( self.spatial_myof_data_fields):
                data_field = []
                for h in self.hs_objs_list:
                    data_field.append(h.myof.data[f]) 
                if f == 'k_1':
                    print 'rank is'
                    print self.comm.Get_rank()
                    print np.array(data_field)
                self.local_spatial_sim_data.at[self.write_counter,f] = np.mean(data_field)

            for f in list(self.spatial_memb_data_fields):
                data_field = []
                for h in self.hs_objs_list:
                    data_field.append(h.memb.data[f]) 
                self.local_spatial_sim_data.at[self.write_counter,f] = np.mean(data_field)
        else:
            for f in self.spatial_hs_data_fields:
                data_field = []
                for h in self.hs_objs_list:
                    data_field.append(h.data[f])
                self.local_spatial_sim_data[f].iloc[self.write_counter] = data_field
                #self.local_spatial_sim_data[f].at[self.write_counter,'time'] = self.data['time']

            for f in self.spatial_myof_data_fields:
                data_field = []
                for h in (self.hs_objs_list):
                    data_field.append(h.myof.data[f])
                self.local_spatial_sim_data[f].iloc[self.write_counter] = data_field
                #self.local_spatial_sim_data[f].at[self.write_counter,'time'] = self.data['time']
            
            for f in self.spatial_memb_data_fields:
                data_field = []
                for h in (self.hs_objs_list):
                    data_field.append(h.memb.data[f])
                self.local_spatial_sim_data[f].iloc[self.write_counter] = data_field
                #self.local_spatial_sim_data[f].at[self.write_counter,'time'] = self.data['time']

    def check_output_directory_folder(self, path=""):
        """ Check output folder"""
        output_dir = os.path.dirname(path)
        print('output_dir %s' % output_dir)
        if not os.path.isdir(output_dir):
            print('Making output dir')
            os.makedirs(output_dir)

    def handle_output(self, output_struct):
        """ Handle output data"""
        if self.comm.Get_size() > 1:
            # first send all local spatial data to root core (i.e. 0)
            if self.comm.Get_rank() != 0 :
                self.comm.send(self.local_spatial_sim_data,dest = 0,tag = 2)

           # let root core recieve them
            if self.comm.Get_rank() == 0:
                temp_data_holders = []
                temp_data_holders.append(self.local_spatial_sim_data)
                # recieve local data from others 
                for i in range(1,self.comm.Get_size()):
                    temp_data_holders.append(self.comm.recv(source = i, tag = 2))
                # now dump them to global data holders
                print 'Spatial variables are being gathered from multiple computing cores'
                if self.spatial_data_to_mean:
                    for c in self.spatial_sim_data.columns:
                        self.spatial_sim_data[c] = \
                            sum([temp_data_holders[i][c]*self.int_points_per_core[i] for i \
                                in range(len(self.int_points_per_core))])/np.sum(self.int_points_per_core)
                else:
                    for j,f in enumerate(list(self.spatial_sim_data.keys())):
                        print '%.0f%% complete' %(100*j/len(list(self.spatial_sim_data.keys())))
                        for id in range(0,self.comm.Get_size()):
                            #i_0 = np.sum(self.int_points_per_core[0:id])
                            #i_1 = i_0 + self.int_points_per_core[id]
                            #cols = np.arange(i_0,i_1)
                            cols = self.dofmap_list[id]
                            self.spatial_sim_data[f][cols] = \
                                temp_data_holders[id][f]
                        
                        self.spatial_sim_data[f]['time'] = self.sim_data['time']

        else:
            self.spatial_sim_data = self.local_spatial_sim_data
        # Now save output data
        # Things to improve: 1) Different data format (e.g. csv, hdf5, etc)
        # 2) Store data at a specified resolution (e.g. every 100 time steps)
        if output_struct and self.comm.Get_rank() == 0:
            if self.output_data_str:
                output_sim_data = pd.DataFrame(data = self.sim_data)
                output_sim_data.to_csv(self.output_data_str)
                #self.sim_data.to_csv(self.output_data_str)

                output_dir = os.path.dirname(self.output_data_str)
                if self.spatial_data_to_mean:
                    out_path = output_dir + '/' + 'spatial_data.csv'
                    self.spatial_sim_data.to_csv(out_path)
                else:
                    for f in list(self.spatial_sim_data.keys()):
                        out_path = output_dir + '/' + f + '_data.csv'
                        self.spatial_sim_data[f].to_csv(out_path)

        return 

        
