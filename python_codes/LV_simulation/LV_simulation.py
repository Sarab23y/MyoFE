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

        self.y_vec = \
            self.mesh.model['functions']['y_vec'].vector().get_local()[:]

        # Create a data structure for holding 
        # half_sarcomere parameters spatially 
        # 4 comes from using degree 2
        self.hs_params_mesh = dict()
        self.local_n_of_int_points = \
            4 * np.shape(self.mesh.model['mesh'].cells())[0]

        # Calculate the total no of integration points
        # First on the root core
        self.global_n_of_int_points = \
            self.comm.reduce(self.local_n_of_int_points)
        # Then broadcast to all other cores
        self.global_n_of_int_points = \
            self.comm.bcast(self.global_n_of_int_points)

        if self.comm.Get_rank() == 1:
            print 'Total no if int points is %0.0f'\
                %self.global_n_of_int_points

        rank_id = self.comm.Get_rank()
        print '%0.0f integer points have been assigned to core %0.0f'\
             %(self.local_n_of_int_points,rank_id)

        # Generating half-sarcomere object list
        self.hs_objs_list = []
        for i in np.arange(self.local_n_of_int_points):
            self.hs_objs_list.append(hs.half_sarcomere(hs_struct))

        # Generating arrays for holding half-sarcomere data
        # accross the mesh
        # Start with half-saromere length
        self.hs_length_list = self.mesh.hs_length_list
        # Delta half-sarcomere length
        self.delta_hs_length_list = np.zeros(self.local_n_of_int_points)
        # Active stress generated by cross-bridge cycling of myosin
        self.cb_stress_list = self.mesh.cb_stress_list
        # Passive stress in half-sarcomeres
        self.pass_stress_list = self.mesh.pass_stress_list

        # Create a circulatory system object
        circ_struct = instruction_data['model']['circulation']
        self.circ = circ(circ_struct,self.mesh)

        # Create a heart-rate object
        hr_struct = instruction_data['heart_rate']
        self.hr = hr(hr_struct)
        self.data['heart_rate'] = \
            self.hr.return_heart_rate()
        
        # Initialize simulation time and counter
        self.data['time'] = 0
        self.t_counter = 0

        # If requried, create the baroreceptor
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

        
    def create_data_structure(self,no_of_data_points):
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
        #sim_data = pd.DataFrame()
        #z = np.zeros(no_of_data_points)

        sim_data = dict()
        for f in data_fields:
            #print f
            sim_data[f] = np.zeros(no_of_data_points)
            #s = pd.Series(data=z, name=f)
            #sim_data = pd.concat([sim_data, s], axis=1)

        return sim_data

    def create_data_structure_for_spatial_variables(self,no_of_data_points, local_n_of_int_points, spatial_data_struct = []):
        """ return a data structure for each spatial variables, specially for MyoSim parameters"""

        print 'creating spatial sim data'
        
        
        i = np.zeros(no_of_data_points)
        c = np.arange(local_n_of_int_points)+1

        data_field = []
        self.spatial_myof_data_fields = []
        self.spatial_memb_data_fields = []
        self.spatial_hs_data_fields = []
        if spatial_data_struct:
            # create data fileds based on what user has asked
            for sd in spatial_data_struct:
                if sd['level'][0] == 'myofilaments':
                    for f in sd['fields']:
                        self.spatial_myof_data_fields.append(f)
                if sd['level'][0] == 'membranes':
                    for f in sd['fields']:
                        self.spatial_memb_data_fields.append(f)


        else:
            # create default data fields
            self.spatial_myof_data_fields = ['M_SRX','M_DRX','M_FG','n_off','n_on','n_overlap',
                                                'n_bound']
            self.spatial_memb_data_fields = ['Ca_cytosol','Ca_SR']

        data_field = list(self.hs.data.keys()) +\
                        self.spatial_myof_data_fields+\
                            self.spatial_memb_data_fields
        if self.spatial_data_to_mean:
            spatial_data = pd.DataFrame()
            data_field += 'time'
            for f in data_field:
                s = pd.Series(data=np.zeros(no_of_data_points), name=f)
                spatial_data = pd.concat([spatial_data, s], axis=1)
                #spatial_data[f]['time'] = pd.Series()
        else:
            spatial_data = dict()
            for f in data_field:
                spatial_data[f] = pd.DataFrame(0,index = i,columns=c)
                #spatial_data[f]['time'] = pd.Series()
        print 'spatial sim data is created'
        
        return spatial_data

    def run_simulation(self,protocol_struct,output_struct=[]):

        self.prot = prot.protocol(protocol_struct)

        """# Manually deform ventricle to slack volume 
        LV_vol_1 = self.circ.data['v'][-1]
        #self.mesh.diastolic_filling(LV_vol_1,loading_steps=5)
        print('initial vol')
        print(self.mesh.model['functions']['LVCavityvol'].vol)
        print self.mesh.model['uflforms'].LVcavityvol()

        print 'initial pressure'
        print self.mesh.model['functions']['Press'].P
        print self.mesh.model['uflforms'].LVcavitypressure()

        print 'circ pressure'
        print self.circ.data['p'][-1]
        print self.circ.data['pressure_ventricle']

        #self.mesh.model['functions'] = \
        #    self.mesh.diastolic_filling(LV_vol=LV_vol,loading_steps=n)"""

        # first calculate total number of integer points

        # Define data holders only on root core
        if self.comm.Get_rank() == 0:
            self.sim_data = \
                    self.create_data_structure(self.prot.data['no_of_time_steps'])

        spatial_data_struct = []
        self.spatial_data_to_mean = False
        if output_struct:
            if 'spatial_data_fileds' in output_struct:
                spatial_data_struct = output_struct['spatial_data_fileds']
            if 'dumping_spatial_in_average' in output_struct:
                if output_struct['dumping_spatial_in_average'][0] == True:
                    self.spatial_data_to_mean = True

        self.spatial_sim_data = \
            self.create_data_structure_for_spatial_variables(self.prot.data['no_of_time_steps'],
                                                                self.local_n_of_int_points,
                                                                spatial_data_struct = spatial_data_struct)
        # Step through the simulation
        self.t_counter = 0
        self.write_counter = 0
        self.envelope_counter = 0

        # Initilize the output files if any
        self.total_file_disp = [] 
        self.output_data_str = [] 
        if output_struct:
            if 'output_mesh_file' in output_struct:
                print 'initializing '
                output_mesh_str = output_struct['output_mesh_file'][0]
                self.check_output_directory_folder(path = output_mesh_str)
                self.total_file_disp = XDMFFile(mpi_comm_world(),output_mesh_str)

            if 'output_data_path' in output_struct:
                self.output_data_str = output_struct['output_data_path'][0]
                self.check_output_directory_folder(path = self.output_data_str)



        for i in np.arange(self.prot.data['no_of_time_steps']):
            if self.total_file_disp:
                #with XDMFFile(mpi_comm_world(),self.total_file_disp,'w') as out:
                #    out.write_function(self.mesh.model['functions']['w'].sub(0))
                self.total_file_disp.write(self.mesh.model['functions']['w'].sub(0))
                #self.total_file_disp << self.mesh.model['functions']['w'].sub(0)

            self.implement_time_step(self.prot.data['time_step'])
            """try:
                if self.total_file_disp:
                    self.total_file_disp << self.mesh.model['functions']['w'].sub(0)

                self.implement_time_step(self.prot.data['time_step'])
            except RuntimeError: 
                print "RuntimeError happend"
                if output_struct:
                    if self.output_data_str:
                        self.sim_data.to_csv(self.output_data_str)

                        output_dir = os.path.dirname(self.output_data_str)
                        if self.spatial_data_to_mean:
                            out_path = output_dir + '/' + 'spatial_data.csv'
                            self.spatial_sim_data.to_csv(out_path)
                        else:
                            for f in list(self.spatial_sim_data.keys()):
                                out_path = output_dir + '/' + f + '_data.csv'
                                self.spatial_sim_data[f].to_csv(out_path)

                return"""

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
            


    def implement_time_step(self, time_step):
        """ Implements time step """
        self.data['time'] = self.data['time'] + time_step
        if self.comm.Get_rank() == 0:
            print '******** NEW TIME STEP ********'
            print (self.data['time'])

            if (self.t_counter % 100 == 0):
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

        # Start updating variables after solving the weak form 

        # First pressure in circulation
        for i in range(self.circ.model['no_of_compartments']-1):
            self.circ.data['p'][i] = (self.circ.data['v'][i] - self.circ.data['s'][i]) / \
                    self.circ.data['compliance'][i]

        self.circ.data['p'][-1] = \
                self.mesh.model['uflforms'].LVcavitypressure()

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
    
        # Update the t counter for the next step
        self.t_counter = self.t_counter + 1

        self.comm.Barrier()
        if self.comm.Get_rank() == 0:
            print 'Dumping data ...'
            self.update_data(time_step)
            self.write_complete_data_to_sim_data()


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
        
        # This works but is very slow
        """if (True):
            for f in list(self.data.keys()):

                if f not in ['hs_length_list','delta_hs_length_list']:
                    self.sim_data.at[self.write_counter, f] = self.data[f]
            for f in list(self.circ.data.keys()):
                if (f not in ['p', 'v', 's', 'compliance', 'resistance',
                            'inertance', 'f']):
                    self.sim_data.at[self.write_counter, f] = self.circ.data[f]
            for f in list(self.hr.data.keys()):
                self.sim_data.at[self.write_counter, f] = self.hr.data[f]

            #for f in list(self.hs.memb.data.keys()):
            #   self.sim_data.at[self.write_counter, f] = self.hs.memb.data[f]
            

            if (self.br):
                for f in list(self.br.data.keys()):
                    self.sim_data.at[self.write_counter, f] = self.br.data[f]
            if (self.gr):
                for f in list(self.gr.data.keys()):
                    self.sim_data.at[self.write_counter, f] = self.gr.data[f]
            self.sim_data.at[self.write_counter, 'write_mode'] = 1

            if self.spatial_data_to_mean:
                self.spatial_sim_data.at[self.write_counter,'time'] = \
                    self.data['time']
                for f in list(self.hs.data.keys()):
                    data_field = []
                    for h in self.hs_objs_list:
                        data_field.append(h.data[f]) 
                    self.spatial_sim_data.at[self.write_counter,f] = np.mean(data_field)

                for f in list(self.hs.myof.data.keys()):
                    data_field = []
                    for h in self.hs_objs_list:
                        data_field.append(h.myof.data[f]) 
                    self.spatial_sim_data.at[self.write_counter,f] = np.mean(data_field)

                for f in list(self.hs.memb.data.keys()):
                    data_field = []
                    for h in self.hs_objs_list:
                        data_field.append(h.memb.data[f]) 
                    self.spatial_sim_data.at[self.write_counter,f] = np.mean(data_field)
            else:
                
                self.write_complete_data_to_spatial_sim_data()"""
        self.write_counter = self.write_counter + 1

    def write_complete_data_to_spatial_sim_data(self):

        print 'writing to spatial sim data'
       
        for f in list(self.hs.data.keys()):
            data_field = []
            for h in self.hs_objs_list:
                data_field.append(h.data[f])
            self.spatial_sim_data[f].iloc[self.write_counter] = data_field
            #self.spatial_sim_data[f].at[self.write_counter,'time'] = self.data['time']

        for f in self.spatial_myof_data_fields:
            data_field = []
            for h in (self.hs_objs_list):
                data_field.append(h.myof.data[f])
            self.spatial_sim_data[f].iloc[self.write_counter] = data_field
            #self.spatial_sim_data[f].at[self.write_counter,'time'] = self.data['time']
        
        for f in self.spatial_memb_data_fields:
            data_field = []
            for h in (self.hs_objs_list):
                data_field.append(h.memb.data[f])
            self.spatial_sim_data[f].iloc[self.write_counter] = data_field
            #self.spatial_sim_data[f].at[self.write_counter,'time'] = self.data['time']

    def check_output_directory_folder(self, path=""):
        """ Check output folder"""
        output_dir = os.path.dirname(path)
        print('output_dir %s' % output_dir)
        if not os.path.isdir(output_dir):
            print('Making output dir')
            os.makedirs(output_dir)

    def evolve_volume(self,time_step,initial_v):
        
        def derivs(t, v):
            dv = np.zeros(self.circ.model['no_of_compartments'])
            flows = self.return_flows(v, time_step)
            for i in np.arange(self.circ.model['no_of_compartments']):
                dv[i] = flows[i] - flows[i+1]
                if (i == (self.circ.model['no_of_compartments']-1)):
                    dv[i] = flows[i] - flows[0] + flows[-1]
                else:
                    dv[i] = flows[i] - flows[i+1]
            return dv

        sol = solve_ivp(derivs, [0, time_step], initial_v)

        # Tidy up negative values
        y = sol.y[:, -1]

        return y 

    def return_flows(self,v,time_step):
        """ return flows between compartments """

        # Calculate pressure in each compartment
        p = np.zeros(self.circ.model['no_of_compartments'])
        for i in np.arange(len(p)-1):
            p[i] = (v[i]-self.circ.data['s'][i]) / self.circ.data['compliance'][i]
        p[-1] = self.mesh.model['uflforms'].LVcavitypressure()

        # Add 1 for VAD
        f = np.zeros(self.circ.model['no_of_compartments']+1)
        r = self.circ.data['resistance']
    
        for i in np.arange(len(p)):
            f[i] = (1.0 / r[i] * (p[i-1] - p[i]))

        # Check for VAD
        if (self.circ.va):
            f[-1] = self.va.data['max_flow'] +\
                (p[-1] - p[0]) * self.va.data['pump_slope']

        # Add in the valves
        # Aortic
        if (p[-1] <= p[0]):
            f[0] = (p[-1] - p[0]) * \
                self.circ.data['aortic_insufficiency_conductance']
        # Mitral
        if (p[-1] >= p[-2]):
            f[-2] = (p[-2]-p[-1]) * \
                self.circ.data['mitral_insufficiency_conductance']

        return f
        
