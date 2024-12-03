# -*- coding: utf-8 -*-
"""
Created on Mon SEP 29 2024

 SARA
"""

import numpy as np

class Circulation():

    def __init__(self, circ_struct, mesh_object):

        self.model = dict()
        self.data = dict()
        self.mesh = mesh_object

        # Model scheme (biventricular model now)
        self.model['model_scheme'] = circ_struct['model_scheme'][0]

        # Number of compartments
        self.model['no_of_compartments'] = len(circ_struct['compartments'])

        # Store blood volume
        self.data['blood_volume'] = circ_struct['blood_volume'][0]

        # Read compartmental data (LV and RV)
        for comp in circ_struct['compartments']:
            if not (comp['name'][0] == 'left_ventricle' or comp['name'][0] == 'right_ventricle'):
                n = comp['name'][0]
                for t in ['resistance', 'compliance', 'slack_volume']:
                    n = ('%s_%s') % (comp['name'][0], t)
                    self.data[n] = comp[t][0]
            else:
                # Handle left ventricle
                if comp['name'][0] == 'left_ventricle':
                    self.data['left_ventricle_resistance'] = comp['resistance'][0]
                    self.data['left_ventricle_slack_volume'] = comp['slack_volume'][0]
                    self.model['left_ventricle_wall_density'] = comp['wall_density'][0]
                    self.model['left_ventricle_initial_edv'] = comp['initial_ed_volume'][0]

                # Handle right ventricle
                if comp['name'][0] == 'right_ventricle':
                    self.data['right_ventricle_resistance'] = comp['resistance'][0]
                    self.data['right_ventricle_slack_volume'] = comp['slack_volume'][0]
                    self.model['right_ventricle_wall_density'] = comp['wall_density'][0]
                    self.model['right_ventricle_initial_edv'] = comp['initial_ed_volume'][0]

        # Model compartments (include RV-related vessels)
        if self.model['model_scheme'] == 'LV_RV_with_6_compartments':
            vessels_list = ['aorta', 'arteries', 'arterioles', 'capillaries', 'venules', 'veins', 
                            'pulmonary_arteries', 'pulmonary_veins']

        # Build the compliance, and resistance arrays
        self.data['compliance'] = []
        for v in vessels_list:
            c = self.data[('%s_compliance' % v)]
            self.data['compliance'].append(c)
        # Add 0 for LV and RV compliance
        self.data['compliance'].extend([0, 0])
        self.data['compliance'] = np.array(self.data['compliance'])

        self.data['resistance'] = []
        vessels_list.extend(['left_ventricle', 'right_ventricle'])
        self.model['compartment_list'] = vessels_list
        for v in self.model['compartment_list']:
            r = self.data[('%s_resistance' % v)]
            self.data['resistance'].append(r)
        self.data['resistance'] = np.array(self.data['resistance'])

        # Create arrays for volume, slack_volume, and pressure
        self.data['v'] = np.zeros(self.model['no_of_compartments'])
        self.data['s'] = np.zeros(self.model['no_of_compartments'])

        # Put blood in the veins and adjust slack volumes
        for i, c in enumerate(self.model['compartment_list']):
            n = ('%s_slack_volume' % c)
            self.data['s'][i] = self.data[n]
            self.data['v'][i] = self.data[n]

        # Update slack volumes for both ventricles based on mesh volume
        self.data['s'][-2] = self.mesh.model['uflforms'].LVcavityvol()
        self.data['s'][-1] = self.mesh.model['uflforms'].RVcavityvol()
        self.data['v'][-2] = self.data['s'][-2]
        self.data['v'][-1] = self.data['s'][-1]
        self.data['total_slack_volume'] = sum(self.data['s'])

        # Excess blood goes to the veins
        self.data['v'][-3] = self.data['v'][-3] + \
            (self.data['blood_volume'] - self.data['total_slack_volume'])

        # Initialize pressures
        self.data['p'] = np.zeros(self.model['no_of_compartments'])
        for i in np.arange(0, self.model['no_of_compartments'] - 2):
            self.data['p'][i] = (self.data['v'][i] - self.data['s'][i]) / self.data['compliance'][i]

        # Initialize pressures for LV and RV (in mm Hg)
        self.data['p'][-2] = 0.0075 * self.mesh.model['uflforms'].LVcavitypressure()
        self.data['p'][-1] = 0.0075 * self.mesh.model['uflforms'].RVcavitypressure()

        # Allocate pressure, volume, slack volume data
        for i, v in enumerate(self.model['compartment_list']):
            self.data['pressure_%s' % v] = self.data['p'][i]
            self.data['volume_%s' % v] = self.data['v'][i]
            self.data['slack_volume_%s' % v] = self.data['s'][i]

        # Allocate space for flows
        self.data['f'] = np.zeros(self.model['no_of_compartments'])
        self.model['flow_list'] = [
            'flow_left_ventricle_to_aorta',
            'flow_right_ventricle_to_pulmonary_arteries',
            'flow_aorta_to_arteries',
            'flow_arteries_to_arterioles',
            'flow_arterioles_to_capillaries',
            'flow_capillaries_to_venules',
            'flow_venules_to_veins',
            'flow_pulmonary_arteries_to_capillaries',
            'flow_pulmonary_veins_to_right_ventricle'
        ]

        for f in self.model['flow_list']:
            self.data[f] = 0

        # Aortic and mitral insufficiency conductance
        self.data['aortic_insufficiency_conductance'] = 0
        self.data['mitral_insufficiency_conductance'] = 0
        self.data['pulmonary_insufficiency_conductance'] = 0
        self.data['tricuspid_insufficiency_conductance'] = 0

        # Regurgitant volumes
        self.data['mitral_reg_volume'] = 0
        self.data['aortic_reg_volume'] = 0
        self.data['tricuspid_reg_volume'] = 0
        self.data['pulmonary_reg_volume'] = 0

    def update_circulation(self, time_step, initial_v):

        # Update volumes
        self.data['v'] = self.evolve_volume(time_step, initial_v)

        # Update pressures
        for i in range(self.model['no_of_compartments'] - 2):
            self.data['p'][i] = (self.data['v'][i] - self.data['s'][i]) / self.data['compliance'][i]

        # Update LV and RV pressures
        self.data['p'][-2] = 0.0075 * self.mesh.model['uflforms'].LVcavitypressure()
        self.data['p'][-1] = 0.0075 * self.mesh.model['uflforms'].RVcavitypressure()

    def evolve_volume(self, time_step, initial_v):
        from scipy.integrate import solve_ivp

        def derivs(t, v):
            dv = np.zeros(self.model['no_of_compartments'])
            flows = self.return_flows(v)
            self.data['f'] = flows
            for i in np.arange(self.model['no_of_compartments']):
                if i == (self.model['no_of_compartments'] - 2):  # LV
                    dv[i] = flows[i] - flows[0] + flows[-2]
                elif i == (self.model['no_of_compartments'] - 1):  # RV
                    dv[i] = flows[i] - flows[-1]
                else:
                    dv[i] = flows[i] - flows[i + 1]
            return dv

        sol = solve_ivp(derivs, [0, time_step], initial_v)

        # Tidy up negative values
        y = sol.y[:, -1]
        y[-3] = y[-3] + (self.data['blood_volume'] - np.sum(y))
        return y

    def return_flows(self, v):
        """ Return flows between compartments """

        # Calculate pressure in each compartment
        p = np.zeros(self.model['no_of_compartments'])
        for i in np.arange(len(p) - 2):
            p[i] = (v[i] - self.data['s'][i]) / self.data['compliance'][i]
        p[-2] = 0.0075 * self.mesh.model['uflforms'].LVcavitypressure()
        p[-1] = 0.0075 * self.mesh.model['uflforms'].RVcavitypressure()

        # Flow array (including for VAD)
        f = np.zeros(self.model['no_of_compartments'] + 1)
        r = self.data['resistance']

        for i in np.arange(len(p)):
            f[i] = (1.0 / r[i] * (p[i - 1] - p[i]))

        # Add in valve regurgitation
        # Aortic valve
        if p[-2] <= p[0]:
            f[0] = (p[-2] - p[0]) * self.data['aortic_insufficiency_conductance']
        # Mitral valve
        if p[-2] >= p[-3]:
            f[-3] = (p[-3] - p[-2]) * self.data['mitral_insufficiency_conductance']
        # Pulmonary valve
        if p[-1] <= p[-4]:
            f[-4] = (p[-1] - p[-4]) * self.data['pulmonary_insufficiency_conductance']
        # Tricuspid valve
        if p[-1] >= p[-5]:
            f[-5] = (p[-5] - p[-1]) * self.data['tricuspid_insufficiency_conductance']

        return f
