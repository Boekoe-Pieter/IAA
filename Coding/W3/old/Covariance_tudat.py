# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import constants

# other useful modules
import numpy as np
import datetime
import rebound

# load python files
import Utilities as Util

# load kernals
spice.load_standard_kernels()
# spice.load_kernel('Coding/W3P1/gm_Horizons.pck')
np.set_printoptions(linewidth=160)

NGA_data = {
    'C/2001 Q4': {
        "Full": {   
            "n5": {"A1": 1.6506, "A2": 0.062406, "A3": 0.001412,
                   'm': 2.15, 'n': 5.093, 'k': 4.6142,
                   'r0': 2.808, 'alph': 0.1113, "tau": 0,

                   'T_peri_sig': 0.00002320, 'Dis_peri_sig':0.00000034, 'e_sig':0.00000079,
                   'om_sig': 0.000050, 'Om_sig': 0.000008, 'i_sig':0.000011, 'a_repi_sig':0.82*1e-6
                   },
        },
    }
}

data = 'Coding/W2P1/unfiltered.txt'

comet_name = 'Comet'


# -----------------------------------
# Get initial conditions from Rebound
# -----------------------------------
comet_dict = Util.get_data(data)
data_dict = Util.select_comet(comet_dict, NGA_data)

current_comet_data = data_dict['C/2001 Q4']["Full"]["n5"]  

arc1_mjd, arc2_mjd, epoch_mjd, T_perihelion_mjd, q, ecc, aop, RAAN, i, a_recip = Util.get_orbital_elements(current_comet_data)

sim = rebound.Simulation()
sim.units = ('Days', 'AU', 'Msun')

sim.add(m=1.0)

a = q / (1.0 - ecc)

sim.add(
    primary=sim.particles[0],
    a=a,
    e=ecc,
    inc=i,
    Omega=RAAN,
    omega=aop,
    T=T_perihelion_mjd   
)

sim.move_to_com()
sim.t = arc1_mjd

comet = sim.particles[1]
AU = constants.ASTRONOMICAL_UNIT
day_sec = constants.JULIAN_DAY
# initial_state_Rebound = np.array([comet.a, comet.e, comet.inc, comet.Omega, comet.omega, comet.theta])
initial_state_Rebound = np.array([comet.x*AU, comet.y*AU, comet.z*AU, comet.vx*AU/day_sec, comet.vy*AU/day_sec, comet.vz*AU/day_sec])

# ------------------------------------------
# Convert times from MJD to J2000 in seconds
# ------------------------------------------
Arc1_J2000 = (arc1_mjd - 51544.5)*constants.JULIAN_DAY
epoch_J2000 = (epoch_mjd - 51544.5)*constants.JULIAN_DAY
Arc2_J20000 = (arc2_mjd - 51544.5)*constants.JULIAN_DAY
T_perihelion_J2000 = (T_perihelion_mjd - 51544.5)*constants.JULIAN_DAY

# ------------------------------------
# TUDAT For propagation and covariance
# ------------------------------------
# Environment setup
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

bodies_to_create = [
    "Sun", 
    # "Mercury", "Venus", "Earth", "Moon",
    # "Mars", "Jupiter", "Uranus", "Neptune"
]

# Default solar system
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Add empty comet body
body_settings.add_empty_settings(comet_name)
bodies = environment_setup.create_system_of_bodies(body_settings)

# -------------------
# Acceleration setup
# -------------------
accelerations_on_comet = dict(
    Sun =  [
        propagation_setup.acceleration.point_mass_gravity(),
        # propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
    ],
    # Mercury =  [propagation_setup.acceleration.point_mass_gravity()],
    # Venus = [propagation_setup.acceleration.point_mass_gravity()],
    # Earth = [propagation_setup.acceleration.point_mass_gravity()],
    # Moon = [propagation_setup.acceleration.point_mass_gravity()],
    # Mars = [propagation_setup.acceleration.point_mass_gravity()],
    # Jupiter = [propagation_setup.acceleration.point_mass_gravity()],
    # Saturn = [propagation_setup.acceleration.point_mass_gravity()],
    # Uranus = [propagation_setup.acceleration.point_mass_gravity()],
    # Neptune = [propagation_setup.acceleration.point_mass_gravity()],
)

acceleration_settings = {comet_name: accelerations_on_comet}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, [comet_name], [global_frame_origin]
)

initial_state_Rebound = np.array(initial_state_Rebound).reshape((6,1))

dependent_variables_to_save = [
    propagation_setup.dependent_variable.relative_position(comet_name, "Sun"),

]

# -------------------
# termination / integrator
# -------------------
termination_time = T_perihelion_J2000  # already in seconds since J2000

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    time_step=day_sec,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

termination_settings = propagation_setup.propagator.time_termination(termination_time)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies=[global_frame_origin],                # list[str]
    acceleration_models=acceleration_models,             # dict (as you created)
    bodies_to_integrate=[comet_name],                    # list[str]
    initial_states=initial_state_Rebound,                # numpy array shape (6,1)
    initial_time=Arc1_J2000,                             # float (seconds since J2000)
    integrator_settings=integrator_settings,             # integrator settings object
    termination_settings=termination_settings,           # termination settings
    output_variables=dependent_variables_to_save        # dependent variables
)

# -------------------
# Run simulation
# -------------------
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

propagation_results = dynamics_simulator.propagation_results
dependent_variables = propagation_results.dependent_variable_history

Independed_values = np.vstack(list(dependent_variables.values()))
print(np.array(Independed_values))
time = dependent_variables.keys()
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(15, 8))

ax=fig.add_subplot(111,projection='3d')
ax.set_aspect('equal', adjustable='box')

ax.plot(Independed_values[:,0]/constants.ASTRONOMICAL_UNIT,Independed_values[:,1]/constants.ASTRONOMICAL_UNIT,Independed_values[:,2]/constants.ASTRONOMICAL_UNIT) 

radius_sun = 696340*1000/constants.ASTRONOMICAL_UNIT # in AU
_u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j]
_x = radius_sun*np.cos(_u)*np.sin(_v)
_y = radius_sun*np.sin(_u)*np.sin(_v)
_z = radius_sun*np.cos(_v)
ax.plot_wireframe(_x,_y,_z,color="r",alpha=0.5,lw=0.5,zorder=0)


ax.set_xlabel('x,AU')
ax.set_ylabel('y,AU')
ax.set_zlabel('z,AU')

plt.title(f"3D trajectory")
plt.savefig(f"Coding/W3P1/3D_trajectory.pdf",dpi=300)
plt.show()


# sigmas = np.array([

#     current_comet_data['T_peri_sig'],
#     current_comet_data['Dis_peri_sig'],
#     current_comet_data['e_sig'],
#     np.deg2rad(current_comet_data['Om_sig']),
#     np.deg2rad(current_comet_data['om_sig']),
#     np.deg2rad(current_comet_data['i_sig']),
#     current_comet_data['a_repi_sig'],
# ])

# covariance = np.diag(sigmas**2)









# # setup environment
# global_frame_origin = "SSB"
# global_frame_orientation = "J2000"

# bodies_to_create = [
#     "Sun",
#     "Mercury",
#     "Venus",
#     "Earth",
#     "Moon",
#     "Mars",
#     "Jupiter",
#     "Saturn",
#     "Uranus",
#     "Neptune",
# ]

# body_settings = environment_setup.get_default_body_settings(
#     bodies_to_create, global_frame_origin, global_frame_orientation
# )

# bodies = environment_setup.create_system_of_bodies(body_settings)
# bodies_to_propagate = body
# central_bodies = [global_frame_origin]

# accelerations = {
#     "Sun": [
#         propagation_setup.acceleration.point_mass_gravity(),
#         propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
#     ],
#     "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
#     "Venus": [propagation_setup.acceleration.point_mass_gravity()],
#     "Earth": [propagation_setup.acceleration.point_mass_gravity()],
#     "Moon": [propagation_setup.acceleration.point_mass_gravity()],
#     "Mars": [propagation_setup.acceleration.point_mass_gravity()],
#     "Jupiter": [propagation_setup.acceleration.point_mass_gravity()],
#     "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
#     "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
# }
# acceleration_settings = {}

# for target in body:
#     acceleration_settings[str(target)] = accelerations

# acceleration_models = propagation_setup.create_acceleration_models(
#     bodies, acceleration_settings, bodies_to_propagate, central_bodies
# )

# system_initial_state = spice.get_body_cartesian_state_at_epoch(
#     target_body_name=body,
#     observer_body_name=global_frame_origin,
#     reference_frame_name=global_frame_orientation,
#     aberration_corrections="NONE",
#     ephemeris_time=arc1_mjd,
# )