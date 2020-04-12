# built-in python modules
import os
import inspect

# scientific python add-ons
import numpy as np
import pandas as pd

# plotting stuff
# first line makes the plots appear in the notebook
import matplotlib.pyplot as plt
import matplotlib as mpl
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
# finally, we import the pvlib library
import pvlib

# Import CoolProp - Thermodynamics Library
from CoolProp.CoolProp import PropsSI, PhaseSI, HAPropsSI
import CoolProp.CoolProp as CP

# Find the absolute file path to your pvlib installation
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))

# absolute path to a data file
datapath = os.path.join(pvlib_abspath, 'data', '703165TY.csv')

# read tmy data with year values coerced to a single year
tmy_data, meta = pvlib.iotools.read_tmy3(datapath, coerce_year=2015)
tmy_data.index.name = 'Time'

# TMY data seems to be given as hourly data with time stamp at the end
# shift the index 30 Minutes back for calculation of sun positions
tmy_data = tmy_data.shift(freq='-30Min')['2015']
tmy_data.head()
plt.figure(1)
plt.plot(tmy_data['GHI'])
plt.ylabel('Irradiance (W/m**2)')
plt.show(block=False)
#Before we can calculate power for all times in the TMY file, we will need to calculate:
#* solar position 
#* extra terrestrial radiation
#* airmass
#* angle of incidence
#* POA sky and ground diffuse radiation
#* cell and module temperatures


#First, define some PV system parameters.
surface_tilt = 45
surface_azimuth = 180 # pvlib uses 0=North, 90=East, 180=South, 270=West convention
albedo = 0.2

# create pvlib Location object based on meta data
sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz='US/Alaska', 
                                     altitude=meta['altitude'], name=meta['Name'].replace('"',''))


# Calculate the solar position for all times in the TMY file. 

# The default solar position algorithm is based on Reda and Andreas (2004). Our implementation is pretty fast, but you can make it even faster if you install [``numba``](http://numba.pydata.org/#installing) and use add  ``method='nrel_numba'`` to the function call below.

solpos = pvlib.solarposition.get_solarposition(tmy_data.index, sand_point.latitude, sand_point.longitude)
plt.figure(2)
plt.plot(solpos)
plt.show(block=False)

# The funny looking jump in the azimuth is just due to the coarse time sampling in the TMY file.

### DNI ET

# Calculate extra terrestrial radiation. This is needed for many plane of array diffuse irradiance models.
dni_extra = pvlib.irradiance.get_extra_radiation(tmy_data.index)
dni_extra = pd.Series(dni_extra, index=tmy_data.index)
plt.figure(3)
plt.plot(dni_extra)
plt.ylabel('Extra terrestrial radiation (W/m**2)')
plt.show(block=False)


### Airmass

# Calculate airmass. Lots of model options here, see the ``atmosphere`` module tutorial for more details.

airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
plt.figure(4)
plt.plot(airmass)
plt.ylabel('Airmass')
plt.show(block=False)

### POA sky diffuse
#Diffuse radiation from the sky dome is typically divided into several components:

#1 - the isotropic component, which represents the uniform irradiance from the sky dome;
#2 - the circumsolar diffuse component, which represents the forward scattering of radiation concentrated in the area immediately surrounding the sun;
#3 - the horizon brightening component.#
#Use the Hay Davies model to calculate the plane of array diffuse sky radiation. 
#See the ``irradiance`` module tutorial for comparisons of different models.
#
poa_sky_diffuse = pvlib.irradiance.haydavies(surface_tilt, surface_azimuth,
                                             tmy_data['DHI'], tmy_data['DNI'], dni_extra,
                                             solpos['apparent_zenith'], solpos['azimuth'])

plt.figure(5)
plt.plot(poa_sky_diffuse)
plt.ylabel('Irradiance (W/m**2)')
plt.show(block=False)

### POA ground diffuse

# Calculate ground diffuse. We specified the albedo above. 
# You could have also provided a string to the ``surface_type`` keyword argument.

poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(surface_tilt, tmy_data['GHI'], albedo=albedo)
plt.figure(6)

plt.plot(poa_ground_diffuse)
plt.ylabel('Irradiance (W/m**2)')
plt.show(block=False)


### AOI

#In geometric optics, the angle of incidence is the angle between a ray incident on a surface and the line perpendicular to the surface at the point of incidence, called the normal. 
aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])
plt.figure(7)
plt.plot(aoi)
plt.ylabel('Angle of incidence (deg)')
# Note that AOI has values greater than 90 deg. This is ok.
plt.show(block=False)
### POA total
# Calculate POA irradiance
poa_irrad = pvlib.irradiance.poa_components(aoi, tmy_data['DNI'], poa_sky_diffuse, poa_ground_diffuse)
plt.figure(8)
legend= poa_irrad.columns.tolist()
plt.plot(poa_irrad)
plt.legend(legend)
plt.ylabel('Irradiance (W/m**2)')
plt.title('POA Irradiance')
plt.show(block=False)


### Cell and module temperature

# Calculate pv cell and module temperature
plt.figure(9) 
params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
pvtemps = pvlib.temperature.sapm_cell(poa_irrad['poa_global'], tmy_data['Wspd'], tmy_data['DryBulb'], **params)
plt.plot(pvtemps)
plt.ylabel('Temperature (C)')
plt.show(block=False)

#computing DC POWER 
sandia_modules = pvlib.pvsystem.retrieve_sam(name='SandiaMod')
sandia_module = sandia_modules.Canadian_Solar_CS5P_220M___2009_
effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irrad.poa_direct, poa_irrad.poa_diffuse, airmass, aoi, sandia_module)
sapm_out = pvlib.pvsystem.sapm(effective_irradiance, pvtemps, sandia_module)
plt.figure(10) 
plt.plot(sapm_out[['p_mp']])
plt.ylabel('DC Power (W)')
plt.show(block=False)


## DC power using single diode
cec_modules = pvlib.pvsystem.retrieve_sam(name='CECMod')
np.set_printoptions(threshold=np.inf)

cec_module = cec_modules.Zytech_Solar_ZT280P
d = {k: cec_module[k] for k in ['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s']}
photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
    pvlib.pvsystem.calcparams_desoto(poa_irrad.poa_global,
                                 pvtemps,
                                 cec_module['alpha_sc'],
                                 EgRef=1.121,
                                 dEgdT=-0.0002677, **d))
single_diode_out = pvlib.pvsystem.singlediode(photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth)
plt.figure(11) 
plt.plot(single_diode_out[['p_mp']])
plt.ylabel('DC Power (W)')
plt.show(block=False)



sapm_inverters = pvlib.pvsystem.retrieve_sam('sandiainverter')
sapm_inverter = sapm_inverters['iPower__SHO_4_8__240V_']
p_acs = pd.DataFrame()
p_acs['sapm'] = pvlib.pvsystem.snlinverter(sapm_out.v_mp, sapm_out.p_mp, sapm_inverter)
p_acs['sd'] = pvlib.pvsystem.snlinverter(single_diode_out.v_mp, single_diode_out.p_mp, sapm_inverter)

plt.figure(12)
plt.plot(p_acs)
plt.ylabel('AC Power (W)')
plt.show(block=False)


diff = p_acs['sapm'] - p_acs['sd']
plt.figure(13)
plt.plot(diff)
plt.ylabel('SAPM - SD Power (W)')
plt.show(block=False)

plt.figure(14)
legend= p_acs.columns.tolist()
plt.plot(p_acs.loc['2015-07-05':'2015-07-06'])
plt.legend(legend)
plt.show(block=False)

# Some statistics on the AC power


# create data for a y=x line
p_ac_max = p_acs.max().max()
yxline = np.arange(0, p_ac_max)

fig = plt.figure(15, figsize=(12,12))
ax = fig.add_subplot(111, aspect='equal')
sc = ax.scatter(p_acs['sd'], p_acs['sapm'], c=poa_irrad.poa_global, alpha=1)  
ax.plot(yxline, yxline, 'r', linewidth=3)
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.set_xlabel('Single Diode model')
ax.set_ylabel('Sandia model')
fig.colorbar(sc, label='POA Global (W/m**2)')
plt.show(block=False)



### Solar Collector and Refrigeration System

def solar_colector(T_ext, T_in, I, g_s):
    # Calculates solar colector parameters based on External Temperature, entrance temperature, Global Irradiance and mass flow rate [g/s]

    Area = 2
    m_dot = Area * g_s * 10**(-3)
    Losses = 3

    # 0.97 and 3 are normal value for heat losses but they must be changed according to the solar colector parameters

    if I > 0:
        efficiency = 0.97 - ( Losses * (T_in - T_ext)/ I )

    # Irradiance can be zero at night so dividing by zero will raise a warning
    else:
        efficiency = 0

    Total_sun_ernergy = Area * I

    Col_eff_power = Total_sun_ernergy * efficiency

    ## It would be very interesting to play with the percentage also
    Col_Fluid = 'R600a'

    # Get speciic heat from fluid
    cp = PropsSI('C', 'T', 273 + T_in , 'P', 101.325, Col_Fluid)

    # Get Exit temperature from the colector - Q = m_dot * cp * (T_exit - T_in)
    T_exit = T_in + Col_eff_power / (m_dot * cp)

    if Col_eff_power < 0:
        Col_eff_power = 0

    return Col_eff_power, efficiency, T_exit, Total_sun_ernergy


# Choose the working fluid inside the refrigeration cycle - This should be inside a for loop to compare various fluids
fluid = 'R152a'
T_sup = 273.15 + 5

# Define evaporator Temperature - Temperature to cool the house
T_e = 10

# Assuming an entrance temperature of 20 for the first solar colector - System turned off
T_in = 20

Q_g = []
Q_sun = []
T_out_circuit = []

i = 0

# Looping through a year to simulate thermal loads and efficiencies
for amb_temp in tmy_data.loc[:,'DryBulb']:

    poa_global = poa_irrad.loc[:,'poa_global'][i]

    hour_power = 0
    sun_energy = 0

    T_out_circuit.append(T_in)

    # Create a for loop to simulate two collectors in series
    for j in range(2):

        Col_eff_power, efficiency, T_exit, _sun = solar_colector(amb_temp, T_in, poa_global, 7)

        hour_power += Col_eff_power
        sun_energy += _sun

        T_in = T_exit

    ##  Still need to model the thermal storage and CHP system

    # Lets assume that the Generator has a 90% efficiency - This depends on the brand and type of mounting
    # Assuming a constant 90ºC temperature from the thermal storage

    Q_g.append(0.9 * hour_power)
    Q_sun.append(sun_energy)

    T_g = 90
    P_g = PropsSI('P', 'T', T_g + 273.15, 'Q', 1, fluid)

    # Define condenser temperature - Ambient Temperature - 10 - In PropSI the Temp. should be in Kelvin
    T_c = amb_temp - 10
    P_c = PropsSI('P', 'T', T_c + 273.15, 'Q', 1, fluid)

    # Q_e will depend on the required heat load to cool the house down - Architectural and thermal modeling in Energy+
    # Assuming 5 kW of cooling power:

    Q_e = 5

    i += 1


heating_power = pd.DataFrame({'Irradiance_Power': pd.Series(Q_sun, index=tmy_data.index),
                              'Heating_power': pd.Series(Q_g, index=tmy_data.index),
                              'Temperature*100': tmy_data.loc[:,'DryBulb'].apply(lambda x: x*100) })

plt.figure(16, figsize=(12,12))
legend = heating_power.columns.tolist()
plt.plot(heating_power)
plt.legend(legend)
plt.ylabel('Power [W]')
plt.title('Collector Efficiency')
plt.show(block=False)

Circuit_Temperatures = pd.DataFrame({'T_Out_Circuit': pd.Series(T_out_circuit, index=tmy_data.index),
                                     'Amb_Temperature': tmy_data.loc[:,'DryBulb'] })

plt.figure(17, figsize=(12,12))
legend = Circuit_Temperatures.columns.tolist()
plt.plot(Circuit_Temperatures)
plt.legend(legend)
plt.ylabel('Temperature [ºC]')
plt.title('Temperatures in the Circuit')
plt.show()
