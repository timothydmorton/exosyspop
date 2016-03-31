__version__ = '0.1'

try:
    __EXOSYSPOP_SETUP__
except NameError:
    __EXOSYSPOP_SETUP__ = False

if not __EXOSYSPOP_SETUP__:
    
    __all__=['BinaryPopulation',
             'PowerLawBinaryPopulation',
             'KeplerBinaryPopulation',
             'KeplerPowerLawBinaryPopulation',
             'BlendedBinaryPopulation',
             'TRILEGAL_BinaryPopulation',
             'BGBinaryPopulation',
             'TRILEGAL_BGBinaryPopulation',
             'PlanetPopulation',
             'PoissonPlanetPopulation',
             'PopulationMixture']

    from .populations import (BinaryPopulation, PowerLawBinaryPopulation,
                              KeplerBinaryPopulation, KeplerPowerLawBinaryPopulation,
                              BlendedBinaryPopulation, TRILEGAL_BinaryPopulation,
                              BGBinaryPopulation, TRILEGAL_BGBinaryPopulation,
                              PlanetPopulation, PoissonPlanetPopulation,
                              PopulationMixture)

