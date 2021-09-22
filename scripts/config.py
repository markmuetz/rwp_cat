import os
import socket

USER = os.environ.get('USER')

ALL_PATHS = {
    'jasmin': {
        'datadir': '/gws/nopw/j04/paracon_rdg/users/mmuetz/rdf_mirror/um10.9_runs/archive/u-au197',
        'outputdir': '/gws/nopw/j04/paracon_rdg/users/mmuetz',
    }
}


def _short_hostname():
    hostname = socket.gethostname()
    if '.' in hostname and hostname.split('.')[1] == 'jasmin':
        return 'jasmin'
    return hostname


hostname = _short_hostname()
if hostname[:4] == 'host':
    hostname = 'jasmin'

if hostname not in ALL_PATHS:
    raise Exception(f'Unknown hostname: {hostname}')

PATHS = ALL_PATHS[hostname]

class Settings:
    def __init__(self, **settings_dict):
        self._settings_dict = settings_dict
        for k, v in settings_dict.items():
            setattr(self, k, v)

    def set(self, key, value):
        if key not in self._settings_dict:
            raise KeyError(key)
        self._settings_dict[key] = value

    def copy(self):
        return Settings(**self._settings_dict)

    def __repr__(self):
        r = []
        r.append('Settings(')
        for k, v in self._settings_dict.items():
            r.append(f'    {k} = {v},')
        r.append(')')
        return '\n'.join(r)


default_settings = Settings(**dict(
    # 23.75N - 23.75S.
    TROPICS_SLICE=slice(53, 92),
    # Minimum explained variance for PCA.
    EXPL_VAR_MIN=0.9,
    # Threshold settings to use. CAPE threshold in J/kg.
    CAPE_THRESH=100,
    SHEAR_PRESS_THRESH_HPA=500,
    SHEAR_PERCENTILE=75,
    # Description of input data.
    NUM_PRESSURE_LEVELS=20,
    INDEX_850HPA=-4,
    # Favour lower trop settings.
    FAVOUR_LOWER_TROP=True,
    FAVOUR_FACTOR=4,
    # where to apply favouring to.
    FAVOUR_INDEX=10,
    # which filters to apply.
    FILTERS=('cape', 'shear'),
    # shear filter only:
    # FILTERS=('shear', )
    NORM='magrot',
))
