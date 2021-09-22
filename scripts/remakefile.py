from glob import glob

from remake import Remake, TaskRule

import config
from shear_profile_filter import (
    shear_profile_filter, filter_profiles, extract_lat_lon,
    find_max_shear, calc_shear, apply_filters
)
from shear_profile_normalize import shear_profile_normalize, normalize_feature_matrix


datadir = config.PATHS['datadir']
outputdir = config.PATHS['outputdir']

remake = Remake()


class Filter(TaskRule):
    @staticmethod
    def rule_inputs():
        if config.USER == 'mmuetz':
            input_filename_glob = f'{datadir}/share/data/history/P5Y_DP20/au197a.pc19*.uvcape.nc'
        filenames = sorted(glob(input_filename_glob))
        return {f'uvcape_{fn}': fn
                for fn in filenames}

    rule_outputs = {'profiles_filtered': f'{outputdir}/rwp_cat_output/analysis/profiles_filtered.hdf'}

    depends_on = [shear_profile_filter, filter_profiles, extract_lat_lon, find_max_shear,
                  calc_shear, apply_filters]

    def rule_run(self):
        settings = config.default_settings
        input_filenames = [str(p) for p in self.inputs.values()]
        output_filename = self.outputs['profiles_filtered']

        shear_profile_filter(settings, input_filenames, output_filename)


class Normalize(TaskRule):
    rule_inputs = Filter.rule_outputs
    rule_outputs = {'profiles_normalized': f'{outputdir}/rwp_cat_output/analysis/profiles_normalized.hdf'}

    depends_on = [shear_profile_normalize, normalize_feature_matrix]

    def rule_run(self):
        settings = config.default_settings
        input_filename = self.inputs['profiles_filtered']
        output_filename = self.outputs['profiles_normalized']

        shear_profile_normalize(settings, input_filename, output_filename)

