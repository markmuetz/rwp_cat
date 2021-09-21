from pathlib import Path

import numpy as np
import pandas as pd

import config


def _normalize_feature_matrix(X_filtered):
    """Apply the normalization. Both mag(nitude) and rot(ation) are normalized. Up to caller
    to decide if just mag both magrot normalization needed. Additionally, apply
    lower troposphere favouring if option is selected."""
    print('normalizing data')

    mag = np.sqrt(X_filtered[:, :config.NUM_PRESSURE_LEVELS] ** 2 +
                  X_filtered[:, config.NUM_PRESSURE_LEVELS:] ** 2)
    rot = np.arctan2(X_filtered[:, :config.NUM_PRESSURE_LEVELS],
                     X_filtered[:, config.NUM_PRESSURE_LEVELS:])
    # Normalize the profiles by the maximum magnitude at each level.
    max_mag = mag.max(axis=0)
    if config.FAVOUR_LOWER_TROP:
        # This is done by modifying max_mag, which means it's easy to undo by performing
        # reverse using max_mag.
        # N.B. increasing max_mag will decrease the normalized values.
        # Because the values are laid out from highest altitude (lowest pressure) to lowest,
        # this will affect the upper trop.
        max_mag[:config.FAVOUR_INDEX] *= config.FAVOUR_FACTOR
    print('max_mag = {}'.format(max_mag))
    norm_mag = mag / max_mag[None, :]
    u_norm_mag = norm_mag * np.cos(rot)
    v_norm_mag = norm_mag * np.sin(rot)
    # Normalize the profiles by the rotation at 850 hPa.
    rot_at_level = rot[:, config.INDEX_850HPA]
    norm_rot = rot - rot_at_level[:, None]
    index_850hPa = config.INDEX_850HPA
    print('# prof with mag<1 at 850 hPa: {}'.format((mag[:, index_850hPa] < 1).sum()))
    print('% prof with mag<1 at 850 hPa: {}'.format((mag[:, index_850hPa] < 1).sum() /
                                                     mag[:, index_850hPa].size * 100))
    u_norm_mag_rot = norm_mag * np.cos(norm_rot)
    v_norm_mag_rot = norm_mag * np.sin(norm_rot)

    Xu_mag = u_norm_mag
    Xv_mag = v_norm_mag
    # Add the two matrices together to get feature set.
    X_mag = np.concatenate((Xu_mag, Xv_mag), axis=1)

    Xu_magrot = u_norm_mag_rot
    Xv_magrot = v_norm_mag_rot
    # Add the two matrices together to get feature set.
    X_magrot = np.concatenate((Xu_magrot, Xv_magrot), axis=1)

    return X_mag, X_magrot, max_mag, rot_at_level


def main():
    """Normalize profiles by rotating and normalizing on magnitude.

    Profiles are normalized w.r.t. rotation by picking a height level, in this case 850 hPa and
    rotating the profiles so that the wind vectors at 850 hPa are aligned.
    They are normalized w.r.t. to magnitude by calculating the max. magnitude at each height
    level (sqrt(u**2 + v**2)) and normalizing by dividing by this.
    Additionally, if FAVOUR_LOWER_TROP is set then the max_mag array is multiplied by
    FAVOUR_FACTOR above the level defined by FAVOUR_INDEX. This effectively *reduces* their
    weighting when PCA/KMeans are applied by decreasing their normalized values.

    Reads in the filtered profiles and outputs normalized profiles and max_mag array (to same HDF5
    file).
    """
    datadir = config.PATHS['datadir']
    outputdir = config.PATHS['outputdir']

    input_filename = f'{outputdir}/rwp_cat_output/analysis/profiles_filtered.hdf'
    output_filename = f'{outputdir}/rwp_cat_output/analysis/profiles_normalized.hdf'
    output_filename = Path(output_filename)

    if output_filename.exists():
        print(f'{output_filename} already exists. Delete to rerun')
        return

    norm = 'magrot'

    df_filtered = pd.read_hdf(input_filename)

    # Sanity checks. Make sure that dataframe is laid out how I expect: first num_pres vals
    # are u vals and num_pres - num_pres * 2 are v vals.
    num_pres = config.NUM_PRESSURE_LEVELS
    assert all([col[0] == 'u' for col in df_filtered.columns[:num_pres]])
    assert all([col[0] == 'v' for col in df_filtered.columns[num_pres: num_pres * 2]])
    X_filtered = df_filtered.values[:, :config.NUM_PRESSURE_LEVELS * 2]

    X_mag, X_magrot, max_mag, rot_at_level = _normalize_feature_matrix(X_filtered)
    if norm == 'mag':
        X = X_mag
    elif norm == 'magrot':
        X = X_magrot

    columns = df_filtered.columns[:-2]  # lat/lon are copied over separately.
    df_norm = pd.DataFrame(index=df_filtered.index, columns=columns, data=X)
    df_norm['lat'] = df_filtered['lat']
    df_norm['lon'] = df_filtered['lon']
    df_norm['rot_at_level'] = rot_at_level
    df_max_mag = pd.DataFrame(data=max_mag)

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f'saving to {output_filename}')

    # Both saved into same HDF file with different key.
    df_norm.to_hdf(output_filename, 'normalized_profile')
    df_max_mag.to_hdf(output_filename, 'max_mag')


if __name__ == '__main__':
    main()

