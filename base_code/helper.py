import numpy as np

def to_observed_matrix(data_file, aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4))

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum