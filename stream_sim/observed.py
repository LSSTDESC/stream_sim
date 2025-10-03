#!/usr/bin/env python

import copy
import os

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import healpy as hp
import numpy as np
import pandas as pd
import pylab as plt
import scipy


class StreamObserved:
    """
    Estimate the observed quantities of stars from a stream for a chosen survey.
    """

    def __init__(self, survey=None, release=None, config_file=None, **kwargs):
        """
        Inputs:
        survey : name of the survey. Available: LSST.
            Type : str

        config_file : configuration file containing survey information.
            Type : dict, pandas df ...
        """
        if config_file:
            self._config = copy.deepcopy(config_file)

        elif survey:
            self._load_survey_config(
                survey, release=release
            )  # Check for the folder of the survey
        else:
            raise ValueError("Either 'survey' or 'config_file' must be provided.")
        self._config.update(**kwargs)
        self.load_survey()

    def _load_survey_config(self, survey, release=None):
        """
        Verify if the survey is available by finding its folder containing the properties.

        Args:
            survey (str): Name of the survey.

        Raises:
            FileNotFoundError: If the config file is not found in the expected directory.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_config_path = os.path.join(current_dir, "..", "config/surveys/")

        if release is None:
            self.survey = survey
        else:
            self.survey = survey + "_" + release

        target_file_path = os.path.join(folder_config_path, self.survey + ".yaml")
        # Allow for flexible file types (YAML, JSON)
        if not os.path.exists(target_file_path):
            target_file_path = target_file_path.replace(".yaml", ".json")
            if not os.path.exists(target_file_path):
                raise FileNotFoundError(
                    f"Config file '{self.survey}' does not exist in '{folder_config_path}'."
                )

        # Load the file based on extension
        if target_file_path.endswith(".yaml"):
            import yaml

            with open(target_file_path, "r") as file:
                config_data = yaml.safe_load(file)
        elif target_file_path.endswith(".json"):
            import json

            with open(target_file_path, "r") as file:
                config_data = json.load(file)

        # Verify that the survey and release corresponds to the one in config file
        if survey != config_data.get("name"):
            raise ValueError(
                f"The survey name '{survey}' does not correspond to the one specified in the config file '{config_data['name']}'."
            )
        if release != config_data.get("release", None):
            raise ValueError(
                f"The release '{release}' does not correspond to the one specified in the config file '{config_data['release']}'."
            )

        self._config = copy.deepcopy(config_data)

    def load_survey(self):
        """
        Load survey information such as magnitude limits, extinction maps, completeness, etc.

        Attributes created:
        -------------------
        maglim_map_g : np.ndarray
            Magnitude limit map in the g band.

        maglim_map_r : np.ndarray
            Magnitude limit map in the r band.

        ebv_map : np.ndarray
            Extinction map.

        completeness : callable
            Function to compute completeness.

        log_photo_error : callable
            Function to compute logarithmic photometric error.

        coverage : np.ndarray
            Coverage map.
        """

        print(
            "###################### Reading Survey property files ######################"
        )
        config = self._config.get("survey_files", None)
        if config is None:
            raise ValueError(
                "Please specify the files containing informations of the survey."
            )
        path = config.get("file_path", "")
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "..", "data/surveys/" + self.survey)
        print("Survey's files are searched at the following path : ", path)

        survey_files = {
            "maglim_map_g": hp.read_map,
            "maglim_map_r": hp.read_map,
            "ebv_map": hp.read_map,
            "completeness": StreamObserved.getCompleteness,
            "log_photo_error": StreamObserved.getPhotoError,
            "coverage": hp.read_map,
        }

        # Loop through the dictionary and load each file if the attribute doesn't exist
        for attr, reader in survey_files.items():
            self.load_map(config, attr, reader, path=path)

        # Load other attributes
        self._load_survey_properties()

        print(
            "###################### Reading Survey property files done ######################"
        )

    def load_map(self, config, attr_name, reader_func, path="."):
        if not hasattr(self, attr_name):
            filename = config.get(
                attr_name, None
            )  # Get filename from config, or None if missing

            if filename is None:
                print(f"Warning: Missing '{attr_name}' in the configuration file.")
                return

            full_path = os.path.join(path, filename)

            if not os.path.exists(full_path):
                print(f"Error: File '{full_path}' for {attr_name} does not exist.")
                return

            print(f"Reading {attr_name.replace('_', ' ')}: {full_path}...")
            setattr(self, attr_name, reader_func(full_path))

    def _load_survey_properties(self):
        """
        Load survey properties like extinction coefficients and system error.
        """
        try:
            self.coeff_extinc = (
                self._config["survey_properties"]["coeff_extinc_g"],
                self._config["survey_properties"]["coeff_extinc_r"],
            )
        except KeyError as e:
            print(f"Missing key {e} in the survey_properties config.")

        try:
            self.sys_error = self._config["survey_properties"]["sys_error"]
        except KeyError as e:
            print(f"Missing key {e} in the survey_properties config.")

        try:
            self.saturation = self._config["survey_properties"]["saturation"]
        except KeyError as e:
            self.saturation = 16.0
            print(
                f"Missing key {e} in the survey_properties config. Using default saturation={self.saturation}."
            )

    def inject(self, data, **kwargs):
        """
        Add observed quantities by the survey to the given data.

        Parameters
        ----------
        data : str or pd.DataFrame
            Input data or path to the file.

        Returns
        -------
        DataFrame with the following columns :
            mag_r_meas, mag_g_meas : observed magnitudes of the survey
            magerr_r, magerr_g : absolute errors magnitudes of the survey
            ra, dec : coordinates of the stars
            flag_detection : 1 for detected stars, 0 for others
        """
        data = self._load_data(data)

        # Select only stars from the stream
        flag = kwargs.pop("flag", [1])
        if "flag" in data.columns:
            mask = data["flag"].isin(flag)
            data = data[mask]
        else:
            print("'flag' column not found, skipping filtering.")

        # set the seed for reproducibility
        seed = kwargs.pop("seed", None)
        rng = np.random.default_rng(seed)

        # Convert coordinates (Phi1, Phi2) into (ra,dec)
        endpoints = kwargs.pop("endpoints", None)
        mask_config = kwargs.pop("mask_config", None)

        self.stream = self.phi_to_radec(
            data["phi1"],
            data["phi2"],
            endpoints=endpoints,
            seed=seed,
            rng=rng,
            mask_config=mask_config,
        )
        pix = hp.ang2pix(
            4096, self.stream.icrs.ra.deg, self.stream.icrs.dec.deg, lonlat=True
        )

        # Estimate the extinction, errors
        nside_ebv = hp.get_nside(self.ebv_map)
        if nside_ebv != 4096:  # adjust the nside to the one of extinction map
            pix = hp.ang2pix(
                nside_ebv,
                self.stream.icrs.ra.deg,
                self.stream.icrs.dec.deg,
                lonlat=True,
            )

        extinction_g, extinction_r = self.extinction(pix)

        nside_maglim = hp.get_nside(self.maglim_map_r)
        if nside_maglim != 4096:  # adjust the nside to the one of magnitude limit maps
            pix = hp.ang2pix(
                nside_maglim,
                self.stream.icrs.ra.deg,
                self.stream.icrs.dec.deg,
                lonlat=True,
            )

        magerr_g, magerr_r = self._get_errors(
            pix, mag_r=data["mag_r"] + extinction_r, mag_g=data["mag_g"] + extinction_g
        )

        # Sample observed magnitude for every stars
        mag_g_meas, mag_r_meas = self.sample(
            mag_g=data["mag_g"] + extinction_g,
            mag_r=data["mag_r"] + extinction_r,
            magerr_g=magerr_g,
            magerr_r=magerr_r,
            rng=rng,
            seed=seed,
        )

        new_columns = pd.DataFrame(
            {
                "mag_g_meas": mag_g_meas,
                "magerr_g": magerr_g,
                "mag_r_meas": mag_r_meas,
                "magerr_r": magerr_r,
                "ra": self.stream.icrs.ra.deg,
                "dec": self.stream.icrs.dec.deg,
            }
        )

        # Reset the index to force alignment by row position
        data = data.reset_index(drop=True)
        new_columns = new_columns.reset_index(drop=True)
        # Concatenate along axis=1 (columns)
        data = pd.concat([data, new_columns], axis=1)

        # Apply detection threshold
        flag_r = (
            mag_r_meas != "BAD_MAG"
        )  # Negative fluxes are set to 'BAD_MAG', so counted as undetected
        data["flag_detection_r"] = flag_r & self.detect_flag(
            pix, mag_r=data["mag_r"] + extinction_r, rng=rng, seed=seed, **kwargs
        )
        flag_g = mag_g_meas != "BAD_MAG"
        data["flag_detection_g"] = flag_g & self.detect_flag(
            pix, mag_g=data["mag_g"] + extinction_g, rng=rng, seed=seed, **kwargs
        )
        data["flag_detection"] = (data["flag_detection_r"] == 1) & (
            data["flag_detection_g"] == 1
        )

        if kwargs.get("save"):
            self._save_injected_data(data, kwargs.get("folder", None))

        return data

    def _load_data(self, data):
        """
        Load data from file or return the provided DataFrame.

        Parameters
        ----------
        data : str or pd.DataFrame
            Path to the file or pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            extension = data.split(".")[-1]
            if extension == "csv":
                return pd.read_csv(data)
            elif extension in ["xls", "xlsx"]:
                return pd.read_excel(data)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
        else:
            raise ValueError(f"Unsupported file format")

    def phi_to_radec(
        self,
        phi1,
        phi2,
        endpoints=None,
        seed=None,
        rng=None,
        mask_config={},
    ):
        """
        Transform coordinates (phi1,phi2) to (ra,dec), ensuring that specified phi1 values map to valid pixels.

        Parameters
        ----------
        phi1, phi2 : array-like
            Stream coordinates.
        endpoints : SkyCoord[2], optional
            Endpoints for the great circle. If None, will be randomly chosen.
        hpxmap : np.ndarray, optional
            HEALPix map to check for valid pixels (should be same nside as used for output).
        hpxmap2 : np.ndarray, optional
            Second HEALPix map to check for valid pixels.
        phi1_check : list or array, optional
            List of phi1 values (e.g., [min, max, median]) to check for valid pixels (hpxmap&hpxmap2 > 0).
        max_trials : int
            Maximum number of random endpoint trials.
        Returns
        -------
        stream : SkyCoord
            Transformed coordinates.
        """
        self.endpoints = endpoints
        if self.endpoints is None:
            mask_config_default = {"phi1_check": "all", "mask_type": ["footprint"], "max_trials": 10}
            if mask_config is not None:
                for key in mask_config_default:
                    if key not in mask_config:
                        mask_config[key] = mask_config_default[key]

            self.endpoints = self._choose_endpoints(
                rng=rng, seed=seed, mask_config=mask_config, phi1=phi1, phi2=phi2
            )

        # Use Gala to create the stream coordinate frame
        frame = gc.GreatCircleICRSFrame.from_endpoints(
            self.endpoints[0], self.endpoints[1]
        )
        phi1 = np.array(phi1) * u.deg
        phi2 = np.array(phi2) * u.deg
        stream = coord.SkyCoord(phi1=phi1, phi2=phi2, frame=frame)

        return stream

    def _choose_endpoints(
        self, rng=None, seed=None, mask_config=None, phi1=None, phi2=None
    ):
        """
        Randomly choose endpoints for the great circle within the survey footprint.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator. If None, uses the default random generator with a seed.
        seed : int, optional
            Seed for the random number generator. Used only if rng is None.

        Returns
        -------
        endpoints : SkyCoord[2]
            Randomly chosen endpoints.
        """

        if rng is None:  # use the seed if provided
            rng = np.random.default_rng(seed)

        if (
            mask_config is None
        ):  # if no mask, choose random endpoints anywhere on the sky
            ra, dec = rng.uniform(0, 360, size=2), rng.uniform(-90, 90, size=2)
            endpoints = coord.SkyCoord(ra * u.deg, dec * u.deg)
            return endpoints

        mask = self._create_mask(mask_config.get("mask_type", ["footprint"]))
        pix = np.arange(len(mask))
        nside = hp.get_nside(mask)

        for _ in range(mask_config.get("max_trials", 10)):
            # Randomly select endpoints inside the footprint of the HEALPix maps
            pixels = rng.choice(pix[mask > 0], size=2, replace=False)
            ra, dec = hp.pix2ang(nside, pixels, lonlat=True)
            endpoints_trial = coord.SkyCoord(ra * u.deg, dec * u.deg)
            frame = gc.GreatCircleICRSFrame.from_endpoints(
                endpoints_trial[0], endpoints_trial[1]
            )

            phi1_check = mask_config.get("phi1_check", None)
            if (
                phi1_check is not None
            ):  # Verify that phi1_check values map to valid pixels
                if isinstance(phi1_check, (list, np.ndarray)):
                    print("Checking provided phi1 values for valid pixels.")
                    phi1_check = np.array(phi1_check) * u.deg
                    phi2_check = np.zeros_like(phi1_check) * u.deg
                elif phi1_check == "all":
                    print("Checking all provided phi1 values for valid pixels.")
                    if phi1 is None or phi2 is None:
                        raise ValueError(
                            "phi1 and phi2 must be provided if phi1_check is 'all'."
                        )
                    phi1_check = np.array(phi1) * u.deg
                    phi2_check = np.zeros_like(phi2) * u.deg
                else:
                    raise ValueError("phi1_check must be a list, array, or 'all'.")

                coords = coord.SkyCoord(phi1=phi1_check, phi2=phi2_check, frame=frame)
                ra_check = coords.icrs.ra.deg
                dec_check = coords.icrs.dec.deg
                pix_check = hp.ang2pix(nside, ra_check, dec_check, lonlat=True)

                if (
                    np.sum(mask[pix_check] > 0) / len(pix_check) > 0.99
                ):  # at least 99% of the points should be valid
                    endpoints = endpoints_trial
                    print(
                        f"Found valid endpoints: {endpoints_trial[0]}, {endpoints_trial[1]}"
                    )
                    break
                else:
                    print(
                        f"Endpoints {endpoints_trial[0]}, {endpoints_trial[1]} do not map to valid pixels for phi1_check."
                    )
            else:
                endpoints = endpoints_trial
                break
        if endpoints is None:
            raise RuntimeError(
                f"Could not find suitable endpoints after {mask_config.get('max_trials', 10)}."
            )
        return endpoints

    def _create_mask(self, mask_type):
        if isinstance(mask_type, str):
            mask_type = [mask_type]
        elif isinstance(mask_type, list):
            pass
        else:
            raise ValueError("mask_type must be a string or a list of strings.")

        # Find the minimum nside among the needed maps
        nside_min = None
        maps = {}

        if "footprint" in mask_type:
            mask_type.remove("footprint")
            mask_type.append("maglim_r")
            mask_type.append("maglim_g")

        for m in mask_type:
            if m == "maglim_g":
                nside = hp.get_nside(self.maglim_map_g)
                maps[m] = self.maglim_map_g
                if nside_min is None or nside < nside_min:
                    nside_min = nside
            elif m == "maglim_r":
                nside = hp.get_nside(self.maglim_map_r)
                maps[m] = self.maglim_map_r
                if nside_min is None or nside < nside_min:
                    nside_min = nside
            elif m == "ebv":
                nside = hp.get_nside(self.ebv_map)
                maps[m] = self.ebv_map
                if nside_min is None or nside < nside_min:
                    nside_min = nside
            else:
                raise ValueError(f"Unknown mask type: {m}")

        for m in maps:
            nside = hp.get_nside(maps[m])
            if nside != nside_min:
                maps[m] = hp.ud_grade(maps[m], nside_min)
                print(f"Upgrading {m} from nside {nside} to {nside_min}.")

        mask_map = np.ones(len(maps[mask_type[0]]), dtype=bool)
        for m in mask_type:
            if m in ["maglim_g", "maglim_r"]:
                mask_map &= maps[m] > 0
            elif m == "ebv":
                mask_map &= maps[m] < 0.2  # arbitrary high value to exclude bad data

        return mask_map

    def extinction(self, pix):
        """
        Estimate the exctinction value for given pixels of the sky.
        Args:
            pix (healpy pixels)

        Returns:
            extinction_g, extinction_r: magnitudes extinctions for the given pixels
        """
        ebv = self.ebv_map[pix].byteswap().view(self.ebv_map.dtype.newbyteorder())
        extinction_g = self.coeff_extinc[0] * ebv
        extinction_r = self.coeff_extinc[1] * ebv
        return extinction_g, extinction_r

    def _get_errors(self, pix, mag_g, mag_r):
        """_summary_

        Args:
            pix (healpy pixels): _description_
            mag_g : magnitude in g band
            mag_r : magnitude in r band

        Returns:
            magerr_g, magerr_r: magnitudes erros in each bands
        """
        # Look up the magnitude limit at the position of each star
        maglim_g = self.maglim_map_g[pix]
        maglim_r = self.maglim_map_r[pix]

        # Magnitude errors
        magerr_g = self._effective_errors(mag_g, maglim_g)
        magerr_r = self._effective_errors(mag_r, maglim_r)
        return magerr_g, magerr_r

    def _effective_errors(self, mag, maglim):
        """Take the saturation into account by using the error value in the bright end"""
        magerr = 10 ** (
            np.where(
                ((mag - maglim) <= -10) & (mag >= self.saturation),
                self.log_photo_error(-10),
                self.log_photo_error(mag - maglim),
            )
        )
        magerr = np.where(
            mag < self.saturation, 10 ** self.log_photo_error(-11), magerr
        )  # saturation at the bright end
        magerr += self.sys_error  # add systematic error
        return magerr

    def sample(self, **kwargs):
        """
        Sample observed magnitudes from estimated ones and thier errors.
        kwargs:
            mag_g (numpy array): g band magnitudes
            mag_r (numpy array): r band magnitudes
            magerr_g (numpy array): g band magnitudes errors
            magerr_r (numpy array): r band magnitudes errors
            rng (numpy.random.Generator, optional): Random number generator. If None, uses the default random generator with a seed.
            seed (int, optional): Seed for the random number generator. Used only if rng is None.
        Returns:
            mag_g_meas,mag_r_meas : sampled observed magnitudes
        """
        mag_g = kwargs.pop("mag_g")
        mag_r = kwargs.pop("mag_r")
        magerr_g = kwargs.pop("magerr_g")
        magerr_r = kwargs.pop("magerr_r")

        rng = kwargs.pop("rng", None)
        if rng is None:
            seed = kwargs.pop("seed", None)
            rng = np.random.default_rng(seed)

        # Sample the fluxes their errors
        flux_g_meas = StreamObserved.magToFlux(mag_g) + rng.normal(
            scale=self.getFluxError(mag_g, magerr_g)
        )
        flux_r_meas = StreamObserved.magToFlux(mag_r) + rng.normal(
            scale=self.getFluxError(mag_r, magerr_r)
        )
        # If the flux is negative, set the magnitude to 99 (not detected). Otherwise, convert the flux back to magnitude
        mag_g_meas = np.where(
            flux_g_meas > 0.0, StreamObserved.fluxToMag(flux_g_meas), "BAD_MAG"
        )
        mag_r_meas = np.where(
            flux_r_meas > 0.0, StreamObserved.fluxToMag(flux_r_meas), "BAD_MAG"
        )

        return mag_g_meas, mag_r_meas

    def detect_flag(self, pix, mag_r=None, mag_g=None, **kwargs):
        """
        Apply the survey selection over the given stars/pixels.
        Args:
            pix (Healpy pixels)
            mag_r (numpy array, optional): magnitude in r band. Defaults to None.
            mag_g (numpy array, optional): magnitude in g band. Defaults to None.
        kwargs:
            maglim0 (float, optional): magnitude limit in the initial completeness. Defaults to 25.0.
            saturation0 (float, optional): saturation limit in the initial completeness. Defaults to 16.4.
            clipping_bounds (tuple, optional): bounds to current magnitude limit. Defaults to (20.0, 30.0).
            rng (numpy.random.Generator, optional): Random number generator. If None, uses the default random generator with a seed.
            seed (int, optional): Seed for the random number generator. Used only if rng is None.
        Raises:
            ValueError: Must provide either mag_g or mag_r values.
        Returns:
            boolean list: 1 for detected stars, 0 for the others
        """
        # Default parameters
        maglim0 = kwargs.pop(
            "maglim0", 26.8
        )  # magnitude limit in the initial completeness
        saturation0 = kwargs.pop(
            "saturation0", 16.4
        )  # saturation limit in the initial completeness
        clipping_bounds = kwargs.pop(
            "clipping_bounds", (20.0, 30.0)
        )  # bounds to current magnitude limit
        rng = kwargs.pop("rng", None)
        if rng is None:
            seed = kwargs.pop("seed", None)
            rng = np.random.default_rng(seed)

        if mag_r is None and mag_g is None:
            raise ValueError("Must provide either mag_g or mag_r values.")

        # Select the appropriate magnitude and map depending on the band
        if mag_r is not None:
            mag = mag_r
            maglim_map = self.maglim_map_r[pix]
        else:
            mag = mag_g
            maglim_map = self.maglim_map_g[pix]

        compl = self._effective_completeness(
            mag, maglim_map, maglim0, saturation0, clipping_bounds
        )

        # Set the threshold using completeness 1-padded at the bright ends
        threshold = rng.uniform(size=len(mag)) <= compl

        return threshold

    def _effective_completeness(
        self, mag, maglim_map, maglim0, saturation0, clipping_bounds
    ):
        delta_mag = mag - np.clip(
            maglim_map, clipping_bounds[0], clipping_bounds[1]
        )  # difference between the mag and the maglim at the position of the object
        eq_mag = (
            delta_mag + maglim0
        )  # convert the delta mag to the equivalent mag at maglim0
        # Apply saturation condition: 1 padding for objects fainter than saturation but equivalent mag brighter than saturation0
        compl = np.where(
            (mag > self.saturation) & (eq_mag < saturation0),
            1.0,
            self.completeness(eq_mag),
        )  # 1 padded
        compl = np.where(
            mag < self.saturation, 0.0, compl
        )  # saturation at the bright end
        compl = np.where(
            (maglim_map < self.saturation) | np.isnan(maglim_map), 0.0, compl
        )  # not observed if the area is not covered
        return compl

    def _save_injected_data(self, data, folder):
        """
        Save the injected data to a CSV file.

        Parameters
        ----------
        data : pd.DataFrame
            Data to save.

        folder : str or Path, optional
            Path to the folder where the file will be saved. If not provided, a default folder is used.
        """

        if folder is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            folder = os.path.join(current_dir, "..", "data/outputs/")
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_name = folder + "data_injected.csv"
        print(f"Saving injected data to {file_name}")
        data.to_csv(file_name, index=False)

    @staticmethod
    def getCompleteness(filename):
        d = np.genfromtxt(
            filename,
            delimiter=",",
            names=True,
        )
        x = d["mag_r"]

        selection = "both"
        if selection == "detected":
            # Detection efficiency:
            y = d["eff_star"]
        elif selection == "classified":
            # Classification efficiency
            y = d["classifiction_eff"]
        elif selection == "both":
            # Detection and classification efficiency
            y = d["classification_detection_eff"]

        # Extend to saturation
        if min(x) > 16:
            x = np.insert(x, 0, 16.0)
            y = np.insert(y, 0, y[0])

        # Make efficiency go to zero at faint end
        if y[-1] > 0:
            x = np.insert(x, -1, x[-1] + 1)
            y = np.insert(y, -1, 0)

        f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=0.0)

        return f

    @staticmethod
    def getPhotoError(filename):
        """Photometric error model based on the delta-mag from the maglim and
        the photometric uncertainty estimated from the data

        Parameters
        ----------
        filename : photometric error file

        Returns
        -------
        interp : interpolation function (mag_err as a function of delta_mag)
        """
        d = np.genfromtxt(
            filename,
            delimiter=",",
            names=True,
        )

        x = d["delta_mag"]
        y = d["log_mag_err"]

        # Extend on the bright end
        if min(x) > -10.0:
            x = np.insert(x, 0, -10.0)
            y = np.insert(y, 0, y[0])

        f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=1.0)

        return f

    @staticmethod
    def magToFlux(mag):
        """
        Convert from an AB magnitude to a flux (Jy)
        """
        return 3631.0 * 10 ** (-0.4 * mag)

    @staticmethod
    def fluxToMag(flux):
        """
        Convert from flux (Jy) to AB magnitude
        """
        return -2.5 * np.log10(flux / 3631.0)

    @staticmethod
    def getFluxError(mag, mag_error):
        return StreamObserved.magToFlux(mag) * mag_error / 1.0857362

    def plot_inject(self, data, **kwargs):
        """
        Plot :
        - the injected stars into the footprint, with a label for observed / unobserved stars.
        - The HR diagram for True colors magnitudes.
        - The HR diagram for observed colors magnitudes.

        Args:
            data (dict, pandas dataframe): has to contain the following columns : (mag_g, mag_r, mag_g_meas,mag_r_meas )
        """
        sel = data["flag_detection"]  # Detected stars

        fig, ax = plt.subplots(1, 3, figsize=(14, 6))
        plt.subplots_adjust(left=0.075, right=0.98)

        ax[0].set_title("Injection of the stream into the footprint")

        x = np.linspace(49, 75, 200)
        y = np.linspace(-46, -25, 200)
        XX, YY = np.meshgrid(x, y)
        pix = np.arange(len(self.maglim_map_r))
        nside = hp.get_nside(self.maglim_map_r)
        ZZ = self.maglim_map_r[hp.ang2pix(nside, XX, YY, lonlat=True)]
        ZZ[ZZ < 0] = np.nan
        ax[0].pcolormesh(XX, YY, ZZ)
        mesh = ax[0].pcolormesh(XX, YY, ZZ, shading="auto")
        fig.colorbar(mesh, ax=ax[0], label="Map Value")

        ax[0].scatter(
            data["ra"], data["dec"], s=2, alpha=0.5, color="gray", label="Unobserved"
        )
        ax[0].scatter(
            data["ra"][sel],
            data["dec"][sel],
            s=4,
            alpha=1.0,
            color="black",
            label="Observed",
        )
        ax[0].set_xlabel("RA (deg)")
        ax[0].set_ylabel("Dec (deg)")
        ax[0].scatter(self.endpoints.ra.deg, self.endpoints.dec.deg, s=25, c="r")
        ax[0].legend()

        ax[1].set_title("HR diagram using True magnitudes")
        ax[1].scatter(
            data["mag_g"] - data["mag_r"],
            data["mag_g"],
            s=2,
            alpha=0.5,
            color="gray",
            label="Unobserved",
        )
        ax[1].scatter(
            (data["mag_g"] - data["mag_r"])[sel],
            data["mag_g"][sel],
            s=4,
            alpha=1.0,
            color="black",
            label="Observed",
        )
        ax[1].set_xlim(-0.5, 1.5)
        ax[1].set_ylim(30, 16)
        ax[1].set_xlabel("(g-r)")
        ax[2].set_ylabel("g")
        ax[1].legend()

        ax[2].set_title("HR diagram using sampled observed magnitudes")
        # Convert mag_r_meas and mag_g_meas to numeric, coercing errors to NaN
        data["mag_r_meas"] = pd.to_numeric(data["mag_r_meas"], errors="coerce")
        data["mag_g_meas"] = pd.to_numeric(data["mag_g_meas"], errors="coerce")
        mask = (data["mag_g_meas"] != "BAD_MAG") & (data["mag_r_meas"] != "BAD_MAG")
        ax[2].scatter(
            data["mag_g_meas"][mask] - data["mag_r_meas"][mask],
            data["mag_g_meas"][mask],
            s=2,
            alpha=0.5,
            color="gray",
            label="Unobserved",
        )
        ax[2].scatter(
            (data["mag_g_meas"] - data["mag_r_meas"])[sel & mask],
            data["mag_g_meas"][sel & mask],
            s=4,
            alpha=1.0,
            color="black",
            label="Observed",
        )
        ax[2].set_xlim(-0.5, 1.5)
        ax[2].set_ylim(30, 16)
        ax[2].set_xlabel("(g-r)")
        ax[2].set_ylabel("g")

        ax[2].legend()

        if kwargs.pop("save", None) is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "..", "data/outputs/")
            folder = kwargs.pop("folder", path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            existing_files = os.listdir(folder)
            num_fichier = len(
                [f for f in existing_files if f.startswith(f"injection_{self.survey}_")]
            )
            file_name = f"injection_{self.survey}__{num_fichier:03}"
            filename = os.path.join(folder, file_name + ".png")
            print(f"Writing data: {filename}...")
            plt.savefig(filename)
