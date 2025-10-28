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
from .plotting import plot_stream_in_mask 
from .surveys import Survey 

class StreamInjector:
    """
    Injects observational effects into stream data for a given survey.
    
    This class handles the core injection logic while keeping survey data separate.
    All survey data is loaded once and cached, making multiple injections efficient.
    """

    def __init__(self, survey, **kwargs):
        """Initialize with survey configuration."""

        if isinstance(survey, str):
            self.survey = Survey.load(
                survey=survey,
                **kwargs
            )
        elif isinstance(survey, Survey):
            self.survey = survey
        else:
            raise ValueError("survey must be a string or Survey instance.")

    def inject(self, data, bands=['r', 'g'], **kwargs):
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
        # Load data
        data = self._load_data(data)

        # Set the seed for reproducibility
        seed = kwargs.pop("seed", None)
        rng = np.random.default_rng(seed)

        # Complete missing columns (ra/dec, magnitudes)
        data = self.complete_data(data, rng=rng, seed=seed, bands=bands, **kwargs)

        # Get HEALPix pixel indices
        nside = kwargs.pop("nside", 4096)
        pix = hp.ang2pix(nside, data["ra"], data["dec"], lonlat=True)

        # Process each band
        for band in bands:
            if band not in ['r', 'g']:
                raise ValueError("Currently only 'r' and 'g' bands are supported.")

            # Get extinction for this band
            nside_ebv = hp.get_nside(self.survey.ebv_map)
            if nside_ebv != nside:
                pix_ebv = hp.ang2pix(nside_ebv, data["ra"], data["dec"], lonlat=True)
            else:
                pix_ebv = pix
            extinction_band = self.survey.get_extinction(band, pixel=pix_ebv)

            # Get magnitude limits
            nside_maglim = hp.get_nside(self.survey.maglim_maps[band])
            if nside_maglim != nside:
                pix_maglim = hp.ang2pix(nside_maglim, data["ra"], data["dec"], lonlat=True)
            else:
                pix_maglim = pix

            # Calculate photometric errors
            mag_err = self.survey.get_photo_error(
                band, 
                data["mag_" + band] + extinction_band, 
                self.survey.get_maglim(band, pixel=pix_maglim)
            )

            # Sample measured magnitudes
            mag_meas = self.sample_measured_magnitudes(
                data["mag_" + band] + extinction_band, 
                mag_err, 
                rng=rng, 
                seed=seed, 
                **kwargs
            )

            # Add new columns
            new_columns = pd.DataFrame({
                "mag_" + band + "_meas": mag_meas,
                "magerr_" + band: mag_err,
            })
            
            # Reset indices and concatenate
            data = data.reset_index(drop=True)
            new_columns = new_columns.reset_index(drop=True)
            data = pd.concat([data, new_columns], axis=1)

            # Compute detection flag for r-band (reference band)
            if band == 'r':
                flag_completeness_r = self.detect_flag(
                    pix_maglim, 
                    mag=data["mag_r"] + extinction_band, 
                    band='r', 
                    rng=rng, 
                    seed=seed, 
                    **kwargs
                )
                    
        # Apply detection threshold
        if flag_completeness_r is None:
            if 'r' in bands:
                raise ValueError("flag_completeness_r must be computed for detection in r band.")
            else:
                raise ValueError("Detection flag requires 'r' band to be in bands.")

        # Check for negative fluxes (set to 'BAD_MAG')
        flag_r = (data["mag_r_meas"] != "BAD_MAG")

        # Combine flags
        flag_detection = flag_r & flag_completeness_r

        if 'g' in bands:
            flag_detection &= data["mag_g_meas"] != "BAD_MAG"

        # Apply SNR cuts if requested
        detection_mag_cut = kwargs.get("detection_mag_cut", ['g'])
        SNR_min = 5.0
        for band in detection_mag_cut:
            print("Applying detection cut on", band, "band with SNR >=", SNR_min)
            SNR = 1 / data["magerr_" + band]
            flag_detection &= SNR >= SNR_min

        data["flag_detection"] = flag_detection

        # Save if requested
        if kwargs.get("save"):
            self._save_injected_data(data, kwargs.get("folder", None))

        # Return data (do NOT store as instance attribute to avoid conflicts between runs)
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

    def complete_data(self, data, bands = ['r', 'g'], **kwargs):
        """
        Ensure the input data contains all required columns.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with all required columns.
        """

        required_columns = ["ra", "dec"] + [f"mag_{band}" for band in bands]

        rng = kwargs.pop("rng", None)
        seed = kwargs.pop("seed", None)

        if not ('ra' in data.columns and 'dec' in data.columns):
            if "phi1" not in data.columns or "phi2" not in data.columns:
                raise ValueError(
                    "Input data must contain either (ra, dec) or (phi1, phi2) columns."
                )
            
            # Convert coordinates (Phi1, Phi2) into (ra,dec)
            stream_coord = self.phi_to_radec(
                data["phi1"],
                data["phi2"],
                seed=seed,
                rng=rng,
                **kwargs,
            )
            data["ra"] = stream_coord.icrs.ra.deg
            data["dec"] = stream_coord.icrs.dec.deg
        
        # Sample missing magnitudes if needed
        mag_bands_missing = [band for band in bands if f"mag_{band}" not in data.columns]
        # to be implemented

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data must contain '{col}' column.")

        return data
        

    def phi_to_radec(
        self,
        phi1,
        phi2,
        gc_frame=None,
        seed=None,
        rng=None,
        mask_type=["footprint"],
        **kwargs,
    ):
        """
        Transform stream coordinates (phi1, phi2) to sky coordinates (RA, Dec).

        This method converts stream coordinates to celestial coordinates using a great circle
        frame. If no frame is provided, it automatically finds one randomly chosen such that a given percentile
        of the points lie within the mask defined with mask_type.

        Parameters
        ----------
        phi1, phi2 : array-like
            Stream coordinates in degrees.
        gc_frame : gala.coordinates.GreatCircleICRSFrame, optional
            Great circle coordinate frame. If None, will be automatically determined.
        seed : int, optional
            Random seed for reproducible frame selection.
        rng : numpy.random.Generator, optional
            Random number generator instance.
        mask_type : list of str, default ["footprint"]
            Types of masks to use for footprint validation.
            Options: ["footprint", "maglim_g", "maglim_r", "ebv"]

        Returns
        -------
        stream : astropy.coordinates.SkyCoord
            Sky coordinates in ICRS frame.

        Raises
        ------
        ValueError
            If phi1 and phi2 have different lengths or contain invalid values.
        RuntimeError
            If no suitable great circle frame could be found.

        Examples
        --------
        >>> phi1 = np.linspace(-10, 10, 1000)
        >>> phi2 = np.zeros_like(phi1)
        >>> coords = obj.phi_to_radec(phi1, phi2, seed=42)
        """
        # Input validation
        phi1_arr = np.asarray(phi1, dtype=float)
        phi2_arr = np.asarray(phi2, dtype=float)

        if phi1_arr.size == 0 or phi2_arr.size == 0:
            raise ValueError("phi1 and phi2 cannot be empty arrays")

        # Find or use provided great circle frame
        if gc_frame is None:
            gc_frame = self._find_gc_frame(
                rng=rng,
                seed=seed,
                mask_type=mask_type,
                phi1=phi1_arr,
                phi2=phi2_arr,
                **kwargs,
            )

        # Transform to sky coordinates
        phi1_deg = phi1_arr * u.deg
        phi2_deg = phi2_arr * u.deg
        stream_coord = coord.SkyCoord(phi1=phi1_deg, phi2=phi2_deg, frame=gc_frame)

        return stream_coord

    def _find_gc_frame(
        self,
        phi1=None,
        phi2=None,
        mask=None,
        mask_type=["footprint"],
        percentile_threshold=0.99,
        max_iter=1000,
        rng=None,
        seed=None,
        verbose=True,
        **kwargs,
    ):
        """
        Find a great circle frame such that a high fraction of points lie within the chosen mask.

        This method iteratively tries random great circle orientations until it finds
        one where at least `percentile_threshold` of the stream points fall within
        the survey mask.

        Parameters
        ----------
        phi1, phi2 : array-like
            Stream coordinates to validate against the mask.
        rng : numpy.random.Generator
            Random number generator instance.
        mask_type : list of str, default ["footprint"]
            Types of masks to combine for footprint validation.
        percentile_threshold : float, default 0.99
            Minimum fraction of points that must be within the mask.
        max_iter : int, default 100
            Maximum number of random trials.
        verbose : bool, default True
            Whether to print progress information.

        Returns
        -------
        gc_frame : gala.coordinates.GreatCircleICRSFrame or None
            Great circle frame, or None if no suitable frame found.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        # Create the mask if not provided
        if mask is None:
            healpix_mask = self._create_mask(mask_type, verbose=verbose)
        else:
            healpix_mask = mask
            if verbose:
                print("Using provided HEALPix mask for footprint checking.")

        # Do NOT store mask as instance attribute to avoid conflicts between runs
        # (each run may need a different mask)

        # If no mask is available, return a random great circle frame
        if healpix_mask is None:
            if verbose:
                print("No mask available, returning a random great circle frame.")
            end1 = self._random_uniform_skycoord(rng)
            end2 = self._random_uniform_skycoord(rng)
            gc_frame = gc.GreatCircleICRSFrame.from_endpoints(end1, end2)
            return gc_frame

        if phi1 is None or phi2 is None:
            raise ValueError("phi1 and phi2 must be provided if no mask is given.")

        # Iteratively try random great circle frames
        trials = 0
        while trials < max_iter:
            trials += 1

            # Generate random endpoints for the great circle
            end1 = self._random_uniform_skycoord(rng)
            end2 = self._random_uniform_skycoord(rng)
            gc_frame = gc.GreatCircleICRSFrame.from_endpoints(end1, end2)

            # Transform stream points to ICRS and check mask coverage
            pts_gc = coord.SkyCoord(
                phi1=phi1 * u.deg, phi2=phi2 * u.deg, frame=gc_frame
            )
            pts_icrs = pts_gc.transform_to("icrs")
            ra_all = pts_icrs.ra.deg
            dec_all = pts_icrs.dec.deg
            frac = self._fraction_inside_mask(ra_all, dec_all, healpix_mask)
            if frac >= percentile_threshold:
                if verbose:
                    print(
                        f"Found suitable great circle frame after {trials} trials with {frac*100:.2f}% points inside the mask."
                    )
                return gc_frame
        if verbose:
            print(
                f"Could not find a suitable great circle frame after {max_iter} trials."
            )
        return None

    def _random_uniform_skycoord(self, rng):
        """Generate a random point uniformly distributed on the sky."""
        z = rng.uniform(-1.0, 1.0)
        dec = np.degrees(np.arcsin(z))
        ra = rng.uniform(0.0, 360.0)
        return coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    def _fraction_inside_mask(self, ra_deg, dec_deg, healpix_mask):
        """
        Calculate the fraction of points that fall within valid mask regions.

        Parameters
        ----------
        ra_deg, dec_deg : array-like
            Coordinates in degrees.
        healpix_mask : array-like
            Boolean HEALPix mask array.

        Returns
        -------
        float
            Fraction of points inside mask (0.0 to 1.0).
        """
        nside = hp.get_nside(healpix_mask)
        pix_indices = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True)
        return np.count_nonzero(healpix_mask[pix_indices]) / len(pix_indices)

    def _create_mask(self, mask_type, verbose=True, ebv_threshold=0.2):
        """
        Create a combined boolean mask from specified mask types.

        Parameters
        ----------
        mask_type : str or list of str
            Type(s) of masks to combine. Options: ["footprint", "maglim_g", "maglim_r", "ebv"]
        ebv_threshold : float, default 0.2
            E(B-V) threshold for extinction mask.

        Returns
        -------
        numpy.ndarray
            Combined boolean mask array.

        Raises
        ------
        ValueError
            If mask_type is invalid or required maps are missing.
        """

        if isinstance(mask_type, str):
            mask_type = [mask_type]
        elif isinstance(mask_type, list):
            pass
        elif mask_type is None:
            if verbose:
                print("No mask_type provided to build mask.")
            return None
        else:
            raise ValueError("mask_type must be a string or a list of strings.")

        # Find the minimum nside among the needed maps
        nside_target = []
        maps = {}
        for m in mask_type:
            if 'maglim' in m:
                band = m.split('_')[-1]
                nside = hp.get_nside(self.survey.maglim_maps[band])
                maps[m] = self.survey.maglim_maps[band]
                nside_target.append(nside)
            elif m == "coverage" or m == "footprint":
                nside = hp.get_nside(self.survey.coverage)
                maps[m] = self.survey.coverage
                nside_target.append(nside)
            elif m == "ebv":
                nside = hp.get_nside(self.ebv_map)
                maps[m] = self.ebv_map
                nside_target.append(nside)
            else:
                raise ValueError(f"Unknown mask type: {m}")
        nside_min = min(nside_target)

        # Upgrade all maps to the minimum nside
        for m in maps:
            nside = hp.get_nside(maps[m])
            if nside != nside_min:
                maps[m] = hp.ud_grade(maps[m], nside_min)
                print(f"Upgrading {m} from nside {nside} to {nside_min}.")

        # Combine the masks
        mask_map = np.ones(len(maps[mask_type[0]]), dtype=bool)
        for m in mask_type:
            if 'maglim' in m: 
                band = m.split('_')[-1]
                mask_map &= maps[m] > self.survey.saturation[band]
            elif m == "coverage" or m == "footprint":
                mask_map &= maps[m] > 0.5  # covered regions
            elif m == "ebv":  # select low extinction regions
                mask_map &= maps[m] < ebv_threshold

        return mask_map

    def sample_measured_magnitudes(self, mag_true, mag_err, **kwargs):
        rng = kwargs.pop("rng", None)
        if rng is None:
            seed = kwargs.pop("seed", None)
            rng = np.random.default_rng(seed)

        # Sample the fluxes their errors
        flux_meas = StreamInjector.magToFlux(mag_true) + rng.normal(
            scale=self.getFluxError(mag_true, mag_err)
        )
        
        # If the flux is negative, set the magnitude to "BAD_MAG" (not detected). Otherwise, convert the flux back to magnitude
        mag_meas = np.where(
            flux_meas > 0.0, StreamInjector.fluxToMag(flux_meas), "BAD_MAG"
        )
   
        return mag_meas

    def detect_flag(self, pix, mag=None, band='r', **kwargs):
        """
        Apply the survey selection over the given stars/pixels.
        Args:
            pix (Healpy pixels)
            mag (numpy array, optional): magnitude in r band. Defaults to None.
            band (str): band to consider for detection. Defaults to 'r'.
        kwargs:
            rng (numpy.random.Generator, optional): Random number generator. If None, uses the default random generator with a seed.
            seed (int, optional): Seed for the random number generator. Used only if rng is None.
        Raises:
            ValueError: Must provide either mag_g or mag_r values.
        Returns:
            boolean list: 1 for detected stars, 0 for the others
        """
        
        rng = kwargs.pop("rng", None)
        if rng is None:
            seed = kwargs.pop("seed", None)
            rng = np.random.default_rng(seed)

  
        # Select the appropriate magnitude and map depending on the band
        maglim = self.survey.get_maglim(band, pixel=pix)

        compl = self.survey.get_completeness(band, mag, maglim)

        # Set the threshold using completeness
        threshold = rng.uniform(size=len(mag)) <= compl

        return threshold

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
        return StreamInjector.magToFlux(mag) * mag_error / 1.0857362


    def plot_stream_in_mask(self, **kwargs):
        """
        Plot the stream over the footprint mask.
        """
        if not hasattr(self, "data"):
            raise ValueError("No data found. Please run the 'inject' method first.")

        fig,ax = plot_stream_in_mask(self.data["ra"], self.data["dec"], self.mask, output_folder=kwargs.get("output_folder"))
        return fig, ax