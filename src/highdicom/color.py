import logging
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image, ImageCms
from PIL.ImageCms import (
    applyTransform,
    getProfileDescription,
    getProfileName,
    ImageCmsProfile,
    ImageCmsTransform,
    isIntentSupported,
)
from pydicom.dataset import Dataset


logger = logging.getLogger(__name__)


class ColorManager(object):

    """Class for color management using ICC profiles."""

    def __init__(self, metadata: Dataset):
        """Construct color manager object.

        Parameters
        ----------
        metadata: Dataset
            Metadata of a color image

        Raises
        ------
        AttributeError
            When attributes ICCProfile or SamplesPerPixels are not found in
            `metadata`
        ValueError
            When value of attribute SamplesPerPixels is not ``3`` or when
            value of attribute ICCProfile cannot be read.

        """
        try:
            if metadata.SamplesPerPixel != 3:
                raise ValueError(
                    'Metadata indicates that instance does not represent '
                    'a color image.'
                )
        except AttributeError:
            raise AttributeError(
                'Metadata indicates that instance does not represent an image.'
            )
        try:
            icc_profile = metadata.ICCProfile
        except AttributeError:
            try:
                if len(metadata.OpticalPathSequence) > 1:
                    # This should not happen in case of a color image, but
                    # better safe than sorry.
                    logger.warning(
                        'metadata describes more than one optical path'
                    )
                icc_profile = metadata.OpticalPathSequence[0].ICCProfile
            except (IndexError, AttributeError):
                raise AttributeError('No ICC Profile found in image metadata.')
        self._icc_transform = self._build_icc_transform(icc_profile)

    def transform_frame(self, array: np.ndarray) -> np.ndarray:
        """Transforms a frame by applying the ICC profile.

        Parameters
        ----------
        array: numpy.ndarray
            Pixel data of a color image frame in form of an array with
            dimensions (Rows x Columns x SamplesPerPixel)

        Returns
        -------
        numpy.ndarray
            Color corrected pixel data of a image frame in form of an array
            with dimensions (Rows x Columns x SamplesPerPixel)

        Raises
        ------
        ValueError
            When `array` does not have 3 dimensions and thus does not represent
            a color image frame.

        """
        if array.ndim != 3:
            raise ValueError(
                'Array has incorrect dimensions for a color image frame.'
            )
        image = Image.fromarray(array)
        applyTransform(image, self._icc_transform, inPlace=True)
        return np.asarray(image)

    @staticmethod
    def _build_icc_transform(icc_profile: bytes) -> ImageCmsTransform:
        """Builds an ICC Transformation object.

        Parameters
        ----------
        icc_profile: bytes
            ICC Profile

        Returns
        -------
        PIL.ImageCms.ImageCmsTransform
            ICC Transformation object

        """
        profile: bytes
        try:
            profile = ImageCmsProfile(BytesIO(icc_profile))
        except OSError:
            raise ValueError('Cannot read ICC Profile in image metadata.')
        name = getProfileName(profile).strip()
        description = getProfileDescription(profile).strip()
        logger.debug(f'found ICC Profile "{name}": "{description}"')

        logger.debug('build ICC Transform')
        intent = ImageCms.INTENT_RELATIVE_COLORIMETRIC
        if not isIntentSupported(
            profile,
            intent=intent,
            direction=ImageCms.DIRECTION_INPUT
        ):
            raise ValueError(
                'ICC Profile does not support desired '
                'color transformation intent.'
            )
        return ImageCms.buildTransform(
            inputProfile=profile,
            outputProfile=ImageCms.createProfile('sRGB'),
            inMode='RGB',  # according to PS3.3 C.11.15.1.1
            outMode='RGB'
        )
