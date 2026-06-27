.. _itk_lib:

ITK
===

`ITK`_ is a widely-used library for volumetric image processing. In order to
use ITK with highdicom, the ``itk`` python package must be installed separately.
Version 5.4.0 or later is required.

.. _itk_vol:

Volume Conversions
------------------

Highdicom supports conversions with the ``itk.Image`` class through the
:meth:`highdicom.Volume.to_itk` and :meth:`highdicom.Volume.from_itk` methods.
Like highdicom, ITK uses the "LPS" convention. However, when converting to and
from NumPy arrays, ITK reverses the order of dimensions. This permutation is
handled automatically by highdicom and requires no intervention by the user.

Creating an ITK Image from a Volume:

.. code-block:: python

    import highdicom as hd


    vol = hd.Volume(...)

    itk_im = vol.to_itk()

Creating a volume from an ITK Image:

.. code-block:: python

    import itk
    import highdicom as hd


    itk_im = itk.image(...)

    vol = hd.Volume.from_itk(
        itk_im=itk_im,
        coordinate_system='PATIENT',
        frame_of_reference_uid=None
    )


.. _`ITK`: https://itk.org/
