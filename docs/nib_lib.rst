.. _nib_lib:

NiBabel
=======

`NiBabel`_ is a widely-used library for processing neuroimaging data. In order to
use NiBabel with highdicom, the ``nibabel`` python package must be installed
separately. Version 4.0.0 or later is required.

.. _nibabel_vol:

Volume Conversions
------------------

Highdicom supports conversions with several ``NiBabel.spatialimages.SpatialImage``
subclasses through the :meth:`highdicom.Volume.to_nibabel` and
:meth:`highdicom.Volume.from_nibabel` methods. Unlike highdicom, which uses the
"LPS" convention, Nibabel uses "RAS" convention for its image classes. This change
in convention is handled automatically by highdicom and requires no intervention
by the user.


Creating a NiBabel Image from a Volume:

.. code-block:: python

    import highdicom as hd


    vol = hd.Volume(...)

    nibabel_image = vol.to_nibabel(image_class='Nifti1Image')

Creating a volume from a NiBabel Image:

.. code-block:: python

    import nibabel as nib
    import highdicom as hd


    nibabel_image = nib.Nifti1Image(...)

    vol = hd.Volume.from_nibabel(
        nibabel_image=nibabel_image,
        coordinate_system='PATIENT',
        frame_of_reference_uid=None
    )


.. _`NiBabel`: https://nipy.org/nibabel/
