.. _sitk_lib:

SimpleITK
=========

`SimpleITK`_ provides a simplified interface for the algorithms and data structures
of `ITK`_. In order to use ITK with highdicom, the ``SimpleITK`` python package must
be installed separately. Version 2.2.1 or later is required.

.. _sitk_vol:

Volume Conversions
------------------

Highdicom supports conversions with the ``SimpleITK.Image`` class through the
:meth:`highdicom.Volume.to_simpletk` and :meth:`highdicom.Volume.from_simpletk`
methods. Like highdicom, SimpleITK uses the "LPS" convention. However, when
converting to and from NumPy arrays, SimpleITK reverses the order of
dimensions. This permutation is handled automatically by highdicom and requires
no intervention by the user.


Creating a SimpleITK Image from a Volume:

.. code-block:: python

    import highdicom as hd


    vol = hd.Volume(...)

    simpleitk_image = vol.to_simpleitk()

Creating a volume from a SimpleITK Image:

.. code-block:: python

    import SimpleITK as sitk
    import highdicom as hd


    simpleitk_image = sitk.Image(...)

    vol = hd.Volume.from_simpleitk(
        simpleitk_image=simpleitk_image,
        coordinate_system='PATIENT',
        frame_of_reference_uid=None
    )


.. _`ITK`: https://itk.org/
.. _`SimpleITK`: https://simpleitk.org/
