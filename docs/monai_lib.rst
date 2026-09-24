.. _monai_lib:

MONAI
=====

`MONAI`_ is a widely-used library for training and evaluating models on medical imaging
data. In order to use MONAI with highdicom, the ``monai`` python package must be
installed separately. Version 1.3.0 or later is required.

.. _monai_vol:

Volume Conversions
------------------

Highdicom supports conversions with the ``monai.data.MetaTensor`` class through the
:meth:`highdicom.Volume.to_monai` and :meth:`highdicom.Volume.from_monai` methods.
Unlike highdicom, which exclusively uses the "LPS" convention, MONAI supports both
"RAS" and "LPS" MetaTensors. By default, highdicom converts to the "RAS" convention.

Creating an MONAI MetaTensor from a Volume:

.. code-block:: python

    import highdicom as hd


    vol = hd.Volume(...)

    metatensor = vol.to_monai()

Creating a volume from a MONAI MetaTensor:

.. code-block:: python

    import monai
    import highdicom as hd


    metatensor = monai.data.MetaTensor(...)

    vol = hd.Volume.from_monai(
        metatensor=metatensor,
        coordinate_system='PATIENT',
        frame_of_reference_uid=None,
        channels=None,
        channel_dim=None
    )


.. _`MONAI`: https://project-monai.github.io/
