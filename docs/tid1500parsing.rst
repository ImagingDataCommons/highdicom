Parsing Measurement Reports
===========================

In addition to the ability to create TID 1500 Structured Reports, *highdicom*
also includes functionality to help you find and extract information from
existing SR documents in this format.

First you must get the SR dataset into the format of a highdicom class. You
can do this using the ``srread()`` function:
highdicom SR object.

.. code-block:: python
    import highdicom as hd

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document.dcm")

Alternatively, if you already have a ``pydicom.Dataset`` in memory, you can use
the relevant ``from_dataset`` method like this:

.. code-block:: python
    import pydicom
    import highdicom as hd

    sr_dataset = pydicom.dcmread("data/test_files/sr_document.dcm")

    # Use the appropriate class depending on the specific IOD, here it is a
    # Comprehensive3DSR
    sr = hd.sr.Comprehensive3DSR.from_dataset(sr_dataset)

If the Structured Report conforms to the TID 1500 measurement report template,
when you access the ``content`` property, a
:class:`highdicom.sr.MeasurementReport` object will be returned. Otherwise,
a general :class:`highdicom.sr.ContentSequence` object is returned.

The resulting :class:`highdicom.sr.MeasurementReport` object has methods that
allow you to find and access the content of the report conveniently.

Searching For Measurement Groups
--------------------------------

To search for measurement groups, the :class:`highdicom.sr.MeasurementReport`
class has
:method:`highdicom.sr.MeasurementReport.get_image_measurement_groups`,
:method:`highdicom.sr.MeasurementReport.get_planar_roi_measurement_groups`, and
:method:`highdicom.sr.MeasurementReport.get_volumetric_roi_measurement_groups`
methods, each of which returns a list of the measurement groups of the three
different types from the structured SR. You can additionally provide filters
to return only those measurement groups that meet certain criteria.

The available search criteria include: tracking UID, finding type, finding
site, referenced SOP instance UID, and referenced SOP class UID. Additionally
for 
:method:`highdicom.sr.MeasurementReport.get_planar_roi_measurement_groups`, and
:method:`highdicom.sr.MeasurementReport.get_volumetric_roi_measurement_groups`
it also possible to filter by graphic type and reference type (how the ROI
is specified in the measurement group).

For example:

.. code-block:: python
    import highdicom as hd
    from pydicom.sr.codedict import codes

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document.dcm")

    # Get a list of all measurement with finding type "tumor" and
    # finding site "lung"
    groups = sr.content.get_image_measurement_groups(
        finding_type=codes.SCT.Tumor,
        finding_sites=codes.SCT.Lung,
    )


Accessing Data in Measurement Groups
------------------------------------

Searching for Measurements
--------------------------

Accessing Data in Measurements
------------------------------

Searching for Evaluations
-------------------------
