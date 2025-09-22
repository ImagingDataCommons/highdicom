.. _uids:

Unique Identifiers (UIDs)
=========================

DICOM uses its own form of unique identifier known as a UID. This is similar to
other UUID concepts used elsewhere in computer science, with some minor
differences in implementation. In principal a DICOM UID should be *globally*
unique, i.e. unique over all DICOM files that exist in the world.

A DICOM UID is a sequence of digits and dots. They look something like this:

``1.2.826.0.1.3680043.10.511.3.34077923876555658791168239952622729``

There is a specific structure to UIDs, but most of these details are not
important for most users of `highdicom`. However, it is important to understand
that the first part of the UID (the *prefix*) corresponds to the issuer of the
UID. *Highdicom* has its own issuer prefix (``1.2.826.0.1.3680043.10.511.3.``),
issued by the DICOM Standard committee. It is fine to use this prefix in
settings such as research and development, but it would not be appropriate to
use this for production, especially to a real clinical environment. In such
settings, you should use your own organization's prefix. Similarly, *pydicom*
has its own prefix (``1.2.826.0.1.3680043.8.498.``).

Generating UIDs
---------------

You can generate a UID using the highdicom prefix by constructing an object of
type :class:`highdicom.uid.UID`. The generated UID should be globally unique.
you can alternatively generate a UID with an alternative prefix using
``pydicom.uid.generate_uid()`` function, or your organization's own method for
generating UIDs.

.. code-block:: python

    import highdicom as hd
    import pydicom


    # Constuct a UID using highdicom's prefix
    uid = hd.UID()

    # Constuct a UID using your own prefix
    uid = hd.UID(pydicom.generate_uid('1.2.826.0.1.3680043.10.511.4.')))

Study, Series, and SOP Instance UIDs
------------------------------------

All DICOM objects have three critical UIDs:

* The *StudyInstanceUID* is shared by all objects within a study (and by no
  other objects, globally).
* The *SeriesInstanceUID* is shared by all objects within a series (and by no
  other objects, globally).
* The *SOPInstanceUID* is unique to the individual object (instance).

All DICOM objects that you create with *highdicom* require you to explicitly
specify these UIDs when you construct them using the ``study_instance_uid``,
``series_instance_uid``, and ``sop_instance_uid`` parameters. Here are some
considerations for specifying UIDs:

* It would be unusual to create entirely new studies with *highdicom*,
  therefore the ``study_instance_uid`` will typically be copied from an
  existing object. If you are creating a derived object (such as a
  segmentation, parametric map, structured report, preentation state, etc) from
  an existing image or image series, use the *StudyInstanceUID* attribute of
  the source image/series.
* You will almost always be generating a new series with *highdicom*, not
  adding to an existing series. A common mistake we see is to specify the
  series instance UID of the source series when creating a derived object. This
  is almost always wrong. Among other constraints, a series can only consist of
  objects of a single modality, so for example a segmentation (modality
  ``'SEG'``) of a CT image (modality ``'CT'``) can never belong in the same
  series as the original CT image. However you may in some situations be
  generating a new series consisting of multiple instances. In this situation,
  you shoudl generate a single series instance UID and re-use it for all items
  in the series.
* The ``sop_instance_uid`` should always be unique to every single object you
  create, even those within a given series.
