.. _legacy:

Legacy Converted Enhanced Images
================================

Many medical images, including CT, MRI, and PET consist of set of 2D frames.
When the DICOM standard was originally developed, such series were typically
stored with each individual 2D frame in a separate DICOM instance (and
therefore a separate DICOM file). Many users will be familiar with MRI, PET,
and CT images stored in this way using the CT Image
("1.2.840.10008.5.1.4.1.1.2"), MR Image ("1.2.840.10008.5.1.4.1.1.4"), and
Positron Emission Tomography Image ("1.2.840.10008.5.1.4.1.1.128") SOP Classes,
with tens or hundreds of files needed to store a single series.

More recently, new additions to the standard prefer a "multiframe" arrangement
where multiple 2D frames are stored as individual frames within a single DICOM
instance. Newer modalities, such as optical coherence tomography, whole slide
microscopy images, or digital breast tomosynthesis (DBT) images, were designed
using these "multiframe" formats from the outset. Furthermore "enhanced"
multiframe versions of the older modalities (PET, CT, MRI) were also defined,
leading to the Enhanced PET Image, Enhanced CT Image, and Enhanced MR Image SOP
Classes. However, adoption of these Enhanced SOP Classes by manufacturers for
established modalities has been slow: and most PET, CT, and MR images you see
still use the original "legacy" SOP Classes.

Because the multiframe formats are generally easier to store and work with, a
common requirement is to convert old legacy (single-frame) series of images to
the corresponding new multiframe/enhanced format. Unfortunately, this is not
typically possible. The new enhanced formats made a number of other changes in
addition to the number of frames in an instance, and there is not usually
enough information in the legacy instances to populate all the required
attributes correctly.

To get around this problem, a set of "Legacy Converted Enhanced" SOP Classes
were created. These share many characteristics with the new Enhanced images,
but are designed such that it is possible to convert existing legacy series
to them. They also provide a mechanism to explicitly encode references to
the original source images.

Highdicom provides Python classes for these legacy converted enhanced images,
enabling their construction from existing legacy (single-frame) instances:

- :class:`highdicom.legacy.LegacyConvertedEnhancedMRImage` for MR images stored
  in the legacy "MR Image Storage" SOP Class.
- :class:`highdicom.legacy.LegacyConvertedEnhancedCTImage` for CT images stored
  in the legacy "CT Image Storage" SOPClass.
- :class:`highdicom.legacy.LegacyConvertedEnhancedPETImage` for PET images
  stored in the legacy "Positron Emission Tomography" SOPClass.


Basic Conversion
----------------

Basic conversion of legacy to enhanced format is very straightforward. You just
pass the legacy instances as a list (order is unimportant) and specify UIDs, an
instance number, and a series number for the new instance. Generally it is
recommended to choose a series description too, but if you don't, the original
series description of the legacy series will be used with "(enhanced
conversion)" as a suffix unless adding that suffix would go beyond the
character limit for series descriptions (64 characters).


.. code-block:: python

  import highdicom as hd
  from pydicom import dcmread
  from pydicom.data import get_testdata_file


  # Use this series of files from the pydicom test data
  legacy_ct_files = [
      get_testdata_file('dicomdirtests/77654033/CT2/17136'),
      get_testdata_file('dicomdirtests/77654033/CT2/17196'),
      get_testdata_file('dicomdirtests/77654033/CT2/17166'),
  ]

  # Read in the files
  ct_series = [dcmread(f) for f in legacy_ct_files]

  # Use the class constructor to perform the conversion
  multiframe = hd.legacy.LegacyConvertedEnhancedCTImage(
      ct_series,
      series_number=1,
      instance_number=1,
      series_instance_uid=hd.UID(),
      sop_instance_uid=hd.UID(),
      series_description="Enhanced Test Files",
  )

  # Save out the new multiframe conversion
  multiframe.save_as("legacy_converted_ct.dcm")


Transcoding
-----------

By default, the new instance will keep the transfer syntax of the legacy
datasets. If the legacy datasets are compressed, highdicom will simply re-use
the existing compressed pixel data without decoding it and combine the
compressed frames together.

Alternatively, you can opt to choose a new transfer syntax for the new
instance, in which case highdicom will decode and re-encode the pixel data from
the legacy instances. Since this can be slow, you can optionally specify a
non-zero number of sub-processes to perform this operation using the
``workers`` parameter. When using multiple workers, you must place your code
within a ``if __name__ == "__main__":`` guard (this is a requirement wherever
you use multiprocessing in Python).

.. code-block:: python

  import highdicom as hd
  from pydicom import dcmread
  from pydicom.uid import RLELossless
  from pydicom.data import get_testdata_file


  if __name__ == "__main__":
      # Use this series of files from the pydicom test data
      legacy_ct_files = [
          get_testdata_file('dicomdirtests/77654033/CT2/17136'),
          get_testdata_file('dicomdirtests/77654033/CT2/17196'),
          get_testdata_file('dicomdirtests/77654033/CT2/17166'),
      ]

      # Read in the files
      ct_series = [dcmread(f) for f in legacy_ct_files]

      # Create multiframe instance using lossless RLE compression
      multiframe = hd.legacy.LegacyConvertedEnhancedCTImage(
          ct_series,
          series_number=1,
          instance_number=1,
          series_instance_uid=hd.UID(),
          sop_instance_uid=hd.UID(),
          series_description="Enhanced Test Files",
          transfer_syntax_uid=RLELossless,
          workers=8,  # 8 conversion sub-processes
      )

      # Save out the new multiframe conversion
      multiframe.save_as("legacy_converted_ct.dcm")


Including Multiple Series
-------------------------

Although a Legacy Converted Enhanced image most often consists of legacy
datasets from a single series, this is not actually a limitation. In some
situations, you may wish to include images from multiple series in one
multiframe file. This is allowed if certain conditions are met, such as images
having the same size, pixel representation, photometric interpretation, etc.

Sorting Frames
--------------

Highdicom automatically sorts the legacy datasets before placing them into the
multiframe instance. The default behavior sorts first by the "SeriesNumber"
attribute (for cases with multiple series), then by the "InstanceNumber"
attribute, then by the "SOPInstanceUID" attribute (to give a predictable sort
order in cases where instance number is not present). If you prefer an
alternative sorting scheme, you can pass a callable to the ``sort_key``
argument of the form accepted by the Python built-in ``sorted`` function. For
example, to sort by ``KVP`` and then ``SliceLocation``:

.. code-block:: python

  import highdicom as hd
  from pydicom import dcmread
  from pydicom.data import get_testdata_file


  # Use this series of files from the pydicom test data
  legacy_ct_files = [
      get_testdata_file('dicomdirtests/77654033/CT2/17136'),
      get_testdata_file('dicomdirtests/77654033/CT2/17196'),
      get_testdata_file('dicomdirtests/77654033/CT2/17166'),
  ]

  # Read in the files
  ct_series = [dcmread(f) for f in legacy_ct_files]

  # Create multiframe instance with custom sort key
  multiframe = hd.legacy.LegacyConvertedEnhancedCTImage(
      ct_series,
      series_number=1,
      instance_number=1,
      series_instance_uid=hd.UID(),
      sop_instance_uid=hd.UID(),
      series_description="Enhanced Test Files",
      sort_key=lambda dcm: (dcm.KVP, dcm.SliceLocation),
  )

  # Save out the new multiframe conversion
  multiframe.save_as("legacy_converted_ct.dcm")


Requiring Volumes
-----------------

The ``require_volume`` option allows you to specify that highdicom should only
convert series that consist of parallel, regularly-spaced frames (i.e. those
that could be used to form a volume). This is entirely optional, Legacy
Converted Enhanced images that are not volumes are still entirely valid
according to the DICOM standard.
