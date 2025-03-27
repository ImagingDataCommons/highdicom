.. _remote:

Reading from Remote Filesystems
===============================

Functions like ``dcmread`` from pydicom and :meth:`highdicom.imread`,
:meth:`highdicom.seg.segread`, :meth:`highdicom.sr.srread`, and
:meth:`highdicom.ann.annread` from highdicom can read from any object that
exposes a "file-like" interface. Many alternative and remote filesystems have
Python clients that expose such an interface, and therefore can be read from
directly.

Coupling this with the :ref:`"lazy" frame retrieval <lazy>` option is
especially powerful, allowing frames to be retrieved from the remote filesystem
only as and when they are needed. This is particularly useful for large
multiframe files such as those found in slide microscopy or multi-segment
binary or fractional segmentations. However, the presence of offset tables in
the files is important for this to be effective (see explanation
in :ref:`lazy`).

Here we give some simple examples of how to do this using two popular cloud
storage providers: Google Cloud Storage (GCS) and Amazon Web Services (AWS) S3
storage. These are the two storage mechanisms underlying the Imaging Data
Commons (`IDC`_), a large repository of public DICOM images. It should be
possible to achieve the same effect with other filesystems, as long as there is
a Python client library that exposes a "file-like" interface.

Google Cloud Storage (GCS)
--------------------------

Blobs within Google Cloud Storage buckets can be accessed through a "file-like"
interface using the official Python SDK (installed through the
``google-cloud-storage`` PyPI package).

In this first example, we use lazy frame retrieval to load only a specific
spatial patch from a large whole slide image from the IDC.

.. code-block:: python

  import numpy as np
  import highdicom as hd

  # Additional libraries (install these separately)
  import matplotlib.pyplot as plt
  from google.cloud import storage


  # Create a storage client and use it to access the IDC's public data package
  client = storage.Client.create_anonymous_client()
  bucket = client.bucket("idc-open-data")

  # This is the path (within the above bucket) to a whole slide image from the
  # IDC collection called "CCDI MCI"
  blob = bucket.blob(
      "763fe058-7d25-4ba7-9b29-fd3d6c41dc4b/210f0529-c767-4795-9acf-bad2f4877427.dcm"
  )

  # Read directly from the blob object using lazy frame retrieval
  im = hd.imread(
      blob.open(mode="rb"),
      lazy_frame_retrieval=True
  )

  # Grab an arbitrary region of tile full pixel matrix
  region = im.get_total_pixel_matrix(
      row_start=15000,
      row_end=15512,
      column_start=17000,
      column_end=17512,
      dtype=np.uint8
  )

  # Show the region
  plt.imshow(region)
  plt.show()

.. figure:: images/slide_screenshot.png
   :width: 512px
   :alt: Image of retrieved slide region
   :align: center

   Figure produced by the above code snippet showing an arbitrary spatial
   region of a slide loaded directly from a Google Cloud bucket

As a further example, we use lazy frame retrieval to load only a specific set
of segments from a large multi-organ segmentation of a CT image in the IDC
stored in binary format (meaning each segment is stored using a separate set of
frames). See :ref:`seg` for more information on working with DICOM
segmentations.

.. code-block:: python

  import highdicom as hd

  # Additional libraries (install these separately)
  from google.cloud import storage


  # Create a storage client and use it to access the IDC's public data package
  client = storage.Client.create_anonymous_client()
  bucket = client.bucket("idc-open-data")

  # This is the path (within the above bucket) to a segmentation of a CT series
  # containing a large number of different organs
  blob = bucket.blob(
      "3f38511f-fd09-4e2f-89ba-bc0845fe0005/c8ea3be0-15d7-4a04-842d-00b183f53b56.dcm"
  )

  # Open the blob with "segread" using the "lazy frame retrieval" option
  seg = hd.seg.segread(
      blob.open(mode="rb"),
      lazy_frame_retrieval=True
  )

  # Find the segment number corresponding to the liver segment
  selected_segment_numbers = seg.get_segment_numbers(segment_label="Liver")

  # Read in the selected segments lazily
  volume = seg.get_volume(
      segment_numbers=selected_segment_numbers,
      combine_segments=True,
  )

This works because running the ``.open("rb")`` method on a Blob object returns
a `BlobReader`_ object, which has a "file-like" interface
(specifically the ``seek``, ``read``, and ``tell`` methods). If you can provide
examples for reading from storage provided by other cloud providers, please
consider contributing them to this documentation.

Amazon Web Services S3
----------------------

The `smart_open`_ package wraps an S3 client to expose a "file-like"
interface for accessing blobs. It can be installed with ``pip install
'smart_open[s3]'``.

In order to be able to access open IDC data without providing AWS credentials,
it is necessary to configure your own client object such that it does not
require signing. This is demonstrated in the following example, which repeats
the GCS from above using the counterpart of the same blob on AWS S3 (each DICOM
file in the IDC is stored in two places, one on GSC and the other on S3). If
you are accessing private files on S3, these steps will be different (consult
the ``smart_open`` documentation for details).

.. code-block:: python

  import boto3
  from botocore import UNSIGNED
  from botocore.config import Config
  import smart_open

  import numpy as np
  import highdicom as hd
  import matplotlib.pyplot as plt


  # Configure a client to avoid the need for AWS credentials
  s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

  # URL to an IDC CT image on AWS S3
  url = 's3://idc-open-data/763fe058-7d25-4ba7-9b29-fd3d6c41dc4b/210f0529-c767-4795-9acf-bad2f4877427.dcm'

  # Read the imge directly from the blob
  im = hd.imread(
      smart_open.open(url, mode="rb", transport_params=dict(client=s3_client)),
      lazy_frame_retrieval=True,
  )

  # Grab an arbitrary region of tile full pixel matrix
  region = im.get_total_pixel_matrix(
      row_start=15000,
      row_end=15512,
      column_start=17000,
      column_end=17512,
      dtype=np.uint8
  )

  # Show the region
  plt.imshow(region)
  plt.show()

The ``smart_open`` package can also wrap many other filesystems in this way,
including Microsoft Azure, Hadoop distributed filesystem (HDFS), gzipped local
files, files over ssh/scp/sftp, and more. In all cases, be aware that the
mechanics of the underlying retrieval, as well as configuration such as
buffering and chunk size, can have a significant impact on the performance of
lazy frame retrieval.

.. _IDC: https://portal.imaging.datacommons.cancer.gov/
.. _BlobReader: https://cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.fileio.BlobReader
.. _smart_open: https://github.com/piskvorky/smart_open
