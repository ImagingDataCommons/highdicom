.. _compression:

Compression and Transfer Syntaxes
=================================

When creating images with *highdicom*, you will usually want to use some form
of compression to reduce the size of the files. Various compression codecs are
available within DICOM, and *highdicom*supports many of them. Although you may
need to install further depenencies to enable them. Each compression method is
identified by a *TransferSyntaxUID* from a set of well-known :ref:`uids`.

When constructing an image, you can instruct *highdicom* to compress it by
passing the relevant UID to the ``transfer_syntax_uid`` argument of the
contructor. By default, the ``ExplicitVRLittleEndian`` will be used, which is
an uncompressed trasnfer syntax.

You can either pass the UID itself as a string, or, more conveniently, import
the values from the ``pydicom.uid`` module (e.g. ``pydicom.uid.JPEG2000``).

Considerations
--------------

The most important conisderations when deciding what codec to use are:

- *Lossy or lossless*: Lossy compression methods discard information to improve
  compression, whereas lossless methods are fully reversible and retain all
  information in the input image, but usually result in a larger file.
- *Compression performance*. Some codecs are better than others at comoression
  data, giving larger or smaller file sizees, or better or lower quality
  compressed images. Some codecs will give better perforamnce on certain types
  of images.
- *Speed*. Some codecs are faster at compressing images than others. Similarly,
  some are quicker to decompress than other.
- *Support*. Some codecs are easily read by most other viewers and tools,
  whereas others are not.

Suuported Transfer Syntaxes
---------------------------

- *Explicit VR Little Endian* and *Implicit VR Little Endian*. These are
  uncompressed transfer syntanxes.
- *JPEG 8-bit baseline* (``pydicom.uid.JPEGBaseline8Bit``): A very
  widely-supported lossy codec, but it only supports 8 bit images.
- *JPEG 2000* (``pydicom.uid.JPEG2000``): Another, newer,
  widely-supported lossy codec. It supports any bitdepth up to 37
  bits and should achieve better compression than the baseline JPEG, but may be
  slower.
- *JPEG 2000 Lossless* (``pydicom.uid.JPEG2000Lossless``): A widely-supported
  and high performing lossless codec for any bitdepth. However, it may be
  rather slow.
- *JPEG LS Lossless* (``pydicom.uid.JPEGLSLossless``): A lossless codec for 8
  or 16 bit images. The major advantage of this over JPEG2000 is that it is
  often a much faster codec for a similar level of compression. Unfortunately,
  it is not widely supported.
- *RLE Lossless* (``pydicom.uid.RLELossless``): A DICOM-specific run length
  encoding scheme for lossless compression of 8 or 16 bit images. It is
  generally well supported by DICOM tooling. This will be particularly
  effective for images that have large areas of homogeneous intensities (such
  as segmentations).
- *DeflateImageFrameComression* (``pydicom.uid.DeflateImageFrameCompression``):
  Compression using the DEFLATE algorithm (via ``zlib``). This will do well in
  the same situations as RLE, but often perform better. Despite being a
  very common codec outside of DICOM, it was only introduced to the standard in
  2024 and therefore very few DICOM tools support it.

``JPEG2000Lossless`` is a good default choice for most situations. If you are
operating in a well-controlled research environment where you can ensure their
support among all other tols, there may be advantages to using
``JPEGLSLossless`` (for images) and ``DeflateImageFrameCompression`` (for
segmentations). If you can accept some information loss, ``JPEG2000`` is a good
choice.

Required Depenencies
--------------------

``pylibjpeg`` is required for compression and decompression of *JPEG 2000* and
*JPEG 2000 Lossless* images. You can install *highdicom* with the following
command to enable this:

``pip install 'highdicom[libjpeg]'``
