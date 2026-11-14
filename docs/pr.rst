.. _pr:

Presentation States
===================

Presentation states are objects that describe how a separate image or image
series images should be displayed by a viewer application. This can include:

- :doc:`pixel_transforms` that override any that are specified within the
  images themselves
- Graphic annotations (such as lines, circles, polygons) that are displayed
  over the images
- Text annotations that are displayed over the images
- Crops/flips/rotations to apply to the image (not currently supported by
  *highdicom*)

Through their ability to overlay graphics on top of image, presentation states
therefore offer a simple way to represent analysis results (e.g. detected areas
of interest or segmentation contours) over an image. However, this is not
generally recommended because other IODs such as :doc:`seg` and :doc:`sr` offer
better ways to do this that are specifically intended for the purpose of
communicating analysis results, and therefore contain more complete,
standardized metadata describing the meaning and source of the analyses.
However, displaying presentation state objects is more widely supported by
DICOM viewers than Segmentations and Structured Reports, which is why they are
sometimes used for this purpose.

Presentation State IODs
-----------------------

There are four presentation state IODs supported by *highdicom* (though others
are defined in the standard). They have many similarities, but differ in the
types of images to which they apply and therefore the types of pixel transforms
they support. They are:

- *Grayscale Softcopy Presentation State*: These are applied to grayscale
  images to render them as altered grayscale images. The transformations of the
  pixel values are limited to grayscale, but this does *not* mean that graphic
  and text annotations are limited to grayscale, they can be rendered in color.
  They are implemented in the class
  :class:`highdicom.pr.GrayscaleSoftcopyPresentationState`
- *Color Softcopy Presentation State*: These are applied to color images to
  render them as altered color images. They are implemented in the class
  :class:`highdicom.pr.ColorSoftcopyPresentationState`
- *Pseudo-Color Softcopy Presentation State*: These are applied to graysale
  images to transform them into color images for rendering. They are
  implemented in the class
  :class:`highdicom.pr.PseudoColorSoftcopyPresentationState`
- *Advanced Blending Presentation State*: These are used to blend multiple
  images together to form a composite image when rendered. They are implemented
  in the class :class:`highdicom.pr.AdvancedBlendingPresenetationState`.

Pixel Transforms
----------------

Pixel transforms are operations that transform the raw pixel values stored in
the image to a potentially different set of values that a viewer should
display. Pixel transforms are also found in image objects themselves as well as
presentation states and therefore have a dedicated documentation page here:
:doc:`pixel_transforms`. When a presentation state is used by a viewer, the
pixel transforms defined in the PR object override those stored in the original
image file.

The different PR IODs support different combinations of pixel transforms,
leading to different sets of parameters for the constructor of the highdicom
class. as follows (in the order they are applied)

- *Grayscale Softcopy Presentation State*:
   - A ``modality_lut_transformation`` of type
     :class:`highdicom.ModalityLUTTransformation` that define a mapping from
     the raw pixel values stored in the image to the range of values acquired
     by the modality. It is quite unusual to need to override the original
     image's modality LUT.
   - One or more ``voi_lut_transformations`` as a sequence of
     :class:`highdicom.SoftcopyVOILUTTransformation`. These define a particular
     intensity range of "values of interest" to which the image should be
     limited for display. Particularly for modalities with high dynamic range
     (e.g., CTs and X-Rays), these allow you to render images that best display
     detail within a particular tissue type (e.g. soft tissue detail exists in
     a different intensity range to bone detail).
   - A ``presentation_lut_transformation`` of type
     :class:`highdicom.PresentationLUTTransformation`. This defines a final
     transformation that maps the output of previous stages to what the viewer
     renders. Most commonly this is used to invert the range of values for
     images stored with Photometric Interpretation of ``"MONOCHROME1"`` so that
     they display as expected. It is unusual to need to override an image's
     presentation LUT.
- *Color Softcopy Presentation State*:
   - An ``icc_profile`` of type ``bytes`` that performs color corrections to
     RGB values to account for differences in acquisition and rendering
     equipment.
- *Pseudo-Color Softcopy Presentation State*:
   - A ``modality_lut_transformation`` of type
     :class:`highdicom.ModalityLUTTransformation`, as in the Grayscale case.
   - One or more ``voi_lut_transformations`` as a sequence of
     :class:`highdicom.SoftcopyVOILUTTransformation` as in the Grayscale case.
   - A ``palette_color_lut_transformation`` of type
     :class:`highdicom.PaletteColorLUTTransformation` (required). This defines
     a lookup table (LUT) that maps grayscale values to 3-channel RGB values
     allowing you to apply color maps to underlying grayscale images for
     rendering.
   - An ``icc_profile`` of type ``bytes``, as in the Color case.
- *Advanced Blending Presentation State*:
   - An ``icc_profile`` of type ``bytes``, as in the Color case.

In highdicom's constructors, all pixel transform parameters are optional unless
noted otherwise above. If they are not specified, but a corresponding transform
exists in the referenced image, the referenced image transform will be copied
to the presentation state such that the image is displayed the same way with
the presentation state as without it.

Advanced Blending Presentation States
-------------------------------------

Graphic Objects, Layers and Annotations
---------------------------------------

A presentation state may contain any number of Graphic Objects and Text
Objects. These are grouped into Graphic Annotations, each of which relates to a
single Graphic Layer.

Graphic Objects
~~~~~~~~~~~~~~~

A Graphic Object is represented by the :class:`highdicom.pr.GraphicObject` class.
The parameters for construction of a Graphic Object are:

- ``graphic_type`` (``str`` or :class:`highdicom.pr.GraphicTypeValues`). The
  type of graphic (such as point, circle, or polygon). This also defines the
  number of coordinates required in ``graphic_data``. See the documentation of
  :class:`highdicom.pr.GraphicTypeValues` for details. Note that the graphic
  types used in presentation states are similar to, but slightly different
  from, those used in Structured Reports.
- ``graphic_data`` (``numpy.ndarray``). An array of coordinates defining the
  graphic object. The shape of the array should be (N, 2), where N is the
  number of 2D points in this graphic object, which depends on the
  graphic type. Each row of the array therefore describes a (column, row)
  value for a single 2D point, and the interpretation of the points depends
  upon the graphic type.
- ``units`` (``str`` or :class:`highdicom.pr.AnnotationUnitsValues` The units
  in which each point in graphic data is expressed. See documentation of
  :class:`highdicom.pr.AnnotationUnitsValues` for details.

Additional optional parameters include:

- ``is_filled`` (``bool``): Whether the graphic object should be rendered as a
  solid shape (``True``), or just an outline (``False``). Using ``True`` is
  only valid when the graphic type is ``'CIRCLE'`` or ``'ELLIPSE'``, or the
  graphic type is ``'INTERPOLATED'`` or ``'POLYLINE'`` and the first and last
  points are equal giving a closed shape.
- ``tracking_id`` (``str``): User defined text identifier for tracking this
  finding or feature. Shall be unique within the domain in which it is used.
- ``tracking_uid`` (``str``): Unique identifier for tracking this finding or
  feature.

Some examples of constructing Graphic Objects are given below:

.. code-block:: python

    import highdicom as hd
    import numpy as np

    # Graphic Object containing a point
    polyline = hd.pr.GraphicObject(
        graphic_type=hd.pr.GraphicTypeValues.POINT,
        graphic_data=np.array([[15.0, 15.0]]),  # coordinates of point
        units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
    )

    # Graphic Object containing a polyline
    polyline = hd.pr.GraphicObject(
        graphic_type=hd.pr.GraphicTypeValues.POLYLINE,
        graphic_data=np.array([
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0]]
        ),  # coordinates of polyline vertices
        units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
        tracking_id='Finding2',  # site-specific ID
        tracking_uid=hd.UID()  # highdicom will generate a unique ID
    )

Text Objects
~~~~~~~~~~~~

Graphic Annotations
~~~~~~~~~~~~~~~~~~~

Graphic Layers
~~~~~~~~~~~~~~

Constructing Presentation States
--------------------------------

To create a Presentation State object, you pass the standard :ref:`basic`, a
list of all referenced images, and any pixel transforms, graphic annotations,
or graphic layers that you want to include.

A full example of creating a GSPS is given below:

.. code-block:: python

    import highdicom as hd

    import numpy as np
    from pydicom.valuerep import PersonName

    # Read in an example CT image
    image_dataset = hd.imread('path/to/image.dcm')

    # Create an annotation containing a polyline
    polyline = hd.pr.GraphicObject(
        graphic_type=hd.pr.GraphicTypeValues.POLYLINE,
        graphic_data=np.array([
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0]]
        ),  # coordinates of polyline vertices
        units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
        tracking_id='Finding1',  # site-specific ID
        tracking_uid=hd.UID()  # highdicom will generate a unique ID
    )

    # Create a text object annotation
    text = hd.pr.TextObject(
        text_value='Important Finding!',
        bounding_box=np.array(
            [30.0, 30.0, 40.0, 40.0]  # left, top, right, bottom
        ),
        units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for bounding box
        tracking_id='Finding1Text',  # site-specific ID
        tracking_uid=hd.UID()  # highdicom will generate a unique ID
    )

    # Create a single layer that will contain both graphics
    # There may be multiple layers, and each GraphicAnnotation object
    # belongs to a single layer
    layer = hd.pr.GraphicLayer(
        layer_name='LAYER1',
        order=1,  # order in which layers are displayed (lower first)
        description='Simple Annotation Layer',
    )

    # A GraphicAnnotation may contain multiple text and/or graphic objects
    # and is rendered over all referenced images
    annotation = hd.pr.GraphicAnnotation(
        referenced_images=[image_dataset],
        graphic_layer=layer,
        graphic_objects=[polyline],
        text_objects=[text]
    )

    # Assemble the components into a GSPS object
    gsps = hd.pr.GrayscaleSoftcopyPresentationState(
        referenced_images=[image_dataset],
        series_instance_uid=hd.UID(),
        series_number=123,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Manufacturer',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ',
        content_label='ANNOTATIONS',
        graphic_layers=[layer],
        graphic_annotations=[annotation],
        institution_name='MGH',
        institutional_department_name='Radiology',
        content_creator_name=PersonName.from_named_components(
            family_name='Doe',
            given_name='John'
        ),
        series_description='Example Presentation State',
    )

    # Save the GSPS file
    gsps.save_as('gsps.dcm')

