.. _coding:

Coding
======

"Coding" is a key concept used throughout `highdicom`. By "coding", we are
referring to the use of standardized nomenclatures or terminologies to describe
medical (or related) concepts. For example, instead of using the English word
"liver" (or a word in another human language) to describe the liver, we instead
use a code such as '10200004' from the SNOMED-CT nomenclature to describe the
liver in standardized way. Use of coding is vital to ensure that these concepts
are expressed unambiguously within DICOM files. Coding is especially
fundamental within structured reporting, but is also found in other places
around the DICOM standard and, in turn, `highdicom`.

To communicate a concept in DICOM using a coding scheme, three elements are
necessary:

- A **coding scheme**: an identifier of the pre-defined terminology used to
  define the concept.
- A code **value**: the code value conveys a unique identifier for the specific
  concept. It is often a number or alphanumeric string that may not have any
  inherent meaning outside of the terminology.
- A code **meaning**. The code meaning conveys the concept in a way that is
  understandable to humans.

Any coding scheme that operates in this way may be used within DICOM objects,
including ones that you create yourself. However, it is highly recommended to
use a well-known and widely accepted standard terminology to ensure that your
DICOM objects will be as widely understood and as interoperable as possible.
Examples of widely used medical terminologies include:

- The `DCM <https://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_D.html>`_
  terminology. This terminology is defined within the DICOM standard itself and
  is used to refer to DICOM concepts, as well as other concepts
  within the radiology workflow.
- `SNOMED-CT <https://www.snomed.org/>`_. This terminology contains codes to
  describe medical concepts including anatomy, diseases, and procedures.
- `RadLex <https://radlex.org/>`_. A standardized terminology for concepts
  in radiology.
- `UCUM <https://ucum.org/>`_. A terminology specifically to describe units of
  measurement.

See
`this page <https://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_8.html>`_
for a list of terminologies used within DICOM.

`Highdicom` defines the :class:`highdicom.sr.CodedConcept` class to encapsulate
a coded concept. To create a :class:`highdicom.sr.CodedConcept`, you pass
values for the coding scheme, code value, and code meaning. For example, to
describe a tumor using the SNOMED-CT terminology, you could do this:

.. code-block:: python

   import highdicom as hd

   tumor_code = hd.sr.CodedConcept(
       value="108369006",
       scheme_designator="SCT",
       meaning="Tumor"
   )

Codes within Pydicom
--------------------

The `pydicom` library, upon which `highdicom` is built, has its own class
``pydicom.sr.coding.Code`` that captures coded concepts in the same way that
:class:`highdicom.sr.CodedConcept` does. The reason for the difference is that
the `highdicom` class is a sub-class of ``pydicom.Dataset`` with the relevant
attributes such that it can be included directly into a DICOM object. `pydicom`
also includes within it values for a large number of coded concepts within
the DCM, SNOMED-CT, and UCUM terminologies. For example, instead of manually
creating the "tumor" concept above, we could have just used the pre-defined
value in `pydicom`:

.. code-block:: python

   from pydicom.sr.codedict import codes

   tumor_code = codes.SCT.Tumor
   print(tumor_code.value)
   # '108369006'
   print(tumor_code.scheme_designator)
   # 'SCT'
   print(tumor_code.meaning)
   # 'tumor'

Here are some other examples of codes within `pydicom`:

.. code-block:: python

   from pydicom.sr.codedict import codes

   # A patient, as described by the DCM terminology
   patient_code = codes.DCM.Patient
   print(patient_code)
   # Code(value='121025', scheme_designator='DCM', meaning='Patient', scheme_version=None)

   # A centimeter, a described by the UCUM coding scheme
   cm_code = codes.UCUM.cm
   print(cm_code)
   # Code(value='cm', scheme_designator='UCUM', meaning='cm', scheme_version=None)


The two classes can be used interoperably throughout highdicom: anywhere in the
`highdicom` API that you can pass a :class:`highdicom.sr.CodedConcept`, you
can pass an ``pydicom.sr.coding.Code`` instead and it will be converted behind
the scenes for you. Furthermore, equality is defined between the two classes
such that it evaluates to true if they represent the same concept, and they
hash to the same value if you use them within sets or as keys in dictionaries.

.. code-block:: python

   import highdicom as hd
   from pydicom.sr.codedict import codes

   tumor_code_hd = hd.sr.CodedConcept(
       value="108369006",
       scheme_designator="SCT",
       meaning="Tumor"
   )
   tumor_code = codes.SCT.Tumor

   assert tumor_code_hd == tumor_code
   assert len({tumor_code_hd, tumor_code}) == 1

For equality and hashing, two codes are considered equivalent if they have the
same coding scheme, and value, regardless of how their meaning is represented.

Context Groups
--------------

Many places where codes are used in the standard are associated with a
`Context Group`. These are sets of codes determined to be suitable for a
particular purpose. Most are `extensible`, meaning you can choose codes 
outside the context group if you wish. However, it is best practice to use
a code within the context group if a suitable code exists within it.

Context groups are have an identifier called a `CID`. For example, :dcm:`CID
7150 <part16/sect_CID_7150.html>` is the context group containing codes to use
to describe segmented properties categories in a DICOM segmentation and
:dcm:`CID 7151 <part16/sect_CID_7151.html>` contains values for the more
specific segmented property types. Note that this second example includes other
CIDs recursively.

Finding Suitable Codes
----------------------

The `pydicom` code dictionary allows searching for concepts via simple string
matching. You can also search with context groups.

.. code-block:: python

   from pydicom.sr.codedict import codes

   print(codes.SCT.dir('liver'))
   # ['DeliveredRadiationDose',
   # 'HistoryOfPrematureDelivery',
   # 'Liver',
   # 'LiverStructure']

   print(codes.SCT.Liver)
   # Code(value='10200004', scheme_designator='SCT', meaning='Liver', scheme_version=None)

   # Search within a context group (e.g. CID7150)
   print(codes.cid7150.dir())
   # ['AnatomicalStructure',
   #  'BodySubstance',
   #  'Function',
   #  'MorphologicallyAbnormalStructure',
   #  'PhysicalObject',
   #  'SpatialAndRelationalConcept',
   #  'Substance',
   #  'Tissue']

   # Check whether a code lies within a context group
   print(codes.SCT.BodySubstance in codes.cid7150)
   # True

However, for more advanced searching it is generally advisable to search the
documentation for the coding scheme itself. This 
`webpage <https://fedorov.github.io/dcmterms/terminology.html>`_ is a nice
interactive tool for searching the terms and Context Groups defined within
DICOM.

UCUM
----

UCUM works a little differently to the other coding schemes discussed here.
Rather than a finite set of codes, it defines codes for a set of units and a
syntax for combining them to give derived units (e.g. `"mm/s"` for millimeters
per second), meaning there are potentially infinite codes. Pydicom defines
a list of the most common units, but this will not cover every use case.

We recommend this `page
<https://ucum.nlm.nih.gov/ucum-lhc/demo.html>`_, a useful validator for
checking UCUM syntax.

.. _coding-modifiers:

Codes With Modifiers
--------------------

Some attributes in DICOM allow a code to be accompanied by one or more
`modifier` codes that refine the meaning of the root code. A typical example is
laterality: rather than needing a separate code for "left kidney" and "right
kidney", you can use the single code for "kidney" together with the modifier
"left" or "right". The codes used as modifiers are generally drawn from
:dcm:`CID 2 <part16/sect_CID_2.html>` "Anatomic Modifier".

In the DICOM standard, each place where modifiers are allowed has its own
sequence attribute for holding them (for example "Segmented Property Type
Modifier Code Sequence" next to "Segmented Property Type Code Sequence").
`Highdicom` provides a class for each such attribute. They are all
sub-classes of :class:`highdicom.sr.CodedConcept`, and are constructed from a
root code and an optional sequence of modifier codes (either
``pydicom.sr.coding.Code`` or :class:`highdicom.sr.CodedConcept` objects):

.. list-table::
   :header-rows: 1

   * - Class
     - Describes
     - Used in
   * - :class:`highdicom.seg.SegmentedPropertyType`
     - The property that a segment represents
       (:dcm:`CID 7151 <part16/sect_CID_7151.html>`)
     - :class:`highdicom.seg.SegmentDescription` (see :ref:`seg-modifiers`)
   * - :class:`highdicom.ann.AnnotatedPropertyType`
     - The property that an annotation group represents
       (:dcm:`CID 8135 <part16/sect_CID_8135.html>`)
     - :class:`highdicom.ann.AnnotationGroup` (see :ref:`ann-modifiers`)
   * - :class:`highdicom.AnatomicRegion`
     - The anatomic region into which an object falls
       (e.g. :dcm:`CID 4 <part16/sect_CID_4.html>`)
     - :class:`highdicom.seg.SegmentDescription` and
       :class:`highdicom.ann.AnnotationGroup`
   * - :class:`highdicom.PrimaryAnatomicStructure`
     - The anatomic structure that an object represents
     - :class:`highdicom.seg.SegmentDescription` and
       :class:`highdicom.ann.AnnotationGroup`

Each class stores its modifiers in the attribute with the corresponding
standard name, so that the object can be placed directly into a DICOM dataset.
The classes are not interchangeable: for example an
:class:`highdicom.AnatomicRegion` cannot be used where a
:class:`highdicom.seg.SegmentedPropertyType` is expected, and doing so raises a
``TypeError``.

To create an instance of one of these classes, pass a
``pydicom.sr.coding.Code`` or :class:`highdicom.sr.CodedConcept` to serve as
the root concept, then optionally a list of further codes to the `modifiers`
parameter.

.. code-block:: python

   import highdicom as hd
   from pydicom.sr.codedict import codes

   # A segmented property type of "Kidney", modified by "Left"
   left_kidney = hd.seg.SegmentedPropertyType(
       codes.SCT.Kidney,
       modifiers=[codes.SCT.Left],
   )

   # The value, meaning, and scheme_designator properties behave as they do
   # for a plain CodedConcept
   print(left_kidney.scheme_designator, left_kidney.value, left_kidney.meaning)
   # SCT 64033007 Kidney

   # Access the modifiers using the modifiers property (this is the same for
   # every one of the classes above)
   print([m.meaning for m in left_kidney.modifiers])
   # ['Left']

   # Access the root code on its own, without the modifiers
   assert left_kidney.unmodified == codes.SCT.Kidney

Two modified codes are equal if they are instances of the same class, have the
same root code, and have the same set of modifiers. The order of the modifiers
is not significant, and the hash is consistent with this definition, so these
objects may be used in sets and as dictionary keys.

.. code-block:: python

   a = hd.seg.SegmentedPropertyType(
       codes.SCT.Kidney,
       modifiers=[codes.SCT.Left, codes.SCT.Posterior],
   )
   b = hd.seg.SegmentedPropertyType(
       codes.SCT.Kidney,
       modifiers=[codes.SCT.Posterior, codes.SCT.Left],
   )
   assert a == b
   assert len({a, b}) == 1

   # Different modifiers means a different code
   assert a != hd.seg.SegmentedPropertyType(codes.SCT.Kidney)

A modified code that has no modifiers is considered equal to the plain code,
but one that has modifiers does not. When you want to compare only the root
code regardless of modifiers, use the ``unmodified`` property:

.. code-block:: python

   assert hd.seg.SegmentedPropertyType(codes.SCT.Kidney) == codes.SCT.Kidney
   assert a != codes.SCT.Kidney
   assert a.unmodified == codes.SCT.Kidney

UCUM
----

UCUM works a little differently to the other coding schemes discussed here.
Rather than a finite set of codes, it defines codes for a set of units and a
syntax for combining them to give derived units (e.g. `"mm/s"` for millimeters
per second), meaning there are potentially infinite codes. Pydicom defines
a list of the most common units, but this will not cover every use case.

We recommend this `page
<https://ucum.nlm.nih.gov/ucum-lhc/demo.html>`_, a useful validator for
checking UCUM syntax.

Codes With Modifiers
--------------------

Some attributes in DICOM allow a code to be accompanied by one or more
`modifier` codes that refine the meaning of the root code. A typical example is
laterality: rather than needing a separate code for "left kidney" and "right
kidney", you can use the single code for "kidney" together with the modifier
"left" or "right". The codes used as modifiers are generally drawn from
:dcm:`CID 2 <part16/sect_CID_2.html>` "Anatomic Modifier".

In the DICOM standard, each place where modifiers are allowed has its own
sequence attribute for holding them (for example "Segmented Property Type
Modifier Code Sequence" next to "Segmented Property Type Code Sequence").
`Highdicom` provides a class for each such attribute. They are all
sub-classes of :class:`highdicom.sr.CodedConcept`, and are constructed from a
root code and an optional sequence of modifier codes (either
``pydicom.sr.coding.Code`` or :class:`highdicom.sr.CodedConcept` objects):

.. list-table::
   :header-rows: 1

   * - Class
     - Describes
     - Used in
   * - :class:`highdicom.seg.SegmentedPropertyType`
     - The property that a segment represents
       (:dcm:`CID 7151 <part16/sect_CID_7151.html>`)
     - ``segmented_property_type`` of :class:`highdicom.seg.SegmentDescription`
   * - :class:`highdicom.ann.AnnotatedPropertyType`
     - The property that an annotation group represents
       (:dcm:`CID 8135 <part16/sect_CID_8135.html>`)
     - ``annotated_property_type`` of :class:`highdicom.ann.AnnotationGroup`
   * - :class:`highdicom.AnatomicRegion`
     - The anatomic region into which a segment or annotation group falls
       (e.g. :dcm:`CID 4 <part16/sect_CID_4.html>`)
     - ``anatomic_regions`` of :class:`highdicom.seg.SegmentDescription` and
       :class:`highdicom.ann.AnnotationGroup`
   * - :class:`highdicom.PrimaryAnatomicStructure`
     - The anatomic structure that a segment or annotation group represents
     - ``primary_anatomic_structures`` of
       :class:`highdicom.seg.SegmentDescription` and
       :class:`highdicom.ann.AnnotationGroup`

Each class stores its modifiers in the attribute with the corresponding
standard name, so that the object can be placed directly into a DICOM dataset.
The classes are not interchangeable: for example an
:class:`highdicom.AnatomicRegion` cannot be passed where a
:class:`highdicom.seg.SegmentedPropertyType` is expected, and doing so raises a
``TypeError``.

Creating Modified Codes
^^^^^^^^^^^^^^^^^^^^^^^

To create an instance of the relevant class, pass a ``pydicom.sr.coding.Code``
or :class:`highdicom.sr.CodedConcept` to serve as the root concept, then
optionally a list of further codes to the `modifiers` parameter.

.. code-block:: python

   import highdicom as hd
   from pydicom.sr.codedict import codes

   # A segmented property type of "Kidney", modified by "Left"
   left_kidney = hd.seg.SegmentedPropertyType(
       codes.SCT.Kidney,
       modifiers=[codes.SCT.Left],
   )

   # The value, meaning, and scheme_designator properties behave as they do for
   a plain CodedConcept
   print(left_kidney.scheme_designator, left_kidney.value, left_kidney.meaning)
   # SCT 64033007 Kidney

   # Access the modifiers using the modifiers property (regradless of the
   specific class used)
   print([m.meaning for m in left_kidney.modifiers])
   # ['Left']

   # Acces the root code on its own, without the modifiers
   assert left_kidney.unmodified == codes.SCT.Kidney

You can then pass these objects to the constructors that accept them. Anywhere
that a modified class is accepted, you may still pass a plain
``pydicom.sr.coding.Code`` or :class:`highdicom.sr.CodedConcept`, in which case
it is treated as a code with no modifiers.

.. code-block:: python

   segment = hd.seg.SegmentDescription(
       segment_number=1,
       segment_label='Kidney',
       segmented_property_category=codes.SCT.AnatomicalStructure,
       segmented_property_type=left_kidney,
       algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
       anatomic_regions=[
           hd.AnatomicRegion(
               codes.SCT.AbdomenAndPelvis,
               modifiers=[codes.SCT.Left],
           ),
       ],
       primary_anatomic_structures=[
           hd.PrimaryAnatomicStructure(
               codes.SCT.Kidney,
               modifiers=[codes.SCT.Left],
           ),
       ],
   )

When reading existing objects, the corresponding properties
(:attr:`highdicom.seg.SegmentDescription.segmented_property_type`,
:attr:`highdicom.ann.AnnotationGroup.annotated_property_type`, and the
``anatomic_regions`` and ``primary_anatomic_structures`` properties of both
classes) always return the modified classes, whether or not any modifiers are
present. Use the ``modifiers`` property to get the modifiers as a tuple of
:class:`highdicom.sr.CodedConcept` (empty if there are none) and the
``unmodified`` property to get the root code alone.

.. code-block:: python

   segment_type = segment.segmented_property_type
   assert isinstance(segment_type, hd.seg.SegmentedPropertyType)
   for modifier in segment_type.modifiers:
       print(modifier.meaning)
   # Left

Equality and Hashing
^^^^^^^^^^^^^^^^^^^^

Two modified codes are equal if they are instances of the same class, have the
same root code, and have the same set of modifiers. The order of the modifiers
is not significant, and the hash is consistent with this definition, so these
objects may be used in sets and as dictionary keys.

.. code-block:: python

   a = hd.seg.SegmentedPropertyType(
       codes.SCT.Kidney,
       modifiers=[codes.SCT.Left, codes.SCT.Posterior],
   )
   b = hd.seg.SegmentedPropertyType(
       codes.SCT.Kidney,
       modifiers=[codes.SCT.Posterior, codes.SCT.Left],
   )
   assert a == b
   assert len({a, b}) == 1

   # Different modifiers means a different code
   assert a != hd.seg.SegmentedPropertyType(codes.SCT.Kidney)

   # ...even though the root codes are the same
   assert a.unmodified == hd.seg.SegmentedPropertyType(codes.SCT.Kidney)

A modified code that has no modifiers is considered equal to the plain code,
but one that has modifiers does not. When you want to compare only the root
code regardless of modifiers, use the ``unmodified`` property:

.. code-block:: python

   assert hd.seg.SegmentedPropertyType(codes.SCT.Kidney) == codes.SCT.Kidney
   assert a != codes.SCT.Kidney
   assert a.unmodified == codes.SCT.Kidney
