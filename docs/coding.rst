.. _coding:

Coding
======

"Coding" is a key concept used throughout `highdicom`. By "coding", we are
referring to the use of standardized nomenclatures or terminologies to describe
medical (or related) concepts. Use of coding is vital to ensure that these
concepts are unambiguously encoded within DICOM files. Coding is especially
fundamental within structured reporting, but is also found in other places
around highdicom.

To communicate a concept in DICOM using a coding scheme, three elements are
necessary:

- A **coding scheme**: the pre-defined terminology used to define the concept.
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

- The DCM terminology. This terminology is defined within the DICOM standard
  itself and is used to refer to DICOM concepts, as well as other concepts
  within the radiology workflow.
- SNOMED-CT. This terminology contains codes to describe medical concepts
  including anatomy, diseases and procedures.
- RadLex. A standardized terminology for concepts in radiology.
- UCUM. A terminology specifically to describe units of measurement.

Highdicom defines the :class:`highdicom.sr.CodedConcept` to encapsulate
a coded concept. To create a coded, you pass values for the coding scheme,
code value, and code meaning. For example, to describe a tumor using the
SNOMED-CT terminology, you could do this:

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
the `highdicom` class is a sub-class of `pydicom.Dataset` with the relevant
attributes such that it can be included directly into a DICOM object. `pydicom`
also includes within it values for a large number of coded concepts within
the DCM, SNOMED-CT, and UCUM terminologies. For example, instead of manually
created the "tumor" concept above, we could have just used the pre-defined
value in `pydicom`:

.. code-block:: python

   from pydicom.sr.codedict import codes

   tumor_code = codes.SCT.Tumor
   print(tumor_code.value)
   # '1083690006'
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


The two classes are used interoperably throughout highdicom: anywhere in the
`highdicom` API that you can pass a `:class:`highdicom.sr.CodedConcept`, you
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

Finding Suitable Codes
----------------------

The `pydicom` code dictionary allows searching for concepts via simple string
matching. However, generally it will be necessary to search the documentation
for the coding scheme itself.

.. code-block:: python

   from pydicom.sr.codedict import codes

   print(codes.SCT.dir('liver'))
   # ['DeliveredRadiationDose',
   # 'HistoryOfPrematureDelivery',
   # 'Liver',
   # 'LiverStructure']
