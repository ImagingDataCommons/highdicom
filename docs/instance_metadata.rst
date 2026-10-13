.. _basic:

Basic Instance Metadata
=======================

This page covers some of the basic metadata that is present in all objects that
can be created with *highdicom*.

The following parameters are required for construction of all DICOM objects:

- ``study_instance_uid``, ``series_instance_uid``, and ``sop_instance_uid``
  (``str`` or :class:`highdicom.uid.UID`) of the instance being created. See
  :ref:`uids` for guidance on what to pass here.
- The (1-based) ``series_number`` (``int``) of this series within the study.
  There are no strict requirements on this other than it should be a positive
  integeer. You should try to ensure that this is not taken by another series
  in the study, though in general this can be difficult to do unless you have
  access to the entire study. As a result, it is common to see non-consecutive
  (and occasionally duplicated) series numbers. A common approach is to pick a
  high "unusual" number and assume that it is not taken by another series.
- The (1-based) ``instance_number`` (``int``) of this instance within the
  study. Again there are few requirements here, but it is generally
  straightforward to set this as a sequence of consecutive positive integers
  starting at ``1``. Many objects created by *highdicom* will be the only
  object in their series. In this case, just pass ``1``.
- The ``manufacturer`` (``str``). This is a string giving the manufacturer of
  the device that creates the instance. Generally this will be the name of your
  organization.

The following parameters are usually optional, but may be require depending on
the IOD you are creating:

- ``series_description`` (``str``). A concise, human-readable desciption of the
  attribute. Though this is technically optional, it is strongly recommended to
  include this in all objects. It is very commonly used by viewers to list
  series present, and generally used by humans to understand the content of a
  series. Therefore including it will make the object more convenient to work
  with in viewers
- ``manufacturer_model_name`` (``str``). The manufacturer's model name of the
  device that created this instance.
- ``software_versions`` (``str``). Version of the software that created this
  instance. This should be a version number of your software (not that the
  version of *highdicom* used is automatically captured in the
  *ContributingEquipmentSequence*).
- ``device_serial_number`` (``str``). Serial number of the individual device
  that created the instance.
- ``institution_name`` (``str``). Name of the instution where the instance was
  created. This will usually be the hospital or healthcare provider using the
  softaare.
- ``institutional_department_name`` (``str``). Name of the department within
  the instution where the instance was created.

.. _patient_metadata:

Patient and Clinical Metadata
-----------------------------

A number of aatributes of patient and clinical metadata are present in every
object that *highdicom* creates. Usually, however, you do not have to specify
these explicitly because they are copied from a source image that you provide.
Currently the only exception to this is the :class:`highdicom.sc.SCImage` class
when the standard constructor is used (rather than the alternative
:meth:`highdicom.sc.SCImage.from_ref_dataset` constructor). In this case, the
following attributes may be specified. All are optional, but should be supplied
if known:

- ``patient_id``; (``str``): ID of the patient (medical record number)
- ``patient_name:`` (``str`` or ``pydicom.valuerep.PersonName``): Name of the patient
- ``patient_birth_date:`` (``str`` or ``datetime.date``): Patient's birth date
- ``patient_sex:`` (``str`` or :class:`highdicom.enum.PatientSex`): Patient's sex
- ``study_id:`` (``str``): ID of the study
- ``accession_number:`` (``str``): Accession number of the study
- ``study_date:`` (``str`` or ``datetime.date``): Date of study creation
- ``study_time:`` (``str`` or ``datetime.time``): Time of study creation
