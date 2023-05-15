.. _ann:

Microscopy Bulk Simple Annotations
==================================

The Microscopy Bulk Simple Annotation IOD is an IOD designed specifically to
store large numbers of similar annotations and measurements from microscopy
images. Annotations of microscopy images typically refer to very large numbers
of cells or cellular structures. Storing these in a Structured Report Document,
with its highly nested structure, would be very inefficient in storage space
and unnecessarily complex and slow to parse. Microscopy Bulk Simple Annotations
("bulk annotations") solve this problem by allowing you to store large number
of similar annotations or measurements in efficient arrays without duplication
of the descriptive metadata.

Each bulk annotation object contains one or more Annotation Groups, each of
which contains a set of graphical annotations, and optionally one or more
numerical measurements relating to those graphical annotations.

Annotation Groups
-----------------

An Annotation Group is a set of multiple similar annotations from a microscopy
image. For example, a single annotation group may contain all annotations of
cell nuclei, lymphocytes, or regions of necrosis in the image. In *highdicom*,
an annotation group is represented by a :class:`highdicom.ann.AnnotationGroup`.

Each annotation group contains some required metadata that describes the contents
of the group, as well as some further optional metadata that may contain further
details about the group or the derivation of the annotations it contains. The
required metadata elements include:

* A `number` (`int`), an integer number for the group.
* A `label` (`str`) giving a human-readable label for the group.
* A `uid` (`str` or :class:`highdicom.UID`) uniquely identifying the group.
* An `annotated_property_category` and `annotated_property_type`
  (:class:`highdicom.sr.CodedConcept`)
