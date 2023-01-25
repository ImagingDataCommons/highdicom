Parsing Measurement Reports
===========================

In addition to the ability to create TID 1500 Structured Reports, *highdicom*
also includes functionality to help you find and extract information from
existing SR documents in this format.

First you must get the SR dataset into the format of a highdicom class. You
can do this using the ``from_dataset()`` method of the relevant top-level
highdicom SR object.

.. code-block:: python
    from pydicom import dcmread
    import highdicom as hd

    


**TODO** usability improvements.

Searching For Measurement Groups
--------------------------------

Accessing Data in Measurement Groups
------------------------------------

Searching for Measurements
--------------------------

Accessing Data in Measurements
------------------------------

Searching for Evaluations
-------------------------
