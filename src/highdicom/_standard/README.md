# Standard Data

This directory contains tables derived from the DICOM standard, in JSON format.

The following files were generated using the `bin/create_iods_modules.py` in
the highdicom repo:

- `sop_class_iod_map.json`
- `iod_module_map.json`
- `module_attribute_map.json`

These were last generated using version 2026b of the standard.

The `anatomic_regions.json` file was manually created by copying the table
[here](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_L.html)
into a spreadsheet and programmatically removing rows without all required
columns before converting to CSV.

This was last updated using version 2026a of the standard.
