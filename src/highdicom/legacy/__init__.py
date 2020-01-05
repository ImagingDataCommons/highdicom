"""Package for CT, MR and PET image modalities that allow conversion of series
of legacy single-frame image instances into enhanced multi-frame image
instances.
"""

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.4.4',    # Legacy Converted Enhanced MR Image
    '1.2.840.10008.5.1.4.1.1.2.2',    # Legacy Converted Enhanced CT Image
    '1.2.840.10008.5.1.4.1.1.128.1',  # Legacy Converted Enhanced PET Image
}
