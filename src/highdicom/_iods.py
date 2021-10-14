"""DICOM Information Object Definitions (IODs)
auto-generated on 2021-09-26 at 20:07:37.
"""
from typing import Dict, List

IOD_MODULE_MAP: Dict[str, List[Dict[str, str]]] = {
    "12-lead-ecg": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "acquisition-context-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "advanced-blending-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "advanced-blending-presentation-state",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "advanced-blending-presentation-state-display",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "displayed-area",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "spatial-transformation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "icc-profile",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "ambulatory-ecg": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "arterial-pulse-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "autorefraction-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "autorefraction-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "autorefraction-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "basic-cardiac-electrophysiology-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "basic-directory": [
        {
            "ie": "Basic Directory",
            "key": "file-set-identification",
            "usage": "M"
        },
        {
            "ie": "Basic Directory",
            "key": "directory-information",
            "usage": "U"
        }
    ],
    "basic-structured-display": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "structured-display",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "structured-display-image-box",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "structured-display-annotation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "basic-text-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "basic-voice-audio-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "blending-softcopy-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-blending",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "displayed-area",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "spatial-transformation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "palette-color-lookup-table",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "icc-profile",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "body-position-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "breast-projection-x-ray-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "dx-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "enhanced-mammography-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-mammography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "breast-view",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "breast-projection-x-ray-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "patient-orientation",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "breast-tomosynthesis-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-mammography-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "breast-tomosynthesis-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image---equipment-coordinate-relationship",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "breast-tomosynthesis-contributing-sources",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "breast-tomosynthesis-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "breast-view",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "c-arm-photon-electron-radiation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "rt-radiation-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "c-arm-photon-electron-delivery-device",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "c-arm-photon-electron-beam",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "c-arm-photon-electron-radiation-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-radiation-record-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "c-arm-photon-electron-delivery-device",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "c-arm-photon-electron-beam",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "chest-cad-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "colon-cad-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "color-palette": [
        {
            "ie": "Color Palette",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Color Palette",
            "key": "color-palette-definition",
            "usage": "M"
        },
        {
            "ie": "Color Palette",
            "key": "palette-color-lookup-table",
            "usage": "M"
        },
        {
            "ie": "Color Palette",
            "key": "icc-profile",
            "usage": "M"
        }
    ],
    "color-softcopy-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-relationship",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-shutter",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "bitmap-display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-activation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "displayed-area",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "spatial-transformation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "icc-profile",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "comprehensive-3d-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "comprehensive-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "content-assessment-results": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Content Assessment Results",
            "key": "content-assessment-results",
            "usage": "M"
        },
        {
            "ie": "Content Assessment Results",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Content Assessment Results",
            "key": "common-instance-reference",
            "usage": "M"
        }
    ],
    "corneal-topography-map": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "corneal-topography-map-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "palette-color-lookup-table",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "corneal-topography-map-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "corneal-topography-map-analysis",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "cr-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "cr-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "display-shutter",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cr-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "modality-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "ct-defined-procedure-protocol": [
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "protocol-context",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "clinical-trial-context",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-specification",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "equipment-specification",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "instructions",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-positioning",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "general-defined-acquisition",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "general-defined-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "defined-storage",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "ct-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-plane",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "ct-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-energy-ct-image",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "ct-performed-procedure-protocol": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ct-protocol-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "protocol-context",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-protocol-context",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "instructions",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-positioning",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "performed-ct-acquisition",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "performed-ct-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "performed-storage",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "deformable-spatial-registration": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "spatial-registration-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Spatial Registration",
            "key": "deformable-spatial-registration",
            "usage": "M"
        },
        {
            "ie": "Spatial Registration",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Spatial Registration",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Spatial Registration",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "dermoscopic-photography-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "vl-photographic-equipment",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "vl-photographic-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dermoscopic-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "digital-intra-oral-x-ray-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "dx-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "intra-oral-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "display-shutter",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dx-anatomy-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-detector",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-collimator",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dx-positioning",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-tomography-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-acquisition-dose",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-generation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-filtration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-grid",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intra-oral-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "image-histogram",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "digital-mammography-x-ray-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "dx-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "mammography-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "display-shutter",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dx-anatomy-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-detector",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-collimator",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dx-positioning",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-tomography-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-acquisition-dose",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-generation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-filtration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-grid",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "mammography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "image-histogram",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "digital-x-ray-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "dx-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "display-shutter",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dx-anatomy-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-detector",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-collimator",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "dx-positioning",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-tomography-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-acquisition-dose",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-generation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-filtration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-grid",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "image-histogram",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "electromyogram": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "electrooculogram": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "encapsulated-cda": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "encapsulated-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "encapsulated-document",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "encapsulated-mtl": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "encapsulated-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "encapsulated-document",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "manufacturing-3d-model",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Encapsulated Document",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "common-instance-reference",
            "usage": "C"
        }
    ],
    "encapsulated-obj": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "encapsulated-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "encapsulated-document",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "manufacturing-3d-model",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Encapsulated Document",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "common-instance-reference",
            "usage": "C"
        }
    ],
    "encapsulated-pdf": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "encapsulated-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "encapsulated-document",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "encapsulated-stl": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "encapsulated-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "encapsulated-document",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "manufacturing-3d-model",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Encapsulated Document",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Encapsulated Document",
            "key": "common-instance-reference",
            "usage": "C"
        }
    ],
    "enhanced-ct-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ct-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "enhanced-ct-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "supplemental-palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-ct-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-multi-energy-ct-acquisition",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "enhanced-mr-color-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "mr-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "enhanced-mr-color-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "bulk-motion-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-mr-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "mr-pulse-sequence",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "enhanced-mr-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "mr-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "enhanced-mr-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "bulk-motion-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "supplemental-palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-mr-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "mr-pulse-sequence",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "enhanced-pet-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "enhanced-pet-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-pet-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-pet-isotope",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-pet-acquisition",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-pet-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-pet-corrections",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "enhanced-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "enhanced-us-volume": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "enhanced-us-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "ultrasound-frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "enhanced-us-volume-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-palette-color-lookup-table",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-us-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ivus-image",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "excluded-intervals",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "enhanced-x-ray-radiation-dose-structured-report": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "enhanced-xa-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "xa-xrf-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "mask",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-xa-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-filtration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-grid",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-xa-xrf-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "xa-xrf-acquisition",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "x-ray-image-intensifier",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "x-ray-detector",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "xa-xrf-multi-frame-presentation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "enhanced-xrf-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "xa-xrf-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "mask",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-xrf-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-tomography-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-filtration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-grid",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-xa-xrf-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "xa-xrf-acquisition",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "x-ray-image-intensifier",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "x-ray-detector",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "xa-xrf-multi-frame-presentation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "extensible-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "general-audio-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "general-ecg": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "generic-implant-template": [
        {
            "ie": "Implant Template",
            "key": "generic-implant-template-description",
            "usage": "M"
        },
        {
            "ie": "Implant Template",
            "key": "generic-implant-template-2d-drawings",
            "usage": "U"
        },
        {
            "ie": "Implant Template",
            "key": "generic-implant-template-3d-models",
            "usage": "U"
        },
        {
            "ie": "Implant Template",
            "key": "generic-implant-template-mating-features",
            "usage": "U"
        },
        {
            "ie": "Implant Template",
            "key": "generic-implant-template-planning-landmarks",
            "usage": "U"
        },
        {
            "ie": "Implant Template",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Surface Mesh",
            "key": "surface-mesh",
            "usage": "C"
        }
    ],
    "grayscale-softcopy-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-relationship",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-shutter",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-mask",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "mask",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "bitmap-display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-activation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "displayed-area",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "spatial-transformation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "modality-lut",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "softcopy-voi-lut",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "softcopy-presentation-lut",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "hanging-protocol": [
        {
            "ie": "Hanging Protocol",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Hanging Protocol",
            "key": "hanging-protocol-definition",
            "usage": "M"
        },
        {
            "ie": "Hanging Protocol",
            "key": "hanging-protocol-environment",
            "usage": "M"
        },
        {
            "ie": "Hanging Protocol",
            "key": "hanging-protocol-display",
            "usage": "M"
        }
    ],
    "hemodynamic-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "implant-assembly-template": [
        {
            "ie": "Implant Assembly",
            "key": "implant-assembly-template",
            "usage": "M"
        },
        {
            "ie": "Implant Assembly",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "implant-template-group": [
        {
            "ie": "Implant Template Group",
            "key": "implant-template-group",
            "usage": "M"
        },
        {
            "ie": "Implant Template Group",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "implantation-plan-sr-document": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "intraocular-lens-calculations": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "intraocular-lens-calculations-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "intraocular-lens-calculations",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "intravascular-optical-coherence-tomography-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "intravascular-oct-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "supplemental-palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "intravascular-optical-coherence-tomography-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "intravascular-oct-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "intravascular-oct-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "intravascular-oct-processing-parameters",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "intravascular-image-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "keratometry-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "keratometry-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "keratometry-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "key-object-selection-document": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "key-object-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "key-object-document",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "legacy-converted-enhanced-ct-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ct-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "legacy-converted-enhanced-ct-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-ct-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "legacy-converted-enhanced-mr-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "mr-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "legacy-converted-enhanced-mr-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "bulk-motion-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-mr-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "legacy-converted-enhanced-pet-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "enhanced-pet-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "legacy-converted-enhanced-pet-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "enhanced-pet-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "lensometry-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "lensometry-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "lensometry-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "macular-grid-thickness-and-volume-report": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "mammography-cad-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "microscopy-bulk-simple-annotations": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "microscopy-bulk-simple-annotations-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Annotation",
            "key": "microscopy-bulk-simple-annotations",
            "usage": "M"
        },
        {
            "ie": "Annotation",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Annotation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Annotation",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "mr-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-plane",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "mr-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "mr-spectroscopy": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "mr-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "mr-spectroscopy-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "bulk-motion-synchronization",
            "usage": "C"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "mr-spectroscopy",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "mr-spectroscopy-pulse-sequence",
            "usage": "C"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "mr-spectroscopy-data",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "MR Spectroscopy",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "multi-channel-respiratory-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "multi-frame-grayscale-byte-sc-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-grayscale-byte-sc-image-multi-frame-functional-groups",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-image",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-vector",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "multi-frame-grayscale-word-sc-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-grayscale-word-sc-image-multi-frame-functional-groups",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-image",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-vector",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "multi-frame-single-bit-sc-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-image",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-vector",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "multi-frame-true-color-sc-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-true-color-sc-image-multi-frame-functional-groups",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-image",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sc-multi-frame-vector",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "nm-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "nm-pet-patient-orientation",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "nm-image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "nm-multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "nm-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "nm-isotope",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "nm-detector",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "nm-tomo-acquisition",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "nm-multi-gated-acquisition",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "nm-phase",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "nm-reconstruction",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-overlay",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "ophthalmic-axial-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-axial-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "ophthalmic-axial-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "ophthalmic-optical-coherence-tomography-b-scan-volume-analysis": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-tomography-b-scan-volume-analysis-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-optical-coherence-tomography-b-scan-volume-analysis-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-optical-coherence-tomography-b-scan-volume-analysis-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "ophthalmic-optical-coherence-tomography-en-face-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-tomography-en-face-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-optical-coherence-tomography-en-face-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ocular-region-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-optical-coherence-tomography-en-face-image-quality-rating",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "ophthalmic-photography-16-bit-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-photography-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ocular-region-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photographic-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "ophthalmic-photography-8-bit-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-photography-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ocular-region-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photographic-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "ophthalmic-thickness-map": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-thickness-map-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "supplemental-palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "bitmap-display-shutter",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-thickness-map",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-thickness-map-quality-rating",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "ophthalmic-tomography-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-tomography-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-tomography-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-tomography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-tomography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-tomography-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ocular-region-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "ophthalmic-visual-field-static-perimetry-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "visual-field-static-perimetry-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "visual-field-static-perimetry-test-parameters",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "visual-field-static-perimetry-test-reliability",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "visual-field-static-perimetry-test-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "visual-field-static-perimetry-test-results",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "ophthalmic-patient-clinical-information-and-test-lens-parameters",
            "usage": "U"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "parametric-map": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "parametric-map-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "floating-point-image-pixel",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "double-floating-point-image-pixel",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "parametric-map-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "parametric-map-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "bulk-motion-synchronization",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "patient-radiation-dose-structured-report": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "performed-imaging-agent-administration-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "pet-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "pet-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "pet-isotope",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "pet-multi-gated-acquisition",
            "usage": "C"
        },
        {
            "ie": "Series",
            "key": "nm-pet-patient-orientation",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-plane",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "pet-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "planar-mpr-volumetric-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volumetric-presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volumetric-presentation-state-relationship",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volume-cropping",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-view-description",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "multi-planar-reconstruction-geometry",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "mpr-volumetric-presentation-state-display",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volumetric-graphic-annotation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-animation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "common-instance-reference",
            "usage": "M"
        }
    ],
    "planned-imaging-agent-administration-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "procedure-log": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "protocol-approval": [
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Approval",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Approval",
            "key": "protocol-approval",
            "usage": "M"
        }
    ],
    "pseudo-color-softcopy-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-relationship",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-shutter",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-mask",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "mask",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "bitmap-display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-activation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "displayed-area",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "spatial-transformation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "modality-lut",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "softcopy-voi-lut",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "palette-color-lookup-table",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "icc-profile",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "radiopharmaceutical-radiation-dose-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "raw-data": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Raw Data",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Raw Data",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Raw Data",
            "key": "raw-data",
            "usage": "M"
        },
        {
            "ie": "Raw Data",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "real-time-audio-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "real-time-bulk-data-flow",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "real-time-audio-waveform-current-frame-functional-groups",
            "usage": "M"
        }
    ],
    "real-time-video-endoscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "real-time-bulk-data-flow",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "real-time-acquisition",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "real-time-video-endoscopic-image-current-frame-functional-groups",
            "usage": "M"
        }
    ],
    "real-time-video-photographic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "real-time-bulk-data-flow",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "real-time-acquisition",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "real-time-video-photographic-image-current-frame-functional-groups",
            "usage": "M"
        }
    ],
    "real-world-value-mapping": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "real-world-value-mapping-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Real World Value Mapping",
            "key": "real-world-value-mapping",
            "usage": "M"
        },
        {
            "ie": "Real World Value Mapping",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Real World Value Mapping",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "rendition-selection-document": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "key-object-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "key-object-document",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "respiratory-waveform": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "robotic-arm-radiation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "rt-radiation-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "robotic-arm-delivery-device",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "robotic-arm-path",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "robotic-arm-radiation-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-radiation-record-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "robotic-arm-delivery-device",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "robotic-arm-path",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "routine-scalp-electroencephalogram": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "rt-beams-delivery-instruction": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-beams-delivery-instruction",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "common-instance-reference",
            "usage": "C"
        },
        {
            "ie": "Plan",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "rt-beams-treatment-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-general-treatment-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-patient-setup",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-machine-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "measured-dose-reference-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "calculated-dose-reference-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-beams-session-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-summary-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "rt-brachy-application-setup-delivery-instruction": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-brachy-application-setup-delivery-instruction",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "rt-brachy-treatment-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-general-treatment-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-patient-setup",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-machine-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "measured-dose-reference-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "calculated-dose-reference-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-brachy-session-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-summary-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "rt-dose": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Dose",
            "key": "general-image",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "image-plane",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "image-pixel",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "multi-frame",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Dose",
            "key": "multi-frame-overlay",
            "usage": "U"
        },
        {
            "ie": "Dose",
            "key": "modality-lut",
            "usage": "U"
        },
        {
            "ie": "Dose",
            "key": "rt-dose",
            "usage": "M"
        },
        {
            "ie": "Dose",
            "key": "rt-dvh",
            "usage": "U"
        },
        {
            "ie": "Dose",
            "key": "structure-set",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "roi-contour",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "rt-dose-roi",
            "usage": "C"
        },
        {
            "ie": "Dose",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Dose",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Dose",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "rt-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "rt-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "modality-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "approval",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "rt-ion-beams-treatment-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-general-treatment-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-patient-setup",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-machine-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "measured-dose-reference-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "calculated-dose-reference-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-ion-beams-session-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-summary-record",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "rt-ion-plan": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-general-plan",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-prescription",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-ion-tolerance-tables",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-patient-setup",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-fraction-scheme",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-ion-beams",
            "usage": "C"
        },
        {
            "ie": "Plan",
            "key": "approval",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "rt-physician-intent": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "RT Physician Intent",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Physician Intent",
            "key": "rt-physician-intent",
            "usage": "M"
        },
        {
            "ie": "RT Physician Intent",
            "key": "rt-enhanced-prescription",
            "usage": "U"
        },
        {
            "ie": "RT Physician Intent",
            "key": "rt-treatment-phase-intent",
            "usage": "C"
        },
        {
            "ie": "RT Physician Intent",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Physician Intent",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Physician Intent",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-plan": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-general-plan",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-prescription",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-tolerance-tables",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-patient-setup",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-fraction-scheme",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "rt-beams",
            "usage": "C"
        },
        {
            "ie": "Plan",
            "key": "rt-brachy-application-setups",
            "usage": "C"
        },
        {
            "ie": "Plan",
            "key": "approval",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Plan",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "rt-radiation-record-set": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-radiation-record-set",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-dose-contribution-record",
            "usage": "C"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-radiation-salvage-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-radiation-record-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-radiation-salvage-record",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-radiation-set": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation Set",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation Set",
            "key": "rt-radiation-set",
            "usage": "M"
        },
        {
            "ie": "RT Radiation Set",
            "key": "rt-dose-contribution",
            "usage": "C"
        },
        {
            "ie": "RT Radiation Set",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation Set",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation Set",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-radiation-set-delivery-instruction": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "rt-radiation-set-delivery-instruction",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Plan",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-segment-annotation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "RT Segment Annotation",
            "key": "rt-segment-annotation",
            "usage": "M"
        },
        {
            "ie": "RT Segment Annotation",
            "key": "segment-reference",
            "usage": "M"
        },
        {
            "ie": "RT Segment Annotation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Segment Annotation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Segment Annotation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Segment Annotation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-structure-set": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Structure Set",
            "key": "structure-set",
            "usage": "M"
        },
        {
            "ie": "Structure Set",
            "key": "roi-contour",
            "usage": "M"
        },
        {
            "ie": "Structure Set",
            "key": "rt-roi-observations",
            "usage": "M"
        },
        {
            "ie": "Structure Set",
            "key": "approval",
            "usage": "U"
        },
        {
            "ie": "Structure Set",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Structure Set",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Structure Set",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "rt-treatment-preparation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "RT Treatment Preparation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Treatment Preparation",
            "key": "rt-treatment-preparation",
            "usage": "M"
        },
        {
            "ie": "RT Treatment Preparation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Treatment Preparation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Treatment Preparation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "rt-treatment-summary-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "rt-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-general-treatment-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "rt-treatment-summary-record",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Treatment Record",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Treatment Record",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "secondary-capture-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "sc-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sc-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "modality-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "segmentation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "segmentation-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "segmentation-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "segmentation-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "simplified-adult-echo-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "timezone",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "sleep-electroencephalogram": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform-identification",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "waveform",
            "usage": "M"
        },
        {
            "ie": "Waveform",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Waveform",
            "key": "waveform-annotation",
            "usage": "C"
        },
        {
            "ie": "Waveform",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "spatial-fiducials": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "spatial-fiducials-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Spatial Fiducials",
            "key": "spatial-fiducials",
            "usage": "M"
        },
        {
            "ie": "Spatial Fiducials",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Spatial Fiducials",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Spatial Fiducials",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "spatial-registration": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "spatial-registration-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Spatial Registration",
            "key": "spatial-registration",
            "usage": "M"
        },
        {
            "ie": "Spatial Registration",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Spatial Registration",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Spatial Registration",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "spectacle-prescription-report": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "stereometric-relationship": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "stereometric-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Stereometric Relationship",
            "key": "stereometric-relationship",
            "usage": "M"
        },
        {
            "ie": "Stereometric Relationship",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Stereometric Relationship",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "subjective-refraction-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "subjective-refraction-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "subjective-refraction-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "surface-scan-mesh": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "optical-surface-scanner-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "surface-mesh",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "uv-mapping",
            "usage": "U"
        },
        {
            "ie": "Surface",
            "key": "scan-procedure",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Surface",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "surface-scan-point-cloud": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "optical-surface-scanner-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "point-cloud",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "uv-mapping",
            "usage": "U"
        },
        {
            "ie": "Surface",
            "key": "scan-procedure",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Surface",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "surface-segmentation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "segmentation-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "surface-segmentation",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "surface-mesh",
            "usage": "M"
        },
        {
            "ie": "Surface",
            "key": "common-instance-reference",
            "usage": "C"
        },
        {
            "ie": "Surface",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Surface",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "tomotherapeutic-radiation": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "rt-radiation-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "tomotherapeutic-delivery-device",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "tomotherapeutic-beam",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "tomotherapeutic-radiation-record": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-rt-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "general-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-delivery-device-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "rt-radiation-record-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "tomotherapeutic-delivery-device",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "tomotherapeutic-beam",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "RT Delivered Radiation",
            "key": "radiotherapy-common-instance",
            "usage": "M"
        }
    ],
    "tractography-results": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "tractography-results-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Tractography Results",
            "key": "tractography-results",
            "usage": "M"
        },
        {
            "ie": "Tractography Results",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Tractography Results",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Tractography Results",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "us-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "us-region-calibration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "us-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "us-multi-frame-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "palette-color-lookup-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "us-region-calibration",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "us-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-overlay",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "video-endoscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "video-microscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "video-photographic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "visual-acuity-measurements": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "visual-acuity-measurements-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "general-ophthalmic-refractive-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "visual-acuity-measurements",
            "usage": "M"
        },
        {
            "ie": "Measurements",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "vl-endoscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "vl-microscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "optical-path",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "vl-photographic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "vl-photographic-equipment",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "vl-photographic-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "vl-photographic-geolocation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "vl-slide-coordinates-microscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "vl-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "slide-coordinates",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "optical-path",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        }
    ],
    "vl-whole-slide-microscopy-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "whole-slide-microscopy-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "vl-whole-slide-microscopy-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "whole-slide-microscopy-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "optical-path",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "slide-label",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "volume-rendering-volumetric-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volumetric-presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volumetric-presentation-state-relationship",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volume-cropping",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-view-description",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volume-render-geometry",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "render-shading",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "render-display",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "volumetric-graphic-annotation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-group",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-animation",
            "usage": "U"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "common-instance-reference",
            "usage": "M"
        }
    ],
    "wide-field-ophthalmic-photography-3d-coordinates-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-photography-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "wide-field-ophthalmic-photography-3d-coordinates",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "wide-field-ophthalmic-photography-quality-rating",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ocular-region-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photographic-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "wide-field-ophthalmic-photography-stereographic-projection-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "ophthalmic-photography-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "wide-field-ophthalmic-photography-stereographic-projection",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "wide-field-ophthalmic-photography-quality-rating",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "ocular-region-imaged",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photography-acquisition-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "ophthalmic-photographic-parameters",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "icc-profile",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "x-ray-3d-angiographic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-angiographic-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "cardiac-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "respiratory-synchronization",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "patient-orientation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image---equipment-coordinate-relationship",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-angiographic-image-contributing-sources",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-angiographic-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "x-ray-3d-craniofacial-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "enhanced-contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "acquisition-context",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-craniofacial-image-multi-frame-functional-groups",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "multi-frame-dimension",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "patient-orientation",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image---equipment-coordinate-relationship",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-craniofacial-image-contributing-sources",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-craniofacial-acquisition",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-3d-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "x-ray-angiographic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "mask",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "display-shutter",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-acquisition",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-collimator",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-table",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "xa-positioner",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "dx-detector",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-overlay",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "modality-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "x-ray-radiation-dose-sr": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "sr-document-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "C"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-general",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sr-document-content",
            "usage": "M"
        },
        {
            "ie": "Document",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "x-ray-radiofluoroscopic-image": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Frame of Reference",
            "key": "synchronization",
            "usage": "U"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "general-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "image-pixel",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "contrast-bolus",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "cine",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "multi-frame",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "frame-pointers",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "mask",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "display-shutter",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "device",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "intervention",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "specimen",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-image",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-acquisition",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "x-ray-collimator",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-table",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "xrf-positioner",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "x-ray-tomography-acquisition",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "dx-detector",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "overlay-plane",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "multi-frame-overlay",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "modality-lut",
            "usage": "C"
        },
        {
            "ie": "Image",
            "key": "voi-lut",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "sop-common",
            "usage": "M"
        },
        {
            "ie": "Image",
            "key": "common-instance-reference",
            "usage": "U"
        },
        {
            "ie": "Image",
            "key": "frame-extraction",
            "usage": "C"
        }
    ],
    "xa-defined-procedure-protocol": [
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "protocol-context",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "clinical-trial-context",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-specification",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "equipment-specification",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "instructions",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-positioning",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "general-defined-acquisition",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "general-defined-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "defined-storage",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "xa-performed-procedure-protocol": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "enhanced-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "xa-protocol-series",
            "usage": "M"
        },
        {
            "ie": "Frame of Reference",
            "key": "frame-of-reference",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "protocol-context",
            "usage": "M"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-protocol-context",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "instructions",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "patient-positioning",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "performed-xa-acquisition",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "performed-xa-reconstruction",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "performed-storage",
            "usage": "U"
        },
        {
            "ie": "Procedure Protocol",
            "key": "sop-common",
            "usage": "M"
        }
    ],
    "xa-xrf-grayscale-softcopy-presentation-state": [
        {
            "ie": "Patient",
            "key": "patient",
            "usage": "M"
        },
        {
            "ie": "Patient",
            "key": "clinical-trial-subject",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "general-study",
            "usage": "M"
        },
        {
            "ie": "Study",
            "key": "patient-study",
            "usage": "U"
        },
        {
            "ie": "Study",
            "key": "clinical-trial-study",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "general-series",
            "usage": "M"
        },
        {
            "ie": "Series",
            "key": "clinical-trial-series",
            "usage": "U"
        },
        {
            "ie": "Series",
            "key": "presentation-series",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "general-equipment",
            "usage": "M"
        },
        {
            "ie": "Equipment",
            "key": "enhanced-general-equipment",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-identification",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-relationship",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "presentation-state-shutter",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "bitmap-display-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-plane",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "overlay-activation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "displayed-area",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-annotation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "spatial-transformation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "graphic-layer",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "softcopy-voi-lut",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "xa-xrf-presentation-state-mask",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "xa-xrf-presentation-state-shutter",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "xa-xrf-presentation-state-presentation",
            "usage": "C"
        },
        {
            "ie": "Presentation State",
            "key": "softcopy-presentation-lut",
            "usage": "M"
        },
        {
            "ie": "Presentation State",
            "key": "sop-common",
            "usage": "M"
        }
    ]
}

SOP_CLASS_UID_IOD_KEY_MAP = {
    "1.2.840.10008.1.3.10": "basic-directory",
    "1.2.840.10008.5.1.4.1.1.1": "cr-image",
    "1.2.840.10008.5.1.4.1.1.1.1": "digital-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.1.1.1": "digital-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.1.2": "digital-mammography-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.1.2.1": "digital-mammography-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.1.3": "digital-intra-oral-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.1.3.1": "digital-intra-oral-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.104.1": "encapsulated-pdf",
    "1.2.840.10008.5.1.4.1.1.104.2": "encapsulated-cda",
    "1.2.840.10008.5.1.4.1.1.104.3": "encapsulated-stl",
    "1.2.840.10008.5.1.4.1.1.104.4": "encapsulated-obj",
    "1.2.840.10008.5.1.4.1.1.104.5": "encapsulated-mtl",
    "1.2.840.10008.5.1.4.1.1.11.1": "grayscale-softcopy-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.10": "volume-rendering-volumetric-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.11": "volume-rendering-volumetric-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.2": "color-softcopy-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.3": "pseudo-color-softcopy-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.4": "blending-softcopy-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.5": "xa-xrf-grayscale-softcopy-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.6": "planar-mpr-volumetric-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.7": "planar-mpr-volumetric-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.8": "advanced-blending-presentation-state",
    "1.2.840.10008.5.1.4.1.1.11.9": "volume-rendering-volumetric-presentation-state",
    "1.2.840.10008.5.1.4.1.1.12.1": "x-ray-angiographic-image",
    "1.2.840.10008.5.1.4.1.1.12.1.1": "enhanced-xa-image",
    "1.2.840.10008.5.1.4.1.1.12.2": "x-ray-radiofluoroscopic-image",
    "1.2.840.10008.5.1.4.1.1.12.2.1": "enhanced-xrf-image",
    "1.2.840.10008.5.1.4.1.1.128": "pet-image",
    "1.2.840.10008.5.1.4.1.1.128.1": "legacy-converted-enhanced-pet-image",
    "1.2.840.10008.5.1.4.1.1.13.1.1": "x-ray-3d-angiographic-image",
    "1.2.840.10008.5.1.4.1.1.13.1.2": "x-ray-3d-craniofacial-image",
    "1.2.840.10008.5.1.4.1.1.13.1.3": "breast-tomosynthesis-image",
    "1.2.840.10008.5.1.4.1.1.13.1.4": "breast-projection-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.13.1.5": "breast-projection-x-ray-image",
    "1.2.840.10008.5.1.4.1.1.130": "enhanced-pet-image",
    "1.2.840.10008.5.1.4.1.1.131": "basic-structured-display",
    "1.2.840.10008.5.1.4.1.1.14.1": "intravascular-optical-coherence-tomography-image",
    "1.2.840.10008.5.1.4.1.1.14.2": "intravascular-optical-coherence-tomography-image",
    "1.2.840.10008.5.1.4.1.1.2": "ct-image",
    "1.2.840.10008.5.1.4.1.1.2.1": "enhanced-ct-image",
    "1.2.840.10008.5.1.4.1.1.2.2": "legacy-converted-enhanced-ct-image",
    "1.2.840.10008.5.1.4.1.1.20": "nm-image",
    "1.2.840.10008.5.1.4.1.1.200.1": "ct-defined-procedure-protocol",
    "1.2.840.10008.5.1.4.1.1.200.2": "ct-performed-procedure-protocol",
    "1.2.840.10008.5.1.4.1.1.200.3": "protocol-approval",
    "1.2.840.10008.5.1.4.1.1.200.7": "xa-defined-procedure-protocol",
    "1.2.840.10008.5.1.4.1.1.200.8": "xa-performed-procedure-protocol",
    "1.2.840.10008.5.1.4.1.1.3.1": "us-multi-frame-image",
    "1.2.840.10008.5.1.4.1.1.30": "parametric-map",
    "1.2.840.10008.5.1.4.1.1.4": "mr-image",
    "1.2.840.10008.5.1.4.1.1.4.1": "enhanced-mr-image",
    "1.2.840.10008.5.1.4.1.1.4.2": "mr-spectroscopy",
    "1.2.840.10008.5.1.4.1.1.4.3": "enhanced-mr-color-image",
    "1.2.840.10008.5.1.4.1.1.4.4": "legacy-converted-enhanced-mr-image",
    "1.2.840.10008.5.1.4.1.1.481.1": "rt-image",
    "1.2.840.10008.5.1.4.1.1.481.10": "rt-physician-intent",
    "1.2.840.10008.5.1.4.1.1.481.11": "rt-segment-annotation",
    "1.2.840.10008.5.1.4.1.1.481.12": "rt-radiation-set",
    "1.2.840.10008.5.1.4.1.1.481.13": "c-arm-photon-electron-radiation",
    "1.2.840.10008.5.1.4.1.1.481.14": "tomotherapeutic-radiation",
    "1.2.840.10008.5.1.4.1.1.481.15": "robotic-arm-radiation",
    "1.2.840.10008.5.1.4.1.1.481.16": "rt-radiation-record-set",
    "1.2.840.10008.5.1.4.1.1.481.17": "rt-radiation-salvage-record",
    "1.2.840.10008.5.1.4.1.1.481.18": "tomotherapeutic-radiation-record",
    "1.2.840.10008.5.1.4.1.1.481.19": "c-arm-photon-electron-radiation-record",
    "1.2.840.10008.5.1.4.1.1.481.2": "rt-dose",
    "1.2.840.10008.5.1.4.1.1.481.20": "robotic-arm-radiation-record",
    "1.2.840.10008.5.1.4.1.1.481.21": "rt-radiation-set-delivery-instruction",
    "1.2.840.10008.5.1.4.1.1.481.22": "rt-treatment-preparation",
    "1.2.840.10008.5.1.4.1.1.481.3": "rt-structure-set",
    "1.2.840.10008.5.1.4.1.1.481.4": "rt-beams-treatment-record",
    "1.2.840.10008.5.1.4.1.1.481.5": "rt-plan",
    "1.2.840.10008.5.1.4.1.1.481.6": "rt-brachy-treatment-record",
    "1.2.840.10008.5.1.4.1.1.481.7": "rt-treatment-summary-record",
    "1.2.840.10008.5.1.4.1.1.481.8": "rt-ion-plan",
    "1.2.840.10008.5.1.4.1.1.481.9": "rt-ion-beams-treatment-record",
    "1.2.840.10008.5.1.4.1.1.6.1": "us-image",
    "1.2.840.10008.5.1.4.1.1.6.2": "enhanced-us-volume",
    "1.2.840.10008.5.1.4.1.1.66": "raw-data",
    "1.2.840.10008.5.1.4.1.1.66.1": "spatial-registration",
    "1.2.840.10008.5.1.4.1.1.66.2": "spatial-fiducials",
    "1.2.840.10008.5.1.4.1.1.66.3": "deformable-spatial-registration",
    "1.2.840.10008.5.1.4.1.1.66.4": "segmentation",
    "1.2.840.10008.5.1.4.1.1.66.5": "surface-segmentation",
    "1.2.840.10008.5.1.4.1.1.66.6": "tractography-results",
    "1.2.840.10008.5.1.4.1.1.67": "real-world-value-mapping",
    "1.2.840.10008.5.1.4.1.1.68.1": "surface-scan-mesh",
    "1.2.840.10008.5.1.4.1.1.68.2": "surface-scan-point-cloud",
    "1.2.840.10008.5.1.4.1.1.7": "secondary-capture-image",
    "1.2.840.10008.5.1.4.1.1.7.1": "multi-frame-single-bit-sc-image",
    "1.2.840.10008.5.1.4.1.1.7.2": "multi-frame-grayscale-byte-sc-image",
    "1.2.840.10008.5.1.4.1.1.7.3": "multi-frame-grayscale-word-sc-image",
    "1.2.840.10008.5.1.4.1.1.7.4": "multi-frame-true-color-sc-image",
    "1.2.840.10008.5.1.4.1.1.77.1.1": "vl-endoscopic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.1.1": "video-endoscopic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.2": "vl-microscopic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.2.1": "video-microscopic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.3": "vl-slide-coordinates-microscopic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.4": "vl-photographic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.4.1": "video-photographic-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.1": "ophthalmic-photography-8-bit-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.2": "ophthalmic-photography-16-bit-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.3": "stereometric-relationship",
    "1.2.840.10008.5.1.4.1.1.77.1.5.4": "ophthalmic-tomography-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.5": "wide-field-ophthalmic-photography-stereographic-projection-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.6": "wide-field-ophthalmic-photography-3d-coordinates-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.7": "ophthalmic-optical-coherence-tomography-en-face-image",
    "1.2.840.10008.5.1.4.1.1.77.1.5.8": "ophthalmic-optical-coherence-tomography-b-scan-volume-analysis",
    "1.2.840.10008.5.1.4.1.1.77.1.6": "vl-whole-slide-microscopy-image",
    "1.2.840.10008.5.1.4.1.1.77.1.7": "dermoscopic-photography-image",
    "1.2.840.10008.5.1.4.1.1.78.1": "lensometry-measurements",
    "1.2.840.10008.5.1.4.1.1.78.2": "autorefraction-measurements",
    "1.2.840.10008.5.1.4.1.1.78.3": "keratometry-measurements",
    "1.2.840.10008.5.1.4.1.1.78.4": "subjective-refraction-measurements",
    "1.2.840.10008.5.1.4.1.1.78.5": "visual-acuity-measurements",
    "1.2.840.10008.5.1.4.1.1.78.6": "spectacle-prescription-report",
    "1.2.840.10008.5.1.4.1.1.78.7": "ophthalmic-axial-measurements",
    "1.2.840.10008.5.1.4.1.1.78.8": "intraocular-lens-calculations",
    "1.2.840.10008.5.1.4.1.1.79.1": "macular-grid-thickness-and-volume-report",
    "1.2.840.10008.5.1.4.1.1.80.1": "ophthalmic-visual-field-static-perimetry-measurements",
    "1.2.840.10008.5.1.4.1.1.81.1": "ophthalmic-thickness-map",
    "1.2.840.10008.5.1.4.1.1.82.1": "corneal-topography-map",
    "1.2.840.10008.5.1.4.1.1.88.11": "basic-text-sr",
    "1.2.840.10008.5.1.4.1.1.88.22": "enhanced-sr",
    "1.2.840.10008.5.1.4.1.1.88.33": "comprehensive-sr",
    "1.2.840.10008.5.1.4.1.1.88.34": "comprehensive-3d-sr",
    "1.2.840.10008.5.1.4.1.1.88.35": "extensible-sr",
    "1.2.840.10008.5.1.4.1.1.88.40": "procedure-log",
    "1.2.840.10008.5.1.4.1.1.88.50": "mammography-cad-sr",
    "1.2.840.10008.5.1.4.1.1.88.59": "key-object-selection-document",
    "1.2.840.10008.5.1.4.1.1.88.65": "chest-cad-sr",
    "1.2.840.10008.5.1.4.1.1.88.67": "x-ray-radiation-dose-sr",
    "1.2.840.10008.5.1.4.1.1.88.68": "radiopharmaceutical-radiation-dose-sr",
    "1.2.840.10008.5.1.4.1.1.88.69": "colon-cad-sr",
    "1.2.840.10008.5.1.4.1.1.88.70": "implantation-plan-sr-document",
    "1.2.840.10008.5.1.4.1.1.88.71": "acquisition-context-sr",
    "1.2.840.10008.5.1.4.1.1.88.72": "simplified-adult-echo-sr",
    "1.2.840.10008.5.1.4.1.1.88.73": "patient-radiation-dose-structured-report",
    "1.2.840.10008.5.1.4.1.1.88.74": "planned-imaging-agent-administration-sr",
    "1.2.840.10008.5.1.4.1.1.88.75": "performed-imaging-agent-administration-sr",
    "1.2.840.10008.5.1.4.1.1.9.1.1": "12-lead-ecg",
    "1.2.840.10008.5.1.4.1.1.9.1.2": "general-ecg",
    "1.2.840.10008.5.1.4.1.1.9.1.3": "ambulatory-ecg",
    "1.2.840.10008.5.1.4.1.1.9.2.1": "hemodynamic-waveform",
    "1.2.840.10008.5.1.4.1.1.9.3.1": "basic-cardiac-electrophysiology-waveform",
    "1.2.840.10008.5.1.4.1.1.9.4.1": "basic-voice-audio-waveform",
    "1.2.840.10008.5.1.4.1.1.9.4.2": "general-audio-waveform",
    "1.2.840.10008.5.1.4.1.1.9.5.1": "arterial-pulse-waveform",
    "1.2.840.10008.5.1.4.1.1.9.6.1": "respiratory-waveform",
    "1.2.840.10008.5.1.4.1.1.9.6.2": "multi-channel-respiratory-waveform",
    "1.2.840.10008.5.1.4.1.1.9.7.1": "routine-scalp-electroencephalogram",
    "1.2.840.10008.5.1.4.1.1.9.7.2": "electromyogram",
    "1.2.840.10008.5.1.4.1.1.9.7.3": "electrooculogram",
    "1.2.840.10008.5.1.4.1.1.9.7.4": "sleep-electroencephalogram",
    "1.2.840.10008.5.1.4.1.1.9.8.1": "body-position-waveform",
    "1.2.840.10008.5.1.4.1.1.90.1": "content-assessment-results",
    "1.2.840.10008.5.1.4.1.1.91.1": "microscopy-bulk-simple-annotations",
    "1.2.840.10008.5.1.4.34.10": "rt-brachy-application-setup-delivery-instruction",
    "1.2.840.10008.5.1.4.34.7": "rt-beams-delivery-instruction",
    "1.2.840.10008.5.1.4.38.1": "hanging-protocol",
    "1.2.840.10008.5.1.4.39.1": "color-palette",
    "1.2.840.10008.5.1.4.43.1": "generic-implant-template",
    "1.2.840.10008.5.1.4.44.1": "implant-assembly-template",
    "1.2.840.10008.5.1.4.45.1": "implant-template-group"
}