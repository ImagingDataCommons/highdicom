from highdicom import UID
from highdicom.map.sop import ParametricMap
from highdicom.map.content import RealWorldValueMapping
from pydicom.filereader import dcmread
import numpy as np

sm_image = dcmread("data/test_files/sm_image.dcm")
rows = sm_image.Rows
cols = sm_image.Columns

test_frames = np.random.choice([0.0, 0.5, 1.0], (15, cols, rows))

test_map = ParametricMap(
    [sm_image],
    test_frames[0],
    UID(),
    1,
    UID(),
    1,
    "Big Stupid Butt Manufacturing Concern",
    "123",
    "Asstoot3600",
    "NO",
    "0.0.1",
)

test_map.save_as("test.dcm")

print("Wazzaaa")
