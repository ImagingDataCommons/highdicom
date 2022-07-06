from uuid import uuid4
import unittest

import pydicom

from highdicom.uid import UID


class TestUID(unittest.TestCase):

    def test_construction_without_value(self):
        uid = UID()
        assert isinstance(uid, str)
        assert isinstance(uid, pydicom.uid.UID)
        assert uid.startswith('1.2.826.0.1.3680043.10.511.3.')

    def test_construction_with_value(self):
        value = '1.2.3.4'
        uid = UID(value)
        assert uid == value

    def test_construction_from_uuid(self):
        uuid = str(uuid4())
        uid = UID.from_uuid(uuid)
        assert uid.startswith('2.25.')
