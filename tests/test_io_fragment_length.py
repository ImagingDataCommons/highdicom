"""Focused malformed-fragment checks without imaging fixtures."""
from types import SimpleNamespace
from unittest.mock import Mock

from pydicom.tag import ItemTag
import pytest

from highdicom.io import ImageFileReader


def test_read_frame_raw_rejects_undefined_fragment_length():
    stream = Mock()
    stream.read_tag.return_value = (ItemTag.group, ItemTag.element)
    stream.read_UL.return_value = 0xFFFFFFFF
    stream.read.side_effect = AssertionError('Payload read must not occur.')
    reader = SimpleNamespace(
        number_of_frames=1,
        _offset_table=[0],
        _first_frame_offset=0,
        _fp=stream,
        transfer_syntax_uid=SimpleNamespace(is_encapsulated=True),
    )

    with pytest.raises(OSError, match='undefined length'):
        ImageFileReader.read_frame_raw(reader, 0)

    stream.read.assert_not_called()
