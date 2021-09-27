#!/usr/bin/env python3
import datetime
import collections
import json
import logging
import os
import sys

from pydicom.datadict import dictionary_keyword
from pydicom.tag import Tag

logger = logging.getLogger(__name__)


PGK_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'src',
    'highdicom'
)


def _load_json_from_file(filename):
    with open(filename) as f:
        return json.load(f)


def _dump_json(data):
    return json.dumps(data, indent=4, sort_keys=True)


def _create_sop_to_iods(directory):
    filename = os.path.join(directory, 'sops.json')
    sops = _load_json_from_file(filename)

    filename = os.path.join(directory, 'ciods.json')
    ciods = _load_json_from_file(filename)

    sop_id_to_ciod_name = {sop['id']: sop['ciod'] for sop in sops}
    ciod_name_to_ciod_id = {ciod['name']: ciod['id'] for ciod in ciods}
    sop_id_to_ciod_id = {}
    for sop_id in sop_id_to_ciod_name:
        ciod_name = sop_id_to_ciod_name[sop_id]
        try:
            sop_id_to_ciod_id[sop_id] = ciod_name_to_ciod_id[ciod_name]
        except KeyError:
            logger.error(f'could not map IOD "{ciod_name}"')
    return sop_id_to_ciod_id


def _create_iods(directory):
    filename = os.path.join(directory, 'ciod_to_modules.json')
    ciod_to_modules = _load_json_from_file(filename)
    iods = collections.defaultdict(list)
    for item in ciod_to_modules:
        mapping = {
            'key': item['moduleId'],
            'usage': item['usage'],
            'ie': item['informationEntity'],
        }
        iods[item['ciodId']].append(mapping)
    return iods


def _create_modules(directory):
    filename = os.path.join(directory, 'module_to_attributes.json')
    module_to_attributes = _load_json_from_file(filename)
    modules = collections.defaultdict(list)
    for item in module_to_attributes:
        path = item['path'].split(':')[1:]
        tag_string = path.pop(-1)
        # Handle attributes used for real-time communication, which are neither
        # in DicomDictionary nor in RepeaterDictionary
        if any(p.startswith('0006') for p in path):
            logger.warning(f'skip attribute "{tag_string}"')
            continue
        logger.debug(f'add attribute "{tag_string}"')
        # Handle attributes that are in RepeatersDictionary
        tag_string = tag_string.replace('xx', '00')
        tag = Tag(tag_string)
        try:
            keyword = dictionary_keyword(tag)
        except KeyError:
            logger.error(f'keyword not found for attribute "{tag}"')
            continue
        mapping = {
            'keyword': keyword,
            'type': item['type'],
            'path': [dictionary_keyword(t) for t in path],
        }
        modules[item['moduleId']].append(mapping)
    return modules


if __name__ == '__main__':

    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Positional argument is path to directory containing JSON files generated
    # using the dicom-standard Python package, see
    # https://github.com/innolitics/dicom-standard/tree/master/standard
    try:
        directory = sys.argv[1]
    except IndexError:
        raise ValueError('Path to directory must be provided.')
    if not os.path.exists(directory):
        raise IOError('Path does not exist: "{}"'.format(directory))
    if not os.path.isdir(directory):
        raise IOError('Path is not a directory: "{}"'.format(directory))

    now = datetime.datetime.now()
    current_date = datetime.datetime.date(now).strftime('%Y-%m-%d')
    current_time = datetime.datetime.time(now).strftime('%H:%M:%S')

    iods = _create_iods(directory)
    iods_docstr = '\n'.join([
        '"""DICOM Information Object Definitions (IODs)',
        f'auto-generated on {current_date} at {current_time}.',
        '"""',
        'from typing import Dict, List'
    ])
    sop_to_iods = _create_sop_to_iods(directory)
    iods_filename = os.path.join(PGK_PATH, '_iods.py')
    with open(iods_filename, 'w') as fp:
        fp.write(iods_docstr)
        fp.write('\n\n')
        iods_formatted = _dump_json(iods).replace('null', 'None')
        fp.write(
            'IOD_MODULE_MAP: Dict[str, List[Dict[str, str]]] = {}'.format(
                iods_formatted
            )
        )
        fp.write('\n\n')
        sop_to_iods_formatted = _dump_json(sop_to_iods).replace('null', 'None')
        fp.write('SOP_CLASS_UID_IOD_KEY_MAP = {}'.format(sop_to_iods_formatted))

    modules = _create_modules(directory)
    modules_docstr = '\n'.join([
        '"""DICOM modules'
        f'auto-generated on {current_date} at {current_time}.'
        '"""',
        'from typing import Dict, List, Sequence, Union'
    ])
    modules_filename = os.path.join(PGK_PATH, '_modules.py')
    with open(modules_filename, 'w') as fp:
        fp.write(modules_docstr)
        fp.write('\n\n')
        modules_formatted = _dump_json(modules).replace('null', 'None')
        fp.write(
            'MODULE_ATTRIBUTE_MAP: Dict[str, List[Dict[str, Union[str, Sequence[str]]]]] = {}'.format(  # noqa: E501
                modules_formatted
            )
        )
