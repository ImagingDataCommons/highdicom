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


PKG_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'src',
    'highdicom',
    '_standard',
)


def _load_json_from_file(filename):
    with open(filename) as f:
        return json.load(f)


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
        try:
            kw_path = [dictionary_keyword(t) for t in path]
        except KeyError:
            logger.error(f'keyword in path of attribute "{tag}" not found')
            continue
        mapping = {
            'keyword': keyword,
            'type': item['type'],
            'path': kw_path,
        }
        modules[item['moduleId']].append(mapping)
    return modules


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Positional argument is path to directory containing JSON files generated
    # using the dicom-standard Python package, see
    # https://github.com/innolitics/dicom-standard/tree/master/standard
    # If generating yourself using the instructions in that repo, be sure to
    # use "make all" instead of just "make", otherwise some of the required
    # files are skipped
    try:
        directory = sys.argv[1]
    except IndexError as e:
        raise ValueError('Path to directory must be provided.') from e
    if not os.path.exists(directory):
        raise OSError(f'Path does not exist: "{directory}"')
    if not os.path.isdir(directory):
        raise OSError(f'Path is not a directory: "{directory}"')

    now = datetime.datetime.now()
    current_date = datetime.datetime.date(now).strftime('%Y-%m-%d')
    current_time = datetime.datetime.time(now).strftime('%H:%M:%S')

    iods = _create_iods(directory)
    iods_filename = os.path.join(PKG_JSON_PATH, 'iods.json')
    with open(iods_filename, 'w') as jf:
        json.dump(iods, jf, indent=2)

    sop_to_iods = _create_sop_to_iods(directory)
    sop_to_iods_filename = os.path.join(PKG_JSON_PATH, 'sop_class_to_iod.json')
    with open(sop_to_iods_filename, 'w') as jf:
        json.dump(sop_to_iods, jf, indent=2)

    modules = _create_modules(directory)
    modules_filename = os.path.join(PKG_JSON_PATH, 'modules.json')
    with open(modules_filename, 'w') as jf:
        json.dump(modules, jf, indent=2)
