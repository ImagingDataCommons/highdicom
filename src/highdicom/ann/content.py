"""Content that is specific to Annotation IODs."""
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code

from highdicom.ann.enum import (
    AnnotationCoordinateTypeValues,
    AnnotationGroupGenerationTypeValues,
    GraphicTypeValues,
)
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from highdicom._module_utils import check_required_attributes


class Measurements(Dataset):

    """Dataset describing measurements of annotations."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        values: np.array,
        unit: Union[Code, CodedConcept]
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Concept name
        values: numpy.ndarray
            One-dimensional array of floating-point values. Some values may be
            NaN (``numpy.nan``) if no measurement may be available for a given
            annotation. Values must be sorted such that the *n*-th value
            represents the measurement for the *n*-th annotation.
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code], optional
            Coded units of measurement (see :dcm:`CID 7181 <part16/sect_CID_7181.html>`
            "Abstract Multi-dimensional Image Model Component Units")

        """  # noqa: E501
        super().__init__()

        if isinstance(name, Code):
            name = CodedConcept(*name)
        self.ConceptNameCodeSequence = [name]

        if isinstance(unit, Code):
            unit = CodedConcept(*unit)
        self.MeasurementUnitsCodeSequence = [unit]

        is_nan = np.isnan(values)
        stored_values = np.array(values[~is_nan], np.float32)
        item = Dataset()
        item.FloatingPointValues = stored_values.tobytes()
        if np.any(is_nan):
            stored_indices = (np.where(~is_nan)[0] + 1).astype(np.int32)
            item.AnnotationIndexList = stored_indices.tobytes()
        self.MeasurementValuesSequence = [item]

    @property
    def name(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded name"""
        return self.ConceptNameCodeSequence[0]

    @property
    def unit(self) -> CodedConcept:
        """highdicom.sr.coding.CodedConcept: coded unit"""
        return self.MeasurementUnitsCodeSequence[0]

    def get_values(self, number_of_annotations: int) -> np.ndarray:
        """Get measured values for annotations.

        Parameters
        ----------
        number_of_annotations: int
            Number of annotations in the annoation group

        Returns
        -------
        numpy.ndarray
            One-dimensional array of floating-point numbers of length
            `number_of_annotations`. The array may be sparse and annotations
            for which no measurements are available have value ``numpy.nan``.

        """
        item = self.MeasurementValuesSequence[0]
        values = np.zeros((number_of_annotations, ), np.float32)
        values[:] = np.nan
        stored_values = np.frombuffer(item.FloatingPointValues, np.float32)
        if hasattr(item, 'AnnotationIndexList'):
            stored_indices = np.frombuffer(item.AnnotationIndexList, np.int32)
            # Convert from DICOM one-based to Python zero-based indexing
            stored_indices = stored_indices - 1
        else:
            stored_indices = np.arange(number_of_annotations)
        try:
            values[stored_indices] = stored_values
        except IndexError as error:
            raise IndexError(
                'Could not get values because of incorrect annotation indices: '
                f'{error}. The specified "number_of_annotations" may be '
                'incorrect.'
            )
        return values

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'Measurements':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an item of the Measurements Sequence.

        Returns
        -------
        highdicom.ann.Measurements
            Item of the Measurements Sequence

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        check_required_attributes(
            dataset,
            module='microscopy-bulk-simple-annotations',
            base_path=['AnnotationGroupSequence', 'MeasurementsSequence'],
        )
        measurements = deepcopy(dataset)
        measurements.__class__ = cls

        measurements.ConceptNameCodeSequence = [
            CodedConcept.from_dataset(
                measurements.ConceptNameCodeSequence[0]
            )
        ]
        measurements.MeasurementUnitsCodeSequence = [
            CodedConcept.from_dataset(
                measurements.MeasurementUnitsCodeSequence[0]
            )
        ]

        return measurements


class AnnotationGroup(Dataset):

    """Dataset describing a group of annotations."""

    def __init__(
        self,
        number: int,
        uid: str,
        label: str,
        annotated_property_category: Union[Code, CodedConcept],
        annotated_property_type: Union[Code, CodedConcept],
        graphic_type: Union[str, GraphicTypeValues],
        graphic_data: Sequence[np.ndarray],
        algorithm_type: Union[str, AnnotationGroupGenerationTypeValues],
        algorithm_identification: Optional[
            AlgorithmIdentificationSequence
        ] = None,
        measurements: Optional[Sequence[Measurements]] = None,
        description: Optional[str] = None
    ):
        """
        Parameters
        ----------
        number: int
            One-based number for identification of the annotation group
        uid: str
            Unique identifier of the annotation group
        label: str
            User-defined label for identification of the annotation group
        annotated_property_category: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Category of the property the annotated regions of interest
            represents, e.g.,
            ``Code("49755003", "SCT", "Morphologically Abnormal Structure")``
            (see :dcm:`CID 7150 <part16/sect_CID_7150.html>`
            "Segmentation Property Categories")
        annotated_property_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Property the annotated regions of interest represents, e.g.,
            ``Code("108369006", "SCT", "Neoplasm")``
            (see :dcm:`CID 8135 <part16/sect_CID_8135.html>`
            "Microscopy Annotation Property Types")
        graphic_type: Union[str, highdicom.ann.GraphicTypeValues]
            Graphic type of annotated regions of interest
        graphic_data: Sequence[numpy.ndarray]
            Graphic data (coordinates) of annotated regions of interest
        algorithm_type: Union[str, highdicom.ann.AnnotationGroupGenerationTypeValues]
            Type of algorithm that was used to generate the annotation
        algorithm_identification: Union[highdicom.AlgorithmIdentificationSequence, None], optional
            Information useful for identification of the algorithm, such
            as its name or version. Required unless the `algorithm_type` is
            ``"MANUAL"``
        measurements: Union[Sequence[highdicom.ann.Measurements], None], optional
            One or more sets of measurements for annotated regions of
            interest
        description: Union[str, None], optional
            Description of the annotation group

        """  # noqa: E501
        super().__init__()

        if not isinstance(number, int):
            raise TypeError('Argument "number" must be an integer.')
        if number < 1:
            raise ValueError('Argument "number" must be a positive integer.')

        self.AnnotationGroupNumber = number
        self.AnnotationGroupUID = str(uid)
        self.AnnotationGroupLabel = str(label)
        if description is not None:
            self.AnnotationGroupDescription = description

        algorithm_type = AnnotationGroupGenerationTypeValues(algorithm_type)
        self.AnnotationGroupGenerationType = algorithm_type.value
        if algorithm_type != AnnotationGroupGenerationTypeValues.MANUAL:
            if algorithm_identification is None:
                raise TypeError(
                    'Argument "algorithm_identification" must be provided if '
                    f'argument "algorithm_type" is "{algorithm_type.value}".'
                )
            if not isinstance(algorithm_identification,
                              AlgorithmIdentificationSequence):
                raise TypeError(
                    'Argument "algorithm_identification" must have type '
                    'AlgorithmIdentificationSequence.'
                )
            self.AnnotationGroupAlgorithmIdentificationSequence = \
                algorithm_identification

        if isinstance(annotated_property_category, Code):
            self.AnnotationPropertyCategoryCodeSequence = [
                CodedConcept(*annotated_property_category)
            ]
        else:
            self.AnnotationPropertyCategoryCodeSequence = [
                annotated_property_category,
            ]
        if isinstance(annotated_property_type, Code):
            self.AnnotationPropertyTypeCodeSequence = [
                CodedConcept(*annotated_property_type),
            ]
        else:
            self.AnnotationPropertyTypeCodeSequence = [
                annotated_property_type,
            ]

        self.NumberOfAnnotations = len(graphic_data)
        graphic_type = GraphicTypeValues(graphic_type)
        self.GraphicType = graphic_type.value

        try:
            coordinates = np.concatenate(graphic_data, axis=0)
        except ValueError:
            raise ValueError(
                'Items of argument "graphic_datat" must be arrays with the '
                'same dimensions.'
            )

        if coordinates.dtype.kind != 'f':
            raise ValueError(
                'Items of argument "graphic_data" must be arrays of '
                'floating-point numbers.'
            )
        if coordinates.ndim != 2:
            raise ValueError(
                'Items of argument "graphic_data" must be two-dimensional '
                'arrays.'
            )
        if coordinates.shape[1] not in (2, 3):
            raise ValueError(
                'Items of argument "graphic_data" must be two-dimensional '
                'arrays where the second array dimension has size 2 or 3.'
            )
        if np.sum(~np.isfinite(coordinates)) > 0:
            raise ValueError(
                'Items of argument "graphic_data" must be arrays of finite '
                'floating-point numbers. Some values are not finite, '
                'i.e., are either NaN, +inf, or -inf.'
            )

        if coordinates.shape[1] == 3:
            unique_z_values = np.unique(coordinates[:, 2])
            if len(unique_z_values) == 1:
                self.CommonZCoordinateValue = unique_z_values[0]
                coordinates_data = coordinates[:, 0:2].flatten()
                dimensionality = 2
            else:
                coordinates_data = coordinates.flatten()
                dimensionality = 3
        else:
            coordinates_data = coordinates.flatten()
            dimensionality = 2

        if coordinates.dtype == np.double:
            self.DoublePointCoordinatesData = coordinates_data.tobytes()
        else:
            self.PointCoordinatesData = coordinates_data.tobytes()

        if graphic_type in (
            GraphicTypeValues.POLYGON,
            GraphicTypeValues.POLYLINE,
        ):
            point_indices = np.concatenate([
                np.ones((item.shape[0] * dimensionality, ), np.int32) + i
                for i, item in enumerate(graphic_data)
            ])
            self.LongPrimitivePointIndexList = point_indices.tobytes()

        self.AnnotationAppliesToAllZPlanes = 'NO'
        self.AnnotationAppliesToAllOpticalPaths = 'YES'

        if measurements is not None:
            self.MeasurementsSequence = []
            for i, item in enumerate(measurements):
                if not isinstance(item, Measurements):
                    raise TypeError(
                        f'Item #{i} of argument "measurements" must have type '
                        'Measurements.'
                    )
                self.MeasurementsSequence.append(item)

    @property
    def label(self) -> str:
        """str: label"""
        return str(self.AnnotationGroupLabel)

    @property
    def number(self) -> int:
        """int: one-based identification number"""
        return int(self.AnnotationGroupNumber)

    @property
    def uid(self) -> UID:
        """highdicom.UID: unique identifier"""
        return UID(self.AnnotationGroupUID)

    @property
    def graphic_type(self) -> GraphicTypeValues:
        """highdicom.ann.GraphicTypeValues: graphic type"""
        return GraphicTypeValues(self.GraphicType)

    @property
    def annotated_property_category(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded annotated property category"""
        return self.AnnotationPropertyCategoryCodeSequence[0]

    @property
    def annotated_property_type(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded annotated property type"""
        return self.AnnotationPropertyTypeCodeSequence[0]

    @property
    def algorithm_type(self) -> AnnotationGroupGenerationTypeValues:
        """highdicom.ann.AnnotationGroupGenerationTypeValues: algorithm type"""
        return AnnotationGroupGenerationTypeValues(
            self.AnnotationGroupGenerationType
        )

    @property
    def algorithm_identification(
        self
    ) -> Union[AlgorithmIdentificationSequence, None]:
        """Union[highdicom.AlgorithmIdentificationSequence, None]:
            Information useful for identification of the algorithm, if any.

        """
        if hasattr(self, 'AnnotationGroupAlgorithmIdentificationSequence'):
            return self.AnnotationGroupAlgorithmIdentificationSequence
        return None

    def get_coordinates(
        self,
        annotation_number: int,
        coordinate_type: Union[str, AnnotationCoordinateTypeValues]
    ) -> np.ndarray:
        """Get spatial coordinates of a graphical annotation.

        Parameters
        ----------
        annotation_number: int
            One-based identification number of the annotation
        coordinate_type: Union[str, highdicom.ann.AnnotationCoordinateTypeValues]
            Coordinate type of annotation

        Returns
        -------
        numpy.ndarray
            Two-dimensional array of floating-point values representing either
            2D or 3D spatial coordinates of a graphical annotation

        """  # noqa: E501
        coordinate_type = AnnotationCoordinateTypeValues(coordinate_type)
        if coordinate_type == AnnotationCoordinateTypeValues.SCOORD:
            coordinate_dimensionality = 2
        else:
            coordinate_dimensionality = 3

        coordinate_index = self._get_coordinate_index(
            annotation_number,
            coordinate_dimensionality
        )
        try:
            coordinates_data = getattr(self, 'DoublePointCoordinatesData')
            coordinates = np.frombuffer(coordinates_data, np.float64)
        except AttributeError:
            coordinates_data = getattr(self, 'PointCoordinatesData')
            coordinates = np.frombuffer(coordinates_data, np.float32)
        coordinates = coordinates[coordinate_index]

        if hasattr(self, 'CommonZCoordinateValue'):
            coordinates = coordinates.reshape(-1, 2)
            if coordinates.dtype == np.double:
                z_values = np.zeros((coordinates.shape[0], 1), np.float64)
            else:
                z_values = np.zeros((coordinates.shape[0], 1), np.float32)
            z_values[:] = float(self.CommonZCoordinateValue)
            return np.concatenate([coordinates, z_values], axis=1)
        else:
            return coordinates.reshape(-1, coordinate_dimensionality)

    @property
    def number_of_annotations(self) -> int:
        """int: Number of annotations in group"""
        return int(self.NumberOfAnnotations)

    def get_measurements(self) -> Tuple[
        List[CodedConcept], np.ndarray, List[CodedConcept]
    ]:
        """numpy.ndarray: Matrix of measurement values"""
        number_of_annotations = self.number_of_annotations
        if hasattr(self, 'MeasurementsSequence'):
            values = np.vstack([
                item.get_values(number_of_annotations)
                for item in self.MeasurementsSequence
            ]).T
            names = [item.name for item in self.MeasurementsSequence]
            units = [item.unit for item in self.MeasurementsSequence]
        else:
            values = np.empty((number_of_annotations, 0), np.float32)
            names = []
            units = []
        return (names, values, units)

    def _get_coordinate_index(
        self,
        annotation_number: int,
        coordinate_dimensionality: int
    ) -> np.ndarray:
        """Get coordinate index.

        Parameters
        ----------
        annotation_number: int
            One-based identification number of the annotation
        coordinate_dimensionality: int
            Dimensionality of coordinate points

        Returns
        -------
        numpy.ndarray
            One-dimensional array of zero-based index values to obtain the
            coordinate points for a given annotation

        """  # noqa: E501
        graphic_type = self.graphic_type
        if graphic_type in (
            GraphicTypeValues.POLYGON,
            GraphicTypeValues.POLYLINE,
        ):
            point_indices = np.frombuffer(
                self.LongPrimitivePointIndexList,
                dtype=np.int32
            )
            coordinate_index = np.where(point_indices == annotation_number)[0]
        else:
            if hasattr(self, 'CommonZCoordinateValue'):
                stored_coordinate_dimensionality = 2
            else:
                stored_coordinate_dimensionality = coordinate_dimensionality
            if graphic_type in (
                GraphicTypeValues.ELLIPSE,
                GraphicTypeValues.RECTANGLE,
            ):
                length = 4 * stored_coordinate_dimensionality
            elif graphic_type == GraphicTypeValues.POINT:
                length = stored_coordinate_dimensionality
            else:
                raise ValueError(
                    'Encountered unexpected graphic type '
                    f'"{graphic_type.value}".'
                )
            start = (annotation_number - 1) * length
            end = start + length
            coordinate_index = np.arange(start, end)

        return coordinate_index

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'AnnotationGroup':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an item of the Annotation Group Sequence.

        Returns
        -------
        highdicom.ann.AnnotationGroup
            Item of the Annotation Group Sequence

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        check_required_attributes(
            dataset,
            module='microscopy-bulk-simple-annotations',
            base_path=['AnnotationGroupSequence'],
        )
        group = deepcopy(dataset)
        group.__class__ = cls

        group.AnnotationPropertyCategoryCodeSequence = [
            CodedConcept.from_dataset(
                group.AnnotationPropertyCategoryCodeSequence[0]
            )
        ]
        group.AnnotationPropertyTypeCodeSequence = [
            CodedConcept.from_dataset(
                group.AnnotationPropertyTypeCodeSequence[0]
            )
        ]
        if hasattr(group, 'AnnotationGroupAlgorithmIdentificationSequence'):
            group.AnnotationGroupAlgorithmIdentificationSequence = \
                AlgorithmIdentificationSequence.from_sequence(
                    group.AnnotationGroupAlgorithmIdentificationSequence
                )
        if hasattr(group, 'MeasurementsSequence'):
            group.MeasurementsSequence = [
                Measurements.from_dataset(ds)
                for ds in group.MeasurementsSequence
            ]

        return group
