"""Content that is specific to Annotation IODs."""
from copy import deepcopy
from typing import cast, List, Optional, Sequence, Tuple, Union

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
        values: np.ndarray,
        unit: Union[Code, CodedConcept]
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Concept name
        values: numpy.ndarray
            One-dimensional array of floating-point values. Some values may be
            NaN (``numpy.nan``) if no measurement is available for a given
            annotation. Values must be sorted such that the *n*-th value
            represents the measurement for the *n*-th annotation.
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code], optional
            Coded units of measurement (see :dcm:`CID 7181 <part16/sect_CID_7181.html>`
            "Abstract Multi-dimensional Image Model Component Units")

        """  # noqa: E501
        super().__init__()

        if isinstance(name, Code):
            name = CodedConcept.from_code(name)
        self.ConceptNameCodeSequence = [name]

        if isinstance(unit, Code):
            unit = CodedConcept.from_code(unit)
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
        """highdicom.sr.CodedConcept: coded unit"""
        return self.MeasurementUnitsCodeSequence[0]

    def get_values(self, number_of_annotations: int) -> np.ndarray:
        """Get measured values for annotations.

        Parameters
        ----------
        number_of_annotations: int
            Number of annotations in the annotation group

        Returns
        -------
        numpy.ndarray
            One-dimensional array of floating-point numbers of length
            `number_of_annotations`. The array may be sparse and annotations
            for which no measurements are available have value ``numpy.nan``.

        Raises
        ------
        IndexError
            In case the measured values cannot be indexed given the indices
            stored in the Annotation Index List.

        """
        item = self.MeasurementValuesSequence[0]
        values = np.zeros((number_of_annotations, ), np.float32)
        values[:] = np.float32(np.nan)
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
                'Could not get values of measurements because of incorrect '
                f'annotation indices: {error}. This may either be due to '
                'incorrect encoding of the measurements or due to incorrectly '
                'specified "number_of_annotations".'
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

        return cast(Measurements, measurements)


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
        description: Optional[str] = None,
        anatomic_regions: Optional[
            Sequence[Union[Code, CodedConcept]]
        ] = None,
        primary_anatomic_structures: Optional[
            Sequence[Union[Code, CodedConcept]]
        ] = None
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
            Array of ordered spatial coordinates, where each row of an array
            represents a (Column,Row) coordinate pair or (X,Y,Z) coordinate
            triplet.
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
        anatomic_regions: Union[Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]], None], optional
            Anatomic region(s) into which annotations fall
        primary_anatomic_structures: Union[Sequence[Union[highdicom.sr.Code, highdicom.sr.CodedConcept]], None], optional
            Anatomic structure(s) the annotations represent
            (see CIDs for domain-specific primary anatomic structures)

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
                CodedConcept.from_code(annotated_property_category)
            ]
        else:
            self.AnnotationPropertyCategoryCodeSequence = [
                annotated_property_category,
            ]
        if isinstance(annotated_property_type, Code):
            self.AnnotationPropertyTypeCodeSequence = [
                CodedConcept.from_code(annotated_property_type),
            ]
        else:
            self.AnnotationPropertyTypeCodeSequence = [
                annotated_property_type,
            ]

        self.NumberOfAnnotations = len(graphic_data)
        graphic_type = GraphicTypeValues(graphic_type)
        self.GraphicType = graphic_type.value

        for i in range(len(graphic_data)):
            num_coords = graphic_data[i].shape[0]
            if graphic_type == GraphicTypeValues.POINT:
                if num_coords != 1:
                    raise ValueError(
                        f'Graphic data of annotation #{i + 1} of graphic type '
                        '"POINT" must be a single coordinate.'
                    )
            elif graphic_type == GraphicTypeValues.RECTANGLE:
                if num_coords != 4:
                    raise ValueError(
                        f'Graphic data of annotation #{i + 1} of graphic type '
                        '"RECTANGLE" must be four coordinates.'
                    )
            elif graphic_type == GraphicTypeValues.ELLIPSE:
                if num_coords != 4:
                    raise ValueError(
                        f'Graphic data of annotation #{i + 1} of graphic type '
                        '"ELLIPSE" must be four coordinates.'
                    )
            elif graphic_type == GraphicTypeValues.POLYLINE:
                if num_coords < 2:
                    raise ValueError(
                        f'Graphic data of annotation #{i + 1} of graphic type '
                        '"POLYLINE" must be at least two coordinates.'
                    )
            elif graphic_type == GraphicTypeValues.POLYGON:
                if num_coords < 3:
                    raise ValueError(
                        f'Graphic data of annotation #{i + 1} of graphic type '
                        '"POLYGON" must be at least three coordinates.'
                    )
                if np.array_equal(graphic_data[i][0], graphic_data[i][-1]):
                    raise ValueError(
                        'The first and last coordinate of graphic data of '
                        f'annotation #{i + 1} of graphic type "POLYGON" '
                        'must not be identical. '
                        'Note that the ANN Graphic Type is different in this '
                        'respect from the corresponding SR Graphic Type.'
                    )
            else:
                raise ValueError(
                    f'Graphic data of annotation #{i + 1} has an unknown '
                    'graphic type.'
                )

        try:
            coordinates = np.concatenate(graphic_data, axis=0)
        except ValueError:
            raise ValueError(
                'Items of argument "graphic_data" must be arrays with the '
                'same dimensions.'
            )

        if coordinates.dtype.kind in ('u', 'i'):
            coordinates = coordinates.astype(np.float32)

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
        coordinate_type = AnnotationCoordinateTypeValues.SCOORD
        if coordinates.shape[1] == 3:
            coordinate_type = AnnotationCoordinateTypeValues.SCOORD3D

        if not np.all(np.isfinite(coordinates)):
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

        self._graphic_data = {coordinate_type: graphic_data}

        if graphic_type in (
            GraphicTypeValues.POLYGON,
            GraphicTypeValues.POLYLINE,
        ):
            spans = [item.shape[0] * dimensionality for item in graphic_data]
            point_indices = np.cumsum(spans, dtype=np.int32) + 1
            point_indices = np.concatenate([
                np.array([1], dtype=np.int32),
                point_indices[:-1]
            ])
            self.LongPrimitivePointIndexList = point_indices.tobytes()

        self.AnnotationAppliesToAllZPlanes = 'NO'
        self.AnnotationAppliesToAllOpticalPaths = 'YES'

        if measurements is not None:
            self.MeasurementsSequence = []
            for i, item in enumerate(measurements):
                if not isinstance(item, Measurements):
                    raise TypeError(
                        f'Item #{i} of argument "measurements" must have '
                        'type Measurements.'
                    )
                error_message = (
                    f'The number of values of item #{i} of argument '
                    '"measurements" must match the number of annotations.'
                )
                try:
                    measured_values = item.get_values(self.NumberOfAnnotations)
                except IndexError:
                    raise ValueError(error_message)
                if len(measured_values) != self.NumberOfAnnotations:
                    # This should not occur, but safety first.
                    raise ValueError(error_message)
                self.MeasurementsSequence.append(item)

        if anatomic_regions is not None:
            self.AnatomicRegionSequence = [
                CodedConcept.from_code(region)
                for region in anatomic_regions
            ]
        if primary_anatomic_structures is not None:
            self.PrimaryAnatomicStructureSequence = [
                CodedConcept.from_code(structure)
                for structure in primary_anatomic_structures
            ]

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

    @property
    def anatomic_regions(self) -> List[CodedConcept]:
        """List[highdicom.sr.CodedConcept]:
            List of anatomic regions into which the annotations fall.
            May be empty.

        """
        if not hasattr(self, 'AnatomicRegionSequence'):
            return []
        return list(self.AnatomicRegionSequence)

    @property
    def primary_anatomic_structures(self) -> List[CodedConcept]:
        """List[highdicom.sr.CodedConcept]:
            List of anatomic anatomic structures the annotations represent.
            May be empty.

        """
        if not hasattr(self, 'PrimaryAnatomicStructureSequence'):
            return []
        return list(self.PrimaryAnatomicStructureSequence)

    def get_graphic_data(
        self,
        coordinate_type: Union[str, AnnotationCoordinateTypeValues]
    ) -> List[np.ndarray]:
        """Get spatial coordinates of all graphical annotations.

        Parameters
        ----------
        coordinate_type: Union[str, highdicom.ann.AnnotationCoordinateTypeValues]
            Coordinate type of annotation

        Returns
        -------
        List[numpy.ndarray]
            Two-dimensional array of floating-point values representing either
            2D or 3D spatial coordinates for each graphical annotation

        """  # noqa: E501
        coordinate_type = AnnotationCoordinateTypeValues(coordinate_type)
        if self._graphic_data:
            if coordinate_type not in self._graphic_data:
                raise ValueError(
                    'Graphic data is not available for Annotation Coordinate '
                    f'Type "{coordinate_type.value}".'
                )
        else:
            if coordinate_type == AnnotationCoordinateTypeValues.SCOORD:
                coordinate_dimensionality = 2
            else:
                coordinate_dimensionality = 3

            try:
                coordinates_data = getattr(self, 'DoublePointCoordinatesData')
                coordinates_dtype = np.float64
            except AttributeError:
                coordinates_data = getattr(self, 'PointCoordinatesData')
                coordinates_dtype = np.float32
            decoded_coordinates_data = np.frombuffer(
                coordinates_data,
                coordinates_dtype
            )

            if hasattr(self, 'CommonZCoordinateValue'):
                stored_coordinate_dimensionality = 2
            else:
                stored_coordinate_dimensionality = coordinate_dimensionality

            # Reshape array to stack of points
            decoded_coordinates_data = decoded_coordinates_data.reshape(
                -1,
                stored_coordinate_dimensionality
            )

            if hasattr(self, 'CommonZCoordinateValue'):
                # Add in a column for the shared z coordinate
                z_values = np.full(
                    shape=(decoded_coordinates_data.shape[0], 1),
                    fill_value=self.CommonZCoordinateValue,
                    dtype=coordinates_dtype
                )
                decoded_coordinates_data = np.concatenate(
                    [decoded_coordinates_data, z_values],
                    axis=1
                )

            # Split into objects down the first dimension
            graphic_type = self.graphic_type
            if graphic_type in (
                GraphicTypeValues.RECTANGLE,
                GraphicTypeValues.ELLIPSE,
            ):
                # Fixed 4 coordinates per object
                split_param: Union[
                    int,
                    Sequence[int]
                ] = len(decoded_coordinates_data) // 4
            elif graphic_type == GraphicTypeValues.POINT:
                # Fixed 1 coordinate per object
                split_param = len(decoded_coordinates_data)
            elif graphic_type in (
                GraphicTypeValues.POLYLINE,
                GraphicTypeValues.POLYGON,
            ):
                # Variable number of coordinates per point
                point_indices = np.frombuffer(
                    self.LongPrimitivePointIndexList,
                    dtype=np.int32
                ) - 1
                split_param = (
                    point_indices // stored_coordinate_dimensionality
                )[1:]
            else:
                raise ValueError(
                    'Encountered unexpected graphic type '
                    f'"{graphic_type.value}".'
                )

            graphic_data = np.split(
                decoded_coordinates_data,
                indices_or_sections=split_param
            )

            self._graphic_data[coordinate_type] = graphic_data

        return self._graphic_data[coordinate_type]

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
        graphic_data = self.get_graphic_data(coordinate_type)
        annotation_index = annotation_number - 1
        return graphic_data[annotation_index]

    @property
    def number_of_annotations(self) -> int:
        """int: Number of annotations in group"""
        return int(self.NumberOfAnnotations)

    def get_measurements(
        self,
        name: Optional[Union[Code, CodedConcept]] = None
    ) -> Tuple[
        List[CodedConcept], np.ndarray, List[CodedConcept]
    ]:
        """Get measurements.

        Parameters
        ----------
        name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Name by which measurements should be filtered

        Returns
        -------
        names: List[highdicom.sr.CodedConcept]
            Names of measurements
        values: numpy.ndarray
            Two-dimensional array of measurement floating point values. The
            array has shape n x m, where where *n* is the number of annotations
            and *m* is the number of measurements. The array may contain
            ``numpy.nan`` values in case a measurement is not available for a
            given annotation.
        units: List[highdicom.sr.CodedConcept]
            Units of measurements

        """  # noqa: E501
        number_of_annotations = self.number_of_annotations
        if hasattr(self, 'MeasurementsSequence'):
            values = [
                item.get_values(number_of_annotations)
                for item in self.MeasurementsSequence
                if name is None or item.name == name
            ]
            if len(values) > 0:
                value_array = np.vstack(values).T
            else:
                value_array = np.empty((number_of_annotations, 0), np.float32)
            names = [
                item.name for item in self.MeasurementsSequence
                if name is None or item.name == name
            ]
            units = [
                item.unit for item in self.MeasurementsSequence
                if name is None or item.name == name
            ]
        else:
            value_array = np.empty((number_of_annotations, 0), np.float32)
            names = []
            units = []
        return (names, value_array, units)

    def _get_coordinate_index(
        self,
        annotation_number: int,
        coordinate_dimensionality: int,
        number_of_coordinates: int
    ) -> np.ndarray:
        """Get coordinate index.

        Parameters
        ----------
        annotation_number: int
            One-based identification number of the annotation
        coordinate_dimensionality: int
            Dimensionality of coordinate points
        number_of_coordinates: int
            Total number of coordinate points

        Returns
        -------
        numpy.ndarray
            One-dimensional array of zero-based index values to obtain the
            coordinate points for a given annotation

        """  # noqa: E501
        annotation_index = annotation_number - 1
        graphic_type = self.graphic_type
        if graphic_type in (
            GraphicTypeValues.POLYGON,
            GraphicTypeValues.POLYLINE,
        ):
            point_indices = np.frombuffer(
                self.LongPrimitivePointIndexList,
                dtype=np.int32
            )
            start = point_indices[annotation_index] - 1
            try:
                end = point_indices[annotation_index + 1] - 1
            except IndexError:
                end = number_of_coordinates
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
            start = annotation_index * length
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
        group._graphic_data = {}

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
        if hasattr(group, 'AnatomicRegionSequence'):
            group.AnatomicRegionSequence = [
                CodedConcept.from_dataset(ds)
                for ds in group.AnatomicRegionSequence
            ]
        if hasattr(group, 'PrimaryAnatomicStructureSequence'):
            group.PrimaryAnatomicStructureSequence = [
                CodedConcept.from_dataset(ds)
                for ds in group.PrimaryAnatomicStructureSequence
            ]

        return cast(AnnotationGroup, group)
