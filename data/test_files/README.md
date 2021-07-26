# Test Files

These files are used for automated tests of the highdicom library. Note that
many more test files are available as part of the pydicom package.

### Images
* `ct_image.dcm` - A small CT image.
* `sm_image.dcm` - A slide microscopy image.

### Segmentation Images
* `seg_image_ct_binary.dcm` - Segmentation image of `BINARY` type derived
  from the series of small CT images in the pydicom test files
  (`dicomdirtests/77654033/CT2/*`) using the highdicom library.
* `seg_image_ct_binary_overlap.dcm` - Segmentation image of `BINARY` type derived
  from the series of small CT images in the pydicom test files
  (`dicomdirtests/77654033/CT2/*`) using the highdicom library. This example
  contains 2 overlapping segments.
* `seg_image_ct_binary_fractional.dcm` - Segmentation image of `FRACTIONAL`
  type but with binary values (i.e. only 0.0 and 1.0) derived from the series
  of small CT images in the pydicom test files (`dicomdirtests/77654033/CT2/*`)
  using the highdicom library.
* `seg_image_ct_true_fractional.dcm` - Segmentation image of `FRACTIONAL`
  type with true fractional values derived from the series
  of small CT images in the pydicom test files (`dicomdirtests/77654033/CT2/*`)
  using the highdicom library.

### Structured Report Documents
* `sr_document.dcm` - A simple SR document following TID 1500 describing
   measurements of a CT image.
