.. _other_libraries:

Interactions with Other Libraries
=================================

Highdicom is able to interact with other imaging and scientific python
libraries. These integrations are intended to provide streamlined access to
complementary functionality available in external ecosystems that would
otherwise require custom user code.

Because these libraries are not required for all workflows, they are
treated as optional dependencies and are not installed as part of the
default ``highdicom`` installation. Functionality requiring an external
library becomes available when an appropriate version of the corresponding
package is installed alongside ``highdicom``. The following sections describe
which libraries are currently supported and the additional features each
library enables.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   itk_lib
   sitk_lib
