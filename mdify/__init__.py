import sys
sys.path.append('mdify')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

from mdify.src.parsers import DocumentParser
from mdify.src.output import OutputArtifact
__all__ = ["DocumentParser", "OutputArtifact"]