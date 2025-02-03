import sys
sys.path.append('mdify')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

from huggingface_hub import snapshot_download
snapshot_download(repo_id='stefanodangelo/chartdet', local_dir='./', allow_patterns='*.pt')
snapshot_download(repo_id='omoured/YOLOv10-Document-Layout-Analysis', local_dir='models', allow_patterns='*l_best.pt')

from mdify.parsers import DocumentParser
from mdify.output import OutputArtifact
__all__ = ["DocumentParser", "OutputArtifact"]