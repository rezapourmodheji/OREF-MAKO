from pathlib import Path
from .adams_runner import AdamsRunnerMixin
from .importGeometries import ImportGeometriesMixin
from .createElements import createElementsMixin
from .sMCL import SMCLMixin
from .distcomp import DistCompMixin
from .updateligs import UpdateLigsMixin
from .preoptimization import PreOptimizationMixin
from .countfiles import CountFilesMixin
from .contact_postprocess import ContactGeometryMixin, ContactIncidentMixin
# from .bones import BonesMixin
# from .markers import MarkersMixin
# from .motions import MotionsMixin
# from .measures import MeasuresMixin
from .simulation import SimTestMixin
from .processmotion import ProcessMotionMixin
# from .extforces import ExternalForcesMixin
# from .visualization import VisualizationMixin


class AdamsKnee(AdamsRunnerMixin,
                ImportGeometriesMixin,
                createElementsMixin,
                SimTestMixin,
                SMCLMixin,
                DistCompMixin,
                UpdateLigsMixin,
                PreOptimizationMixin,
                CountFilesMixin,
                ProcessMotionMixin,
                ContactGeometryMixin,
                ContactIncidentMixin,
                ):

    def __init__(self,
                subject, case, side, coltension,
                study_root,
                adams_mdi = r'C:/Program Files/MSC.Software/Adams/2021_4_856550/common/mdi.bat',
                ):
        self.subject = subject
        self.case = case
        self.side = side
        self.coltension = coltension
        self.study_root = Path(study_root)
        self.adams_mdi = Path(adams_mdi)
        

        # Derived paths
        self.data_raw_dir       = self.study_root / "Data_Raw"
        self.data_reduced_dir   = self.study_root / "Data_Reduced"
        self.subject_dir        = self.study_root / "Data_Reduced" / "Subjects" / subject / case / coltension
        self.cmd_dir            = self.subject_dir / "Macros and CMD"
        self.geom_dir           = self.subject_dir / "model_inputs" / "Geometries"
        self.model_inputs_dir   = self.subject_dir / "model_inputs"
        self.bin_dir            = self.subject_dir / "bin_models"
        self.transforms_dir     = self.subject_dir / "model_inputs" / "Transformations"
        self.output_dir         = self.subject_dir / "model_outputs" 
        self.lig_update_dir     = self.subject_dir / "model_inputs" / "Lig_update"
        self.contact_dir        = self.subject_dir / "contact_incidents"