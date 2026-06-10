import subprocess
import os
from pathlib import Path
from tabnanny import verbose

class AdamsRunnerMixin:
    verbose = False
    @property
    def name(self) -> str:
        return f"{self.subject}"
    
    def cmd_path(self, name: str) -> Path:
        """Helper: build full path to a .cmd file"""
        return self.cmd_dir / f"{self.name}_{name}.cmd"
    
    def clean_cmd_dir(self):
        """Helper: remove old .cmd, .log, .bat files from cmd_dir"""
        for ext in ["*.cmd", "*.log"]:            
            for f in self.cmd_dir.glob(ext):
                f.unlink(missing_ok=True)

    def run_adams(self, cmd_filename: str | Path, clean_old_aview: bool = True):
        """Execute aview ru-s b <cmdfile>"""
        cmd_path = Path(cmd_filename)

        if clean_old_aview:
            try:
                for f in self.cmd_dir.glob("aview*"):
                    f.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning cleaning aview files: {e}")

        full_cmd = [
            str(self.adams_mdi),
            "aview", "ru-s", "b",
            str(cmd_path.name)   # run from inside cmd_dir
        ]
        if verbose:
            print(f"→ Running Adams for {self.name} : {' '.join(full_cmd)}")
        

        try:
            subprocess.run(
                full_cmd,
                cwd=self.cmd_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"   Adams implementation finished OK for {self.name}")
            if (self.cmd_dir / f"{cmd_path.stem}.log").exists():
                os.unlink(self.cmd_dir / f"{cmd_path.stem}.log")
            os.rename(self.cmd_dir / "aview.log", self.cmd_dir / f"{cmd_path.stem}.log")
        except subprocess.CalledProcessError as err:
            print(f"   !!! Adams ERROR for {self.name}")
            print(err.stderr)

