import subprocess
result = subprocess.run(
    [
        r"C:\Program Files\3D Systems\Geomagic Wrap 2021\wrapCORE.exe",
        r"C:\Users\pourmodhejir\Documents\GitHub\OREF-MAKO\remeshCTs.py"
    ],
    capture_output=True,
    text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)