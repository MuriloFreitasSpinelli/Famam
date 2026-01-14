import sys
import traceback
sys.path.insert(0, r"C:\Users\Radul\Documents\PSE\Famam\src")
try:
    import data.scripts.create_enchanced_dataset as mod
    print('OK - imported:', mod.__file__)
except Exception:
    traceback.print_exc()
