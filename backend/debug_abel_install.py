import os
import abel
import sys

print(f"Abel path: {abel.__path__}")
if hasattr(abel, '__file__'):
    print(f"Abel file: {abel.__file__}")
else:
    print("Abel has no __file__ attribute (Namespace package?)")

# Get the directory from path
if len(abel.__path__) > 0:
    dir_path = list(abel.__path__)[0]
    print(f"Listing {dir_path}:")
    try:
        files = os.listdir(dir_path)
        print(files)
        
        if '__init__.py' in files:
            print("\nContent of __init__.py:")
            with open(os.path.join(dir_path, '__init__.py'), 'r') as f:
                print(f.read()[:500]) # Read first 500 chars
        else:
            print("\n__init__.py NOT FOUND")
    except Exception as e:
        print(f"Error listing dir: {e}")
