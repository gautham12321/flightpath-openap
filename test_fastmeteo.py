"""Test script to verify fastmeteo API"""
import sys

try:
    import fastmeteo
    print("✓ fastmeteo imported successfully")
    print(f"Location: {fastmeteo.__file__}")
    
    # Check what's in the module
    print("\nAvailable in fastmeteo:")
    items = [x for x in dir(fastmeteo) if not x.startswith('_')]
    for item in items:
        print(f"  - {item}")
    
    # Try to import Grid
    print("\nTrying to import Grid:")
    try:
        from fastmeteo import Grid
        print("✓ Grid class exists")
        print(f"Grid methods: {[m for m in dir(Grid) if not m.startswith('_')]}")
    except ImportError as e:
        print(f"✗ Grid not found: {e}")
    
    # Check for common fastmeteo functions
    print("\nChecking for common functions:")
    for name in ['Grid', 'get', 'interpolate', 'download']:
        if hasattr(fastmeteo, name):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            
except ImportError as e:
    print(f"✗ Cannot import fastmeteo: {e}")
    sys.exit(1)
