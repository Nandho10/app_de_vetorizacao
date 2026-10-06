for mod in ['cv2', 'shapely', 'skimage', 'geopandas', 'scipy', 'osgeo']:
    try:
        __import__(mod)
        print(f"{mod}: available")
    except Exception as e:
        print(f"{mod}: NOT available ({e})")

import cv2
print("cv2.ximgproc:", hasattr(cv2, 'ximgproc'))
print("cv2.createLineSegmentDetector:", hasattr(cv2, 'createLineSegmentDetector'))
