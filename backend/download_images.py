from fastbook import *

path = Path("plants")
plants_types = 'aloe vera', 'Aptenia cordifolia', 'spathiphyllum', 'epipremnum aureum', 'sansevieria trifasciata'

if path.exists():
    for o in plants_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f'{o} plant')
        download_images(dest, urls=results)