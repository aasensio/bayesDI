
MARCS models

1. kurucz_models/ATMOS/MARCS/marcs.py
   Transform MARCS atmospheric models into our format
2. synthesize_marcs.py
   Generate marcs.zarr file with all synthetic spectra
3. gen_star_spots.py
   Generate the artificial surfaces. Adapt the temperatures so that you make sure that the appropriate models exist
4. gen_db.py
   mpiexec -n N python gen_db.py -> it will generate all synthetic spectra for all stars
   diablos is a good machine for this with N=64. Don't forget to copy it to drogon
