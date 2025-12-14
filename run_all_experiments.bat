@echo off
set PYTHON=python

cd /d C:\Users\gtren\MLQCD-SV

echo Running K_ll_to_qsq0 ...
%PYTHON% lattice_qcd_analysis.py K_ll_to_qsq0

echo Running K_ll_to_2qsqmaxby3 ...
%PYTHON% lattice_qcd_analysis.py K_ll_to_2qsqmaxby3

echo Running localscalar_T16_to_qsq0 ...
%PYTHON% lattice_qcd_analysis.py localscalar_T16_to_qsq0

echo Running localscalar_T19_to_qsqmaxby3 ...
%PYTHON% lattice_qcd_analysis.py localscalar_T19_to_qsqmaxby3

echo Running localscalar_T22_to_2qsqmaxby3 ...
%PYTHON% lattice_qcd_analysis.py localscalar_T22_to_2qsqmaxby3

echo Running localtempvector_T16_to_qsq0 ...
%PYTHON% lattice_qcd_analysis.py localtempvector_T16_to_qsq0

echo Running localtempvector_T22_to_2qsqmaxby3 ...
%PYTHON% lattice_qcd_analysis.py localtempvector_T22_to_2qsqmaxby3

echo --------------------------------------
echo ALL EXPERIMENTS COMPLETED
echo --------------------------------------
pause
