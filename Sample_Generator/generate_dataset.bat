
call get_simulation_name.bat

set sample_count=100000

set step_sample_count=100

:loop
python dataset_generation.py %step_sample_count%

@echo off
set cnt=0
for %%A in (%name%_dataset/*) do set /a cnt+=1
set /a cnt/=5
echo File count = %cnt%
@echo on

if %cnt% LSS %sample_count% goto loop
