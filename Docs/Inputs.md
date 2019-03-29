# Inputs
MatWAND was originally designed to process data from labchart.

The inputs need to be structured in one of two formats in order to be processed by MatWAND

a) .mat files containing the raw traces
Input structure is data file contains

b) .binary files

Any of the following file formats might be used (supported by matlab memmap function https://www.mathworks.com/help/matlab/ref/memmapfile.html)

'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'single', 'double'

In case single/double files are not used a normalisation number is needed to convert the files back to their original value.
e.g. int16 files from labchart need to be divided by 320000


**[<< Back to Main Page](/README.md)**
