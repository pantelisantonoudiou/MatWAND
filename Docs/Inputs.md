# Inputs
MatWAND was originally designed to process data from labchart.

The inputs need to be structured in one of two formats in order to be processed by MatWAND

# a) .mat files containing the raw traces
Input structure is a .mat file that contains:

i) A vector called `data` containing the raw voltage trace 

For multiple channels data = channel1 + channel2 + chanel3

ii) A variable called `samplerate` containing the sampling rate per second 

iii) A vector called `datastart` containing the start time(s) for each channel (in samples)

iv)  A vector called `dataend` containing the end time(s) for each channel (in samples)

-Optional-

v) For automatic sepration of files include comments in a vector



# b) .binary files

Any of the following file formats might be used (supported by matlab memmap function https://www.mathworks.com/help/matlab/ref/memmapfile.html)

'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'single', 'double'

In case single/double files are not used a normalisation number is needed to convert the files back to their original value.
e.g. int16 files from labchart need to be divided by 320000


**[<< Back to Main Page](/README.md)**
