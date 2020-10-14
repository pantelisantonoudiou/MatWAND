# Inputs
MatWAND was originally designed to process data saved from labchart into matlab and adibin files.

Therefore the inputs need to be structured in one of two following formats in order to be processed by MatWAND.

# a) .mat files containing the raw traces
Input structure is a .mat file that contains:

    i) A vector called `data` containing the raw voltage trace 

    **For multiple channels `data` = channel 1 + channel 2 + channel 3**

    ii) A variable called `samplerate` containing the sampling rate per second 

    iii) A vector called `datastart` containing the start time(s) for each channel (in samples)

    iv)  A vector called `dataend` containing the end time(s) for each channel (in samples)

    -Optional-

    v) For automatic sepration of files based on conditions please include:

    i) comments in a Rectangular Character Array called `comtext`

    ii) and their associated times in a vector called `com` (time in samples)

You can find an example **[<< Here](/examples)**


# b) .binary files
The binary files contain only the data in raw voltage trace. Any of the following file formats can be used (supported by matlab memmap function https://www.mathworks.com/help/matlab/ref/memmapfile.html)

    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'single', 'double'


**[<< Back to Main Page](/README.md)**
