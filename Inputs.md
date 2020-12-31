# Inputs

MatWAND was originally designed to process data saved from labchart into matlab files.

* The inputs have to be structured in one of two following formats in order to be processed by MatWAND.

---

## 1. mat files containing the raw traces

- Input structure is a ***.mat*** file that contains:

        data: vector containing the raw voltage trace for all channels

        samplerate: numerical variable/vector containing the sampling rate per second for each channel

        datastart: numerical variable/vector containing the start time(s) for each channel (in samples)

        dataend: numerical variable/vector containing the end time(s) for each channel (in samples)

        -Optional-

        For automatic separation of files based on conditions please include:

        i) comments in a Rectangular Character Array called `comtext`

        ii) and their associated times in a vector called `com` (time in samples)

- Data should be structured so that channels follow sequentially as can be seen in the example below.

![Banner](/Images/data_format_matlab.png)

- You can find an example .mat file **[<< Here](/examples)**
- Extensive information about LabChart to MATLAB export can be found on the **[ADINSTRUMENTS](https://www.adinstruments.com/support/knowledge-base/how-does-matlab-open-exported-data)** website.

---

## 2. binary files

- The binary files should only contain LFP/EEG voltage data.
- Any of the following file formats can be used as supported by matlab [memmap](https://www.mathworks.com/help/matlab/ref/memmapfile.html) function.

        'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'single', 'double'

- Data from multiple channels should be interleaved into one signal as illustrated below.

![Banner](/Images/data_format_binary.png)

---

**[<< Back to Main Page](/index.md)**
