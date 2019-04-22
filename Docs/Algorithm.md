MatWAND data processing:

1) Load Folder Directory (matlab or binary files).

2) Convert voltage traces to power spectral density using the fft transform (https://www.mathworks.com/help/matlab/ref/fft.html) and store
to raw_psd_user folder.

+ optional file separation for .mat files.

3) Pre-process the analyzed PSD data  and store to "processed_psd" folder.

4) Plot data from the "processed_psd" folder.

