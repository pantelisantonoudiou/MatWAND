# Short-time Fourier Transform (STFT)

- The algorithm used in MatWAND for spectral analysis is a custom implmentation of the Short-time Fourier Transform (STFT).
- The voltage signal is analyzed using a sliding window of fixed length (default = 5 seconds) similar to 
[this illustration](https://www.mathworks.com/help/signal/ref/iscola_stft.png).
- The sliding window overlap is set at 50%.
- For each segment a windowing function is applied (default = [hann](https://www.mathworks.com/help/signal/ref/hann.html) window).
- These properties can be adjusted in MatWAND [parameters menu](/Images/tutorial/input_parameters_gui.png) after analysis 
[initiation](/Docs/Step-by-Step.md/#2-choose-file-and-analysis-parameters).
- Each segment is converted to the frequency domain by applying the [FFT](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html).
- The frequency range can be set during [MatWAND analysis](/Docs/Step-by-Step.md/#4-initiate-spectral-analysis) (default range = 0.4 - 200 Hz).

**[<< Back to Main Page](/README.md)**
