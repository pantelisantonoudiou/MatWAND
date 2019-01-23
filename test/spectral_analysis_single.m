classdef spectral_analysis_single < handle
    %spectral analysis for a single experiment
    % x = spectral_analysis_single
    % load_labchart_mat(x)
    % or
    % load_labchart_bin(x)
    %
    % down_sample (x)
    % plot_raw_volt(x,x.input_wave,[0 0 0],1)
    % psd_only(x)
    % or load_p_matrix(x)
    % plot_PSD(x,1,20,0)
    % obtain_PSD_params(x,5,20)
    % plot_PSD_params(x,'peak_power')
    % remove_outliers(x,5)
    % remove_outliers_psd(x)
    % plot_PSD(x,5,20,0)
    % plot_power_ratio(x,5,10,2)
    % filtered_signal=x.BandPassFilter(x.input_wave,x.Fs,1,150);
    
    properties
        %wave properties
        input_wave
        dur  = 5  % Window duration of the fft transform (in seconds)
        Fs %sampling rate
        %loading properties
        %paths
        load_path = 'CC:\Users\panton01\Desktop\INvivo_data\CUS\analysis\raw_psd_BLA';
        desktop_path;
        exp_id %experiment id
        
        %analysis properties
        freq % frequency vector for psd
        LowFCut = 0.4; %low frequency boundary
        HighFCut = 200; %high frequency boundary
        winsize;
        F1 % lower freq of extracted raw psd
        F2 % higher freq of extracted raw psd
        analysed_range
        norm_type
        
        %extracted parameters from psd
        power_matrix
        power_area
        peak_freq
        peak_power
        
        %outliers
        outl_vector
        timex
    end
    
    methods(Static)
%         % class constructor
%         function obj = spectral_analysis_single()
%             % get path to dekstop
%             obj.desktop_path = cd;
%             obj.desktop_path = obj.desktop_path(1:strfind( obj.desktop_path,'Documents')-1);
%             obj.desktop_path = fullfile( obj.desktop_path,'Desktop\');
%                         
%         end
        
        %set font and axis properties
        function prettify_o(axhand)
            % prettify_o(obj,line_width,font_size)
            %figure handle
            set(gcf,'color','w');
            
            %axis handle
            axhand.Box = 'off';
            axhand.TickDir = 'out';
            axhand.XColor = 'k';
            axhand.YColor = 'k';
            axhand.FontName = 'Arial';
            axhand.FontSize = 14;
            axhand.FontWeight = 'bold';
            axhand.LineWidth = 1;
            
        end
        
        %autoscale time
        function [units,factor] = autoscale_x(x,samp_rate)
            %autoscale_x - Autoset time Scale - must be in seconds
            factor = 1;
            
            if x>(3*24*60*60*samp_rate)               %more than 72 hrs shown in days
                factor = 1/(24*60*60);
                units = 'days';
            elseif x>(3*60*60*samp_rate)              %more than 180 min shown in hours
                factor = 1/(60*60);
                units = 'hrs';
            elseif x>(3*60*samp_rate)                 %more than 180 sec shown in min
                factor = 1/60;
                units = 'min';
            elseif x<3*samp_rate                      %less than 3 sec shown in msec
                factor = 1000;
                units = 'msec';             %There's no time format for msec
            else
                units = 'sec';
            end
        end
        
        %get frequncy index based on sampling rate
        function Freal = getfreq(Fs,dur,FreqX)
            %Freal = getfreq(Fs,dur,FreqX)
            % give frequency value scaled according to sampling rate Fs
            % and window size for calculate fft winsize
            winsize=round(Fs*dur); %5seconds
            Freal=FreqX*(winsize/Fs); Freal=ceil(Freal)+1;
        end
        
        %filter list
        function filtered_list = filter_list(xstruct,xstring)
            cntr=1;
            for i = 1:length(xstruct)
                if isempty(strfind(xstruct(i).name,xstring))==0
                    filtered_list{cntr}=xstruct(i).name;
                    cntr=cntr+1;
                end
            end
            filtered_list=filtered_list';
        end
        
        %filter wave
        function filtered_signal = BandPassFilter(original_signal,Fs,F_low,F_high)
            if F_high>=1000
                disp('error!!! F_high is higher than downsampling rate')
                filtered_signal=nan;
                return
            end
            
            % Sampling Frequency (Hz)
            Fn = Fs/2;                                                  % Nyquist Frequency (Hz)
            Wp = [F_low   F_high]/Fn;                                         % Passband Frequency (Normalised)
            Ws = [(F_low-0.5)   (F_high+0.5)]/Fn;                                         % Stopband Frequency (Normalised)
            Rp =   1;                                                   % Passband Ripple (dB)
            Rs = 150;                                                   % Stopband Ripple (dB)
            [n,Ws] = cheb2ord(Wp,Ws,Rp,Rs);                             % Filter Order
            [z,p,k] = cheby2(n,Rs,Ws);                                  % Filter Design
            [sosbp,gbp] = zp2sos(z,p,k);                                % Convert To Second-Order-Section For Stability
            
            filtered_signal = filtfilt(sosbp, gbp, original_signal);
            
            % % freqz(sosbp, 2^16, Fs)  % Filter Bode Plot
            % fvtool(sosbp,1,'Fs',Fs)
            % figure
            % plot(original_signal);hold on; plot(filtered_signal,'linewidth',2)
        end
        
        %extract power area, peak power and peak freq
        function [Peak,peak_freqx,p_area] = psd_parameters(psd_raw,freqx) % ,peak_width
            %%[Peak,peak_freqx,p_area]= psd_parameters(psd_raw,freqx)
            %input parameters are PSD and frequency vectors
            
            %smooth psd curve (smooth factor of 5)
            smth_psd = smooth(psd_raw);
            
            %find peak power and index at which the peak power occurs
            [Peak, x_index] = max(smth_psd);
            peak_freqx = freqx(x_index);
            
            %get power area
            p_area = sum(smth_psd);
            
        end      
                
    end
    
    methods % dynamic methods
        
        function obj = spectral_analysis_single()
            % get path to dekstop
            obj.desktop_path = cd;
            obj.desktop_path = obj.desktop_path(1:strfind( obj.desktop_path,'Documents')-1);
            obj.desktop_path = fullfile( obj.desktop_path,'Desktop\');              
        end
        
        %sine wave generator
        function obj = sine_gen(obj,freq_sin,mins)
            %[input_wave, Fs]= sine_gen(freq_sin,Fs_in,mins)
            %inputs freq = frequency of sine wave
            %mins = length of generated wave in mins
            
            obj.Fs = 1000; %sampling frequency
            T = 1/obj.Fs;             % Sampling period
            L = obj.Fs*60*mins;             % Length of signal
            t = (0:T:(L-1)/obj.Fs);        % Time vector
            
            obj.input_wave = sin(2*pi*freq_sin*t); %+ 2*rand(1,L);
            
        end
        
        %load matlab data from labchart
        function obj = load_labchart_mat(obj)
            % obj = load_labchart(obj)
            obj.exp_id=uigetfile(obj.load_path);
            if obj.exp_id ==0 %proceed only if a selection was made
                return
            end
            Full_path=[obj.load_path obj.exp_id];
            load(Full_path,'data','samplerate','datastart','dataend','blocktimes','com','comtext')
            obj.input_wave = data;
            obj.Fs =samplerate(1);
        end
        
        % multi channel lab chart
        
        % load binary data from labchart
        function obj = load_labchart_bin(obj)
            %obj = load_labchart_bin(obj)
            %Load int16 binary data from labchart
            %Selected_Channel=1 BLA, 2 PFC/other brain region, 3 EMG
            %data_start=4000*900; %data_start must be 0 or a multiple of 10
            
            % USER INPUT%%%
            prompt = {'Analyse entire file (yes/no) ?','Start time (mins):','Analysed period (mins):','Enter Channel (BLA, PFC, EMG):'};
            Prompt_title = 'Input';
            dims = [1 35];
            definput = {'no','0','10','BLA'};
            answer = inputdlg(prompt,Prompt_title,dims,definput);
                              
            N_channels = 3; %total number of channels stored in labchart binary file
            
            % get file to load
            obj.exp_id=uigetfile([obj.load_path '*.adibin']);
            
            if obj.exp_id ==0 %proceed only if a selection was made
                return
            end
            % get full file path
            Full_path=[obj.load_path obj.exp_id];
            
            % sampling rate
            obj.Fs = 4000;

            % channel selection
            switch answer{4}
                case 'BLA'
                    Selected_Channel = 1;
                case 'PFC'
                    Selected_Channel = 2;
                case 'EMG'
                    Selected_Channel = 3;
            end
            
            
            % map the whole file to memory
            m = memmapfile(Full_path,'Format','int16'); %'single'
            
            %convert to lower case
            file_analysis = lower(answer{1});
            switch file_analysis
                case 'yes'
                    obj.input_wave = double(m.Data(Selected_Channel:N_channels:length(m.data)-Selected_Channel));
                case 'no'
                    data_start = str2double(answer{2})*obj.Fs*60;
                    data_end = str2double(answer{3})*obj.Fs*60 + (data_start);
                     %get part of the channel
                    obj.input_wave = double(m.Data(data_start*N_channels+Selected_Channel:N_channels:data_end*N_channels));
            end
            
           
            %clear  memmap object
            clear m; %;
            
            %set correct units to Volts
            obj.input_wave = obj.input_wave/320000;
            
        end
                
        % down sample to 1000 Hz
        function obj = down_sample (obj)
            %[ input_wave, Fs] = down_sample (obj)
            %obj = down_sample (obj)
            down_factor = obj.Fs/1000; %downsmaple to 1000 Hz
            obj.Fs = obj.Fs/down_factor;
            obj.input_wave = downsample(obj.input_wave,down_factor);
            
        end
        
        % plot individual trace
        function [handle1,axis1] = plot_raw_volt(obj,y,trace_col,line_width)
            % volt_trace_handle = plot_raw_volt(obj)
            
            %create time
            t = (0:1:length(y)-1)/obj.Fs;
            
             % use appropriate time scaling
            [time_units,time_factor] = obj.autoscale_x(length(t),obj.Fs);
            t = t*time_factor;
            
            %plot trace, get handles and add labels
            handle1 = plot(t,y,'color',trace_col,'LineWidth',line_width);
            axis1 = gca;
            xlabel(['Time (' time_units ')'])
            ylabel('Volts (V)')
            
            %modify plot properties
            obj.prettify_o(axis1)
        end
        
        % load analysed power matrix
        function obj = load_p_matrix(obj)
            % get power matrix file
            [obj.exp_id,obj.load_path] = uigetfile(obj.load_path);
            
            if obj.exp_id ==0 % check for file load
                return
            end
            
            % load power_matrix
            s = load(fullfile(obj.load_path, obj.exp_id),'power_matrix');
            obj.power_matrix = s.power_matrix;
            
            % get object
            idx = strfind(obj.load_path,'analysis');   
            obj_path = obj.load_path(1:idx-1);
            obj_path = fullfile(obj_path,'analysis','psd_object');
            load (obj_path,'psd_object')
            
            % load object properties
            obj.Fs = psd_object.Fs;
            obj.F1 = psd_object.F1;
            obj.F2 = psd_object.F2;
            obj.winsize = psd_object.winsize;
            obj.freq = eval(psd_object.freq_cmd);
        end
        
        % fft analysis with hanning window
        function obj = psd_only(obj)
            tic
            %[power_matrix,PowerArea,PeakPower,PeakFreq]= psd_only(obj)
            %outputs = power_matrix,PowerArea,PeakPower,PeakFreq
            %inputs = input_wave,dur,Fs,LowFCut,HighFCut
            
            % Spectrogram settings:
            obj.winsize = round(obj.Fs*obj.dur);
            overlap = round(obj.winsize/2);
            
            % create hanning window
            winvec = hann(obj.winsize);
            
            %ensure that input matches vector format of hann window
            if isrow(obj.input_wave) ==1
                winvec=winvec';
            end
            %length of wave
            Channel_length=length(obj.input_wave)-(rem(length(obj.input_wave),obj.winsize));
            % fprintf('%d samples were discarded to produce equal size windows\n',(rem(length(obj.input_wave),winsize)))
            
            %get index values of frequencies
            obj.F1 = obj.getfreq(obj.Fs,obj.dur,obj.LowFCut); %lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.dur,obj.HighFCut);%upper boundary
            %             %get index values of frequencies
            %             F1=obj.LowFCut*(winsize/obj.Fs); F1=ceil(F1)+1;%lower boundary
            %             F2=obj.HighFCut*(winsize/obj.Fs); F2=ceil(F2)+1;%upper boundary
            
            
            %create a frequency vector
            obj.freq = 0:obj.Fs/obj.winsize:obj.Fs/2;
            
            %preallocate waves
            obj.power_matrix=zeros(obj.F2-obj.F1+1,(Channel_length/overlap)-2);
            
            %removes dc component
            obj.input_wave=obj.input_wave-mean(obj.input_wave);
            
            %initialise counter
            Cntr=1;
            for i=1:overlap:Channel_length -(overlap) %loop through signal segments with overlap
                %%get segment
                signal=obj.input_wave(i:i+objwinsize-1);
                %%multiply the fft by hanning window
                signal=signal.*winvec; %check if the signal has to inverted
                %%get normalised power spectral density
                xdft = (abs(fft(signal)).^2);
                xdft=2*xdft*(1/(obj.Fs*length(signal)));
                %%2 .* to conserve energy across
                psdx = xdft((1:length(xdft)/2+1));
                
                %save power spectral density over time
                obj.power_matrix(:,Cntr)=psdx(obj.F1:obj.F2);
                
                %increase counter
                Cntr = Cntr+1;
            end
            obj.timex = toc;
        end
        
        % plot power area, peak freq and peak power
        function obj = obtain_PSD_params(obj,Flow,Fhigh)
            % plot_PSD_params(obj,Flow,Fhigh)
            %plot power spectral density parameters withing desired
            %frequencies Flow to Fhigh
            %plot power area, peak freq and peak power
            [~,L] = size(obj.power_matrix);
            %pre-allocate vectors
            obj.peak_power = zeros(1,L);
            obj.peak_freq = zeros(1,L);
            obj.power_area = zeros(1,L);
            
            %frequency parameters
            freq_range(1) = obj.getfreq(obj.Fs,obj.dur,Flow)- obj.F1+1; %new low boundary
            freq_range(2) = obj.getfreq(obj.Fs,obj.dur,Fhigh) -obj.F1+1; %new high boundary
            freqx = obj.freq(freq_range(1):freq_range(2));
            
            for i = 1:L %get psd_parameters
                [peak,peak_freqx,p_area]= obj.psd_parameters(obj.power_matrix(freq_range(1):freq_range(2),i),freqx);
                obj.peak_power(i) = peak;
                obj.peak_freq(i) = peak_freqx;
                obj.power_area(i) = p_area;
            end
            
        end
        
        % remove outliers that are 5 x bigger than median of peak_power
        function obj = remove_outliers(obj,median_mult)
            % remove_outliers(obj,median_mult)
            % Returns the the outlier free signal (outfree_signal) where outliers are replaced by NaNs
            % index_vec indicates by 1 which points are outliers
            
            %get threshold
            threshold = median_mult*median(obj.peak_power);
            
            %find outliers
            obj.outl_vector = abs(obj.peak_power)> threshold;
            
            %replace with nan if value exceeds threshold according to index array
            obj.peak_power(obj.outl_vector)= NaN;
            obj.power_area(obj.outl_vector)= NaN;
            obj.peak_freq(obj.outl_vector)= NaN;
        end
        
        % replace values of the matrixA with the median when idx_vec is 1
        function obj = remove_outliers_psd(obj)
            %out_Mat = logical_idx_mat(matrixA,idx_vec)
            %replace values of the matrixA with median
            
            for i = 1:length(obj.outl_vector)
                if obj.outl_vector(i) == 1
                    obj.power_matrix(:,i) =  NaN;
                end
            end
            
            %remove nans
            temp_mat_out_free =  obj.power_matrix;
            temp_mat_out_free(isnan(temp_mat_out_free))=[];
            
            %replace nans with the median
            obj.power_matrix(isnan(obj.power_matrix)) = median(temp_mat_out_free(:));
            
        end
        
        
        % plot power spectral density with SEM
        function plot_PSD(obj,Flow,Fhigh,rawPSD)
            %plot_PSD_single(obj,Flow,Fhigh)
            %plot power spectral density within desired frequencies
            %from Flow to Fhigh
            %plot colours
            col_sem = [0.5 0.5 0.5]; col_mean = [0 0 0];
            
            %get freq index of new frequency values
            Flow = obj.getfreq(obj.Fs,obj.dur,Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.dur,Fhigh);
            
            %get frequency parameters
            freqx_bound = obj.freq(Flow:Fhigh);
            
            %get new freq index
            Flow = Flow-obj.F1+1; %new low boundary
            Fhigh = Fhigh-obj.F1+1; %new high boundary
            
            %create figure
            figure('Position',[100 100 816 804])
            
            if rawPSD ==1
                plot(freqx_bound,obj.power_matrix(Flow:Fhigh,:),'color', [0.5 0.5 0.5]);
                axis1 = gca;
                xlabel('Freq. (Hz)')
                ylabel ('Power (V^2 Hz^{-1})')
                % prettify and add title
                title('Power Spectral Density')
                obj.prettify_o(axis1)
                return
            end
            
            %plot_PSD_single(power_matrix,Flow,Fhigh)
            [~, nperiods] = size(obj.power_matrix);
            %get mean and sem
            mean_wave = mean(obj.power_matrix(Flow:Fhigh,:),2)';
            sem_wave = std(obj.power_matrix(Flow:Fhigh,:),0,2)'/sqrt(nperiods);
            mean_wave_plus = mean_wave+sem_wave;
            mean_wave_minus = mean_wave-sem_wave;
                      
            
            %plot mean and shaded sem
            Xfill= horzcat(freqx_bound, fliplr(freqx_bound));   %#create continuous x value array for plotting
            Yfill= horzcat(mean_wave_plus, fliplr(mean_wave_minus));
            fill(Xfill,Yfill,col_sem,'LineStyle','none');hold on;
            plot(freqx_bound,mean_wave,'color', col_mean,'Linewidth',1.5);
            axis1 = gca;
            xlabel('Freq. (Hz)')
            ylabel ('Power (V^2 Hz^{-1})')
            % prettify and add title
            title('Power Spectral Density')
            obj.prettify_o(axis1)
        end
        
        % plot PSD parameters
        function plot_PSD_params(obj,plot_type)
            % plot_PSD_params(obj,'peak_power') 'peak_power' 'peak_freq' 'power_area'
            %get time
            t = (0:1:length(obj.peak_power)-1)*(obj.dur/2);
            
            % use appropriate time scaling
            [time_units,time_factor] = obj.autoscale_x(length(obj.peak_power)*(obj.dur/2)*obj.Fs,obj.Fs);
            t = t*time_factor;
            
            %select plot type
            switch plot_type
                case 'peak_power'
                    %plot peak power
                    figure('Position',[100 100 816 804])
                    plot(t,obj.peak_power,'k')
                    axis_pkpower = gca;  %get axis handle
                    
                    %axis labels and plot formatting
                    xlabel(['Time (' time_units ')'])
                    ylabel('Peak Power (V^2 Hz^{-1})')
                    obj.prettify_o(axis_pkpower)
                    
                case 'peak_freq'
                    %plot peak freq
                    figure('Position',[100 100 816 804])
                    plot(t,obj.peak_freq,'k')
                    axis_pkpower = gca;  %get axis handle
                    
                    %axis labels and plot formatting
                    xlabel(['Time (' time_units ')'])
                    ylabel('Peak Freq. (Hz)')
                    obj.prettify_o(axis_pkpower)
                case 'power_area'
                    %plot peak power
                    figure('Position',[100 100 816 804])
                    plot(t,obj.power_area,'k')
                    axis_pkpower = gca;  %get axis handle
                    
                    %axis labels and plot formatting
                    xlabel(['Time (' time_units ')'])
                    ylabel('Power Area (V^2)')
                    obj.prettify_o(axis_pkpower)
            end
            
        end
        
        % plot power ratio
        function plot_power_ratio(obj,Flow,Fhigh,Fwidth)
            %create time vector
            t = (0:1:length(obj.peak_power)-1)*(obj.dur/2);
            % use appropriate time scaling
            [time_units,time_factor] = obj.autoscale_x(length(obj.peak_power)*(obj.dur/2)*obj.Fs,obj.Fs);
            t = t*time_factor;
            
            %get power matrix length
            [~,L] = size(obj.power_matrix);
           
            %frequency parameters
            
            %power 1
            freq_range(1) = obj.getfreq(obj.Fs,obj.dur,Flow - Fwidth +0.4)- obj.F1+1; %new low boundary - power1
            freq_range(2) = obj.getfreq(obj.Fs,obj.dur,Flow + Fwidth +0.4) -obj.F1+1; %new high boundary - power1
            freqx = obj.freq(freq_range(1):freq_range(2));
            
            %power2
            freq_range(3) = obj.getfreq(obj.Fs,obj.dur,Fhigh - Fwidth +0.4)- obj.F1+1; %new low boundary - power1
            freq_range(4) = obj.getfreq(obj.Fs,obj.dur,Fhigh + Fwidth +0.4) -obj.F1+1; %new high boundary - power1
            freqx2 = obj.freq(freq_range(3):freq_range(4));
            
            %preallocate vectors
            p_area1 = zeros(1,L);
            p_area2 = zeros(1,L);
            
            for i = 1:L %get psd_parameters
                [~,~,p_a1]= obj.psd_parameters(obj.power_matrix(freq_range(1):freq_range(2),i),freqx);
                p_area1(i) = p_a1;
                
                [~,~,p_a2]= obj.psd_parameters(obj.power_matrix(freq_range(3):freq_range(4),i),freqx2);
                p_area2(i) = p_a2;
            end
            
            % get power ratio
            p_ratio = p_area1./p_area2;
            
            %create figure
            figure('Position',[100 100 816 804])
            
            % plot power ratio
            h3 = subplot(3,1,1);
            plot(t,p_ratio,'color', [0.5 0 0.2]);
            ax_power_ratio = gca;  %get axis handle
            obj.prettify_o(ax_power_ratio)
            % axis labels and plot formatting
            set(gca,'XTickLabel',[]);%remove Xaxis tick labels for h1
            ylabel(['Power ratio ' num2str(Flow) ' / ' num2str(Fhigh)  ' Hz'])
            
            % plot power1
            h1 = subplot(3,1,2);
            plot(t,p_area1,'k');
            ax_pkpower1 = gca;  %get axis handle
            obj.prettify_o(ax_pkpower1)
            set(gca,'XTickLabel',[]);%remove Xaxis tick labels for h1
            ylabel(['Power Area ' num2str(Flow)  'Hz (V^2)'])
            
            % plot power2
            h2 = subplot(3,1,3);
            plot(t,p_area2,'k');
            ax_pkpower2 = gca;  %get axis handle
            obj.prettify_o(ax_pkpower2)
            ylabel(['Power Area ' num2str(Fhigh)  'Hz (V^2)'])
            xlabel(['Time (' time_units ')'])
                                          
            
        end
                
    end
    
end


% % extract power matrix from bin file (files per condition separated)
%         function obj = load_labchart_bin2(obj)
% 
%             % USER INPUT%%%
%             prompt = {'Analyse entire file (yes/no) ?','Start time (mins):','Analysed period (mins):'...
%                 ,'Channel Selection','Enter Channel (BLA, PFC, EMG):','Sampling rate:'};
%             Prompt_title = 'Input';
%             dims = [1 35];
%             definput = {'no','0','10','1','BLA;PFC;EMG','4000'};
%             answer = inputdlg(prompt,Prompt_title,dims,definput);
%                               
%           
%             
%             %get file to load
%             obj.exp_id = uigetfile([obj.load_path '*.adibin']);
%             
%             if obj.exp_id ==0 %proceed only if a selection was made
%                 return
%             end
%             %get full file path
%             Full_path=[obj.load_path obj.exp_id];   
%             
%             channel_No = str2double(answer{4}); % selected channel
%             Tchannels  = strsplit(answer{5},';'); % total channels
%             obj.Fs = str2double(answer{6}); % sampling rate
%             % get lfp directory
%             lfp_dir = dir(fullfile(obj.lfp_data_path,'*.adibin'));
%             
%             % get winsize
%             obj.winsize = round(obj.Fs*obj.dur);
%             
%             % get index values of frequencies
%             obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut); %lower boundary
%             obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut);%upper boundary
%             
%             % get epoch in seconds
%             epoch = 60 * 60 * obj.Fs;
%             
%             %initialise progress bar
%             progressbar('Total')
%                            
%             % set data starting point for analysis
%             data_start = 0;
%             
%             %initalise power matrix
%             obj.power_matrix = [];
%                 
%             for i = 1:obj.Nperiods % loop across total number of periods
%                 
%                 % update data end
%                 if i == 1
%                     data_end = data_start + epoch;  
%                 else  % get back winsize
%                     data_end = data_start + epoch + obj.winsize/2;
%                 end
%                 
%                 % map the whole file to memory
%                 m = memmapfile(Full_path,'Format','int16');  
%                 
%                 %get part of the channel
%                 OutputChannel = double(m.Data(data_start*Tchannels+channel_No : Tchannels : data_end*Tchannels));
%                 
%                 % clear  memmap object
%                 clear m;
%                 
%                 %set correct units to Volts
%                 OutputChannel = OutputChannel/320000;
%                 
%                 % obtain power matrix
%                 power_matrix_single = obj.fft_hann(OutputChannel,obj.winsize,obj.F1,obj.F2,obj.Fs);
%                 
%                 % concatenate power matrix
%                 obj.power_matrix  = [obj.power_matrix, power_matrix_single];
%                 
%                 % update data start 
%                 data_start = data_start + epoch - obj.winsize/2;
%                                
%                 % update progress bar
%                 progressbar( i/ (obj.Nperiods))
%             end
%             
%             
%         end