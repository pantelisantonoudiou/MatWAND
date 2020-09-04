classdef spectral_analysis_batch < matlab.mixin.Copyable
    % Class for analyzing LFP/EEG recordings using the fourier transform
    %
    % -Properties- Properties are initiated updated from the application user
    % input
    %
    % -Static Methods-
    % --- Misc
    % A - Array sorting and filtering
    % B - PSD (fft) related
    % C - User related
    % D - Plot related
    %
    % -Dynamic Methods-
    % --- Misc & object load
    % A - FFT analysis
    % B - PSD pre-processing
    % C - Plots
    % D - Export & Statistics
    % E - Comparative Plots and statistics
    %
    %
    % ------ Examples ------
    % %% init
    % x = spectral_analysis_batch
    %
    % %% simulate user app input to get parameters
    % x = get_parameters(x)
    % 
    %
    % %% reload object
    % x = reload_object(x)
    %
    % folder_list = x.obtain_dir_list(x.desktop_path)
    %
    % extract_pmatrix_mat_user(x,5,80), extract_pmatrix_mat(x), extract_pmatrix_bin(x)
    %
    % psd_processing(x)
    %
    % plot_subPSD(x,2,20)
    %
    % psd_prm_time(x,a), aver_psd_prm(x,a)
    % 1- peak power, 2 - peak freq, 3. power area
    %
    %     a.Flow = 2
    %     a.Fhigh = 15
    %     a.par_var = 2
    %     a.norms_v = false
    %     a.ind_v = false
    %     a.mean_v = true
    %
    %     a.band1 = 5
    %     a.band2 = 10
    %     a.band_width = 4
    %     a.par_var = 1
    %     a.norms_v = true
    %     a.ind_v = true
    %     a.mean_v = true
    
    %%% Class parameters %%%
    properties
        
        % User input from application or
        set_array = struct; % structure to store class data
 
        % wave properties
        dur  = 5  % spectral window duration of the fft transform (in seconds)
        Fs = 4000; % sampling rate
        winsize % spectral window duration in samples (depends on sampling rate)
        block_number = 1; % block
        channel_No = 1; % channel to be analyzed
        channel_struct = {'BLA'};
        
        % for multichannel long recordings
        Tchannels = 3; % channels per file
        Nperiods = 12; % in hours
        start_time = 0 % in hours
        period_dur = 60; % in minutes  % sets blocks to be analysed from binary files
        
        % loading paths
        lfp_data_path = 'C:\Users\panton01\Desktop\INvivo_data\Test\raw_data\';% LFP folder path
        desktop_path; % path to desktop
        save_path; % analysed folder path
        raw_psd_user; % raw unseparated psd path
        raw_psd_path % raw psd folder path
        proc_psd_path % processed psd path
        excld_path % excluded psd path
        export_path % path for exported tables and parameters
        
        % freq analysis properties
        freq_cmd = '0:obj.Fs/obj.winsize:obj.Fs/2' % frequency vector for psd
        LowFCut = 0.4; %low frequency boundary
        HighFCut = 200; %high frequency boundary
        F1 % lower freq index of extracted raw psd
        F2 % higher freq index of extracted raw psd
        analysed_range
        
        % fft separation table
        box_or_table= 'box_plot'; % table OR box_plot
        
        % psd processing variables
        norm_var = 'no'; % log(10), log(e), max_val
        linear_var = 'no'; %yes/no
        noise_var = -1; % in Hz
        noisewidth_var = 2; % in Hz
        outlier_var = -1; % median multiple
        bin_size = -1; % new bin for merging (in seconds)
        
        % conditions
        cond_orig_sep_time = []; % initial condition separation (minutes)
        condition_id = []; % condition identifier
        condition_time = []; % condition time duration

        
    end
    
    
    %%% Static methods %%%
    
    methods(Access = public, Static) % MISC %
        
        % get unseparated condition (varied between conditions)
        function copy_user_files(source,dest,file_type)
            %copy_user_files(obj.raw_psd_user,obj.raw_psd_path,'mat')
            % get mat files in load_path directory
            mat_dir = dir(fullfile(source,['*.' file_type]));
            
            for i = 1:length(mat_dir)
                copyfile(fullfile(source,mat_dir(i).name)...
                    ,fullfile(dest,mat_dir(i).name))
            end
        end
        
        % get multiple folders
        function folder_list = obtain_dir_list(starter_path)
            % folder_list = obtain_dir_list(starter_path)
            % get a folder list
            folder_list{1} = uigetdir(starter_path);
            
            cntr = 2;
            while folder_list{cntr-1} ~= 0
                folder_list{cntr} = uigetdir(folder_list{cntr-1},'Choose folder for analysis');
                cntr = cntr +1;
            end
            folder_list(cntr-1) = [];
            %             cellfun(@(folder_list)[folder_list 'GHz'],folder_list,'UniformOutput',false)
        end
     
        % autoscale time
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
        
        % get frequncy index based on sampling rate
        function index_freq = getfreq(Fs,winsize,freqx)
            % index_freq = getfreq(Fs,winsize,freqx)
            % give frequency value scaled according to sampling rate Fs
            % and window size for calculate fft obj.winsize
            index_freq = freqx*(winsize/Fs); index_freq = ceil(index_freq)+1;
        end
        
        % down sample data using decimate
        function down_dec_sample(mat_path,down_rate,ch_name)
            
            % get lfp directory
            lfp_dir = dir(fullfile(mat_path,'*.mat'));
            
            % make new folder to store down_sampled traces
            k = strfind(mat_path,'\');
            downsamp_path = fullfile(mat_path(1:k(end)-1),['resamp_traces_' ch_name ]);
            mkdir(downsamp_path);
            
            % Initialize progress bar
            progressbar('Down sampling ...')
            
            % loop through experiments and perform fft analysis
            for i = 1:length(lfp_dir)
                
                % load file
                load(fullfile(mat_path, lfp_dir(i).name),'data','samplerate','datastart','dataend','com','comtext')
                
                % low pass filter data by 1/down_rate and down sample
                data = decimate(data,down_rate,30,'fir');
                
                % resize samplerate , comments, datastart and dataend
                samplerate = samplerate/down_rate;
                com(:,3) = round(com(:,3)/down_rate);
                datastart = ceil(datastart/down_rate);
                dataend = round(dataend/down_rate);
                
                % save filtered traces in new folder
                save(fullfile(downsamp_path,lfp_dir(i).name),'data','samplerate','datastart','dataend','com','comtext')
                
                % update progress bar
                progressbar(i/length(lfp_dir))
                
            end
        end
        
        % band pass filter signal
        function filtered_signal = BandPassFilter(original_signal,Fs,Flow,Fhigh)
            % filtered_signal = short_lfp.BandPassFilter(original_signal,Fs,Flow,Fhigh)
            % Sampling Frequency (Hz)
            
            Fn = Fs/2;                                                  % Nyquist Frequency (Hz)
            Wp = [Flow   Fhigh]/Fn;                                     % Passband Frequency (Normalised)
            Ws = [(Flow-0.5)   (Fhigh+0.5)]/Fn;                         % Stopband Frequency (Normalised)
            Rp = 1;                                                     % Passband Ripple (dB)
            Rs = 75;                                                    % Stopband Ripple (dB)
            [n,Ws] = cheb2ord(Wp,Ws,Rp,Rs);                             % Filter Order
            [z,p,k] = cheby2(n,Rs,Ws);                                  % Filter Design
            [sosbp,gbp] = zp2sos(z,p,k);                                % Convert To Second-Order-Section For Stability
            
            
            filtered_signal = filtfilt(sosbp, gbp, original_signal);
            
        end
        
    
        
    end
    
    methods(Access = public, Static) % A - Array sorting and filtering %
        
        %%%%%%%%%%% ----- create sorted array - MAIN ------------ %%%%%%%%
        function exp_list = get_exp_array(mat_dir, conditions, paired)
            
            % get exp array
            exp_id = spectral_analysis_batch.cellfromstruct(mat_dir,1);
            
            % get list
            exp_list = spectral_analysis_batch.filter_list(exp_id,conditions);
            
            % get exp IDs
            num_array = spectral_analysis_batch.get_exp_id(exp_list,'_',conditions);
            
            % get unique list
            num_list = unique(num_array); num_list = num_list(~isnan(num_list));
            
            % find empty cells
            emptyCells = cellfun(@isempty,exp_list);
            exp_list(emptyCells) = {'nan'};
            
            % built sorted array
            exp_list = spectral_analysis_batch.sorted_array(exp_list,conditions,num_list);
            
            if paired == 1
               exp_list(any(cellfun(@isempty, exp_list), 2),:) = [];
            end
        end
        %%%%%%%%%%% --------------------------------------------- %%%%%%%%
        
        % extract column from struct
        function new_list = cellfromstruct(struct_mat, col)
            % cell_col = cellfromstruct(struct_mat,col)
            
            %convert structure to cell
            s = struct2cell(struct_mat);
            
            %assign column to new array
            new_list = s(col,:)';
        end
        
        % filter list based on conditions
        function filtered_list = filter_list(raw_list, str_conds)
            % filtered_list = filter_list(raw_list,str_conds)
            % returns a filtered array separated by conditions
            
            % get filterer array
            for ii = 1:length(str_conds)
                % get cell index for matching string
                index = find(contains(raw_list,str_conds{ii}));
                
                % pass to filtered list
                filtered_list(1:length(index),ii) = raw_list(index);
            end
            
            
        end
        
        % get experiment identifier
        function num_array = get_exp_id(raw_list, start_str, condition_list)
            % num_array = sort_array(raw_list,start_str,condition_list)
            % get list size
            [len, wid]= size(raw_list);
            num_array = zeros(len,wid);
            for ii = 1:wid
                
                for i = 1: len
                    % check if condition exists
                    if isempty(raw_list{i,ii}) == 0
                        num_array(i,ii) = str2double(cell2mat(extractBetween(...
                            raw_list{i,ii},start_str,['_' cell2mat(condition_list(ii))])));
                    else
                        
                        num_array(i,ii) = nan;
                    end
                end
                
                % sort index in each column
                %                 [~,idx1(:,ii)] = sort(num_list);
            end
            
        end
        
        % build sorted array
        function sorted_list = sorted_array(filt_array, str_conds, num_list)
            % sorted_list = sorted_array(filt_array,str_conds,num_list)
            % builds a sorted array
            % filt_array = m x n array  where m is experiment number and n is
            % condition number
            % str_conds = conditions
            % num_list unique exp identifier list
            
            % get list length and width
            [~, wid] = size(filt_array); % [len, wid] = size(filt_array);
            len = max(num_list);
            % pre-allocate list
            sorted_list = cell(len,wid);
            
            for ii = 1:wid % loop through conditions
                for i = 1:len % loop across experiments and find match
                    % check if exp is present
                    x = find(contains(filt_array,['_' num2str(i) '_' str_conds{ii}]));
                    % x = find(contains(filt_array,['_' num2str(num_list(i)) '_' str_conds{ii}]));
                    if x~=0
                        sorted_list{i,ii} = filt_array{x};
                    else
                        sorted_list{i,ii} = [];
                    end
                end
            end
            
        end
        
        %%% ----------------- %%%
        
        
        % sort single column experiment list
        function exp_list = sort_rawLFP_list(mat_dir)
            % exp_list = sort_rawLFP_list(mat_dir)
            
            % get exp array
            exp_list = spectral_analysis_batch.cellfromstruct(mat_dir,1);
            
            for i = 1 : length(exp_list)
                k  = strfind(exp_list{i},'_');
                index_list(i) = str2double(exp_list{i}(k(1)+1:k(2)-1));
            end
            
            % sort
            [~,idx] = sort(index_list);
            exp_list = exp_list(idx);
        end
        
        % get unique conditions from cell array
        function unique_conds = isolate_condition(explist, nfromlast)
            % unique_conds = isolate_condition(explist,nfromlast)
            
            % isolate_condition(list,nfromlast)
            for i = 1:length(explist)
               
               % get condition
               k = strfind(explist{i},'_');
               templist{i} = explist{i}(k(end-nfromlast)+1:end);
            end
            
            % get unique conditions
            unique_conds = unique(templist);
            unique_conds = erase(unique_conds,'.mat');
        end
        
        % check if input files are in the correct format
        function file_correct = check_inputs(exp_path, file_ext)
            % file_correct = spectral_analysis_batch.check_inputs(exp_path, file_ext)
            % file_correct = spectral_analysis_batch.check_inputs(exp_path, '.mat')
            
            % get file paths
            mat_dir = dir(fullfile(exp_path, ['*' file_ext]));
            if isempty(mat_dir)
                file_correct = -1;
                return
            end
            
            try % try to get conditions   
                unique_conds = spectral_analysis_batch.isolate_condition({mat_dir.name},0); % get unique conditions
            catch
                file_correct = 0;
                return
            end
            try
                % try to get experiment file list
                exp_list = spectral_analysis_batch.get_exp_array(mat_dir, unique_conds, 1);
                
                if size(exp_list,2) ~= length(unique_conds) % check if exp list has columns equal to unique conditions
                    file_correct = 0;
                    return
                end
                
                if  numel(exp_list) ~= length(mat_dir) % if exp list is not equal to file size
                    file_correct = 0;
                else % if exp list size is bigger than zero
                    file_correct = 1;
                end
                
            catch % if get_exp_array fails to execute
                file_correct = 0;
            end
        end
        
    end
    
    methods(Access = public, Static) % B - PSD related %
        
        %%% Extracting and manipulating the PSD %%%
        % fft analysis with hanning window
        function power_matrix = fft_hann(input_wave,winsize,F1,F2,Fs)
            % power_matrix = fft_hann(input_wave,winsize,F1,F2,Fs)
            % outputs = power_matrix
            % inputs = input_wave,winsize,F1,F2,Fs
            
            % Spectrogram settings:
            overlap = round(winsize/2);
            
            % create hanning window
            winvec = hann(winsize);
            
            % ensure that input matches vector format of hann window
            if isrow(input_wave) == 1
                winvec = winvec';
            end
            
            % get correct channel length for analysis
            channel_length = length(input_wave)-(rem(length(input_wave), overlap));
            input_wave = input_wave(1:channel_length);
            
            % PAD start and end
            input_wave = horzcat(input_wave(1: overlap), input_wave, input_wave(end + 1 - overlap :end));
            
            % preallocate waves
            power_matrix = zeros(F2 - F1+1, round(length(input_wave)/overlap)-2);
            
            % removes dc component
            input_wave = input_wave - mean(input_wave);
            
            % initialise counter
            Cntr = 1;
            for i = 1:overlap:length(input_wave)-winsize % loop through signal segments with overlap

                % get segment
                signal = input_wave(i:i+winsize-1);
                
                % multiply the fft by hanning window
                signal = signal.*winvec;
                
                % get normalised power spectral density
                xdft = (abs(fft(signal)).^2);
                xdft = 2*xdft*(1/(Fs*length(signal)));
                
                % 2 .* to conserve energy across
                psdx = xdft((1:length(xdft)/2+1));
                
                % save power spectral density over time
                power_matrix(:,Cntr)= psdx(F1:F2);
                
                % increase counter
                Cntr = Cntr+1;
            end
            
        end
        
        % remove outliers that are x bigger than median of peak_power
        function [input,out_vector] = remove_outliers(input,median_mult)
            % input = remove_outliers(input,median_mult)
            % Returns the the outlier free signal (outfree_signal) where outliers are replaced by NaNs
            % index_vec indicates by 1 which points are outliers
            
            % get threshold
            threshold_p = median(input) * median_mult;
            threshold_m = median(input) / median_mult;
            
            %find outliers
            out_vector = input > threshold_p | input < threshold_m;
            
            %replace with nan if value exceeds threshold according to index array
            %             input(out_vector)= NaN; %- originally with Nans - NaN;
        end
        
        % noise removal
        function psd_out = remove_noise_psd(psd,Fs,winsize,noise_freq,width_f,F1)
            %psd_noise_remove(band_freq,width_f,psd)
            
            % removing boise boundaries
            Fnoisemin = spectral_analysis_batch.getfreq(Fs,winsize,noise_freq - width_f); 
            Fnoisemin= Fnoisemin + 1-F1;
            Fnoisemax = spectral_analysis_batch.getfreq(Fs,winsize,noise_freq + width_f); 
            Fnoisemax= Fnoisemax + 1-F1;
            
            % replace noise band range with nans
            psd(Fnoisemin:Fnoisemax) = nan;
            
            % replace nans with pchip interpolatipjn
            psd_out = fillmissing(psd,'pchip');
            
        end
        
        % extract power area, peak power and peak freq
        function [Peak,peak_freqx,p_area] = psd_parameters(psd_raw,freqx) % ,peak_width
            %%[Peak,peak_freqx,p_area]= psd_parameters(psd_raw,freqx)
            % input parameters are PSD and frequency vectors
            
            % smooth psd curve (smooth factor of 5)
            smth_psd = smooth_v1(psd_raw,5);
            
            % find peak power and index at which the peak power occurs
            [Peak, x_index] = max(smth_psd);
            peak_freqx = freqx(x_index);
            
            % get power area
            p_area = sum(smth_psd);
            
        end
        
        % linearise PSD
        function y_fit = linearise_fft(x,y)
            % linear_fft=linearise_fft(x,y,upper_bound,ploton)
            % outputs the linearised fft by removing 1/f exp fit
            % y = power spectral density;
            % x = freqx;
            
            % get positive log values
            psd_log = 1./-log10(y);
            
            %%% -exp fit- %%%
            % % % fit an exp line through the logged power spectral density
            exp_fit = fit(x,psd_log,'exp2');
            
            MyCoeffs = coeffvalues(exp_fit);
            y_est = MyCoeffs(1)*exp(MyCoeffs(2)*x) + MyCoeffs(3)*exp(MyCoeffs(4)*x);
            
            % readjust fit to always be below the min psd value
            y_fit  = y_est; %-abs(mean(psd_log - y_est));
            
            %             plot(y_fit); hold on; plot(psd_log)
            %             %linearise psd
            %             linear_fft = psd_log - y_new;
        end
        
        % Normalise power matrix by total power
        function norm_mat = TotalPower_norm(power_matrix)
            % norm_mat = TotalPower_norm(power_matrix)
            
            [W,L] = size(power_matrix);
            norm_mat =  zeros(W,L);
            for i = 1:L %loop through psd bins
                norm_mat(:,i) = power_matrix(:,i)/sum(power_matrix(:,i));
            end
        end
        
        % multiple fft choices and functions --- needs completion
        function choose_psd(choice)
            % welch
            %             tic;[pxx,f] = pwelch(data,5*samplerate,2.5*samplerate,[1:0.2:100],samplerate);toc
        end
    end
    
    methods(Access = public, Static) % C - User related %
        %%% unpacking matlab and adibin data %%%
        
        % unpack labchart matlab data
        function output = unpack_data(data,datastart,dataend,selected_block,channel_n)
            % output = unpack_data(data,datastart,dataend,selected_block,channel_n)
            % separate the data file to channels and blocks for analysis %%
            
            % get channel and block number  -  [total_Channel_N, blocks] = size(datastart);
            [~, blocks] = size(datastart);
            
            %check that the block of data exists
            if selected_block > blocks
                output=nan;
                return
            end
            
            output = data(datastart(channel_n,selected_block): dataend(channel_n,selected_block));
            
        end
        
        %unpack labchart bin data
        function output = BatchLoad(PathName,data_start,Fs,Selected_Channel,Period)
            %data_start=4000*900; %data_start must be 0 or a multiple of 10
            %PathName='C:\Users\panton01\Desktop\LFP data for analysis\Pre_Post_Partum\Virg_1706.adibin';
            
            data_end=data_start+(Fs*Period);
            %Load int16 binary data from labchart
            %Selected_Channel=1 BLA, 2 PFC/other brain region, 3 EMG
            %set variables
            N_channels=3;
            
            %map the whole file to memory
            m = memmapfile(PathName,'Format','int16'); %'single'
            
            %get part of the channel
            output = double(m.Data(data_start*N_channels+Selected_Channel:N_channels:data_end*N_channels));
            %clear  memmap object
            clear m; %;
            
            %set correct units to Volts
            output = output/320000;
        end
        
        %%% -------------------------------------------------------------------------------%%%
        
        %%% USER comments and input for condition separation %%%
        
        % add baseline and drug index on power plot from lab chart LFP
        function [com_time,Txt_com] = add_comments(Fs,input,com,comtext,t,block)
            %[com_time,Txt_com] = add_comments(Fs,input,com,comtext,t,block)
            % add lines according to comments
            hold on;
            counter=0;
            % get normalised comment times in minutes
            % com_time = round(com(:,3)./Fs/60);
            com_time = com(:,3)./Fs;
            %choose only com times in the analyzed block
            com_time = com_time(com(:,2)== block);
            Txt_com = comtext(com(:,2)== block,:);
            %add last point
            com_time(length(com_time)+1)= t(end);
            %Txt_com(length(com_time),:)='baseline                   ';
            C = {'k','b','r','c','y','m'};
            Yval = 1.1* max(input);
            
            for ii=1:length(com_time)
                
                plot([counter+10,com_time(ii)-10],[Yval,Yval],C{ii},'linewidth',2)
                %label
                if(ii == 1)
                    text((com_time(ii)+counter)/2,Yval*1.05, 'baseline'...
                        ,'HorizontalAlignment','center','Color','black','FontSize',14)
                else
                    text((com_time(ii)+counter)/2,Yval*1.05, Txt_com(ii-1,:),...
                        'HorizontalAlignment','center','Color','black','FontSize',14)
                end
                counter = com_time(ii);
                
            end
        end
        
        % get user input to define baseline and drug application
        function output = separate_conds(com_time,list_A)%% USER INPUT%%%
            
            % output = separate_conds(com_time,list_A)
            %             % create input list structure
            %             cond_array = 'input';
            %            for i = 1 :length(com_time)-1
            %               cond_array = strcat(cond_array,';input');
            %            end
            
            % prompt user for input
            prompt = {'Save experiment? (Yes = 1 and No = 0)','Conditions:'};
            Prompt_title = 'Input';
            dims = [1 35];
            definput = {'1',list_A};
            output = inputmod(prompt,Prompt_title,dims,definput);
            
        end
        
        % separate psds base on user input
        function separate_psd(raw_pmat,exp_name,sep_vector,com_time,dur_sec)
            % convert to drug times to power matrix block times
            com_time = com_time/(dur_sec/2);
            com_time = [1 floor(com_time(1:end-1))' com_time(end)];
            
            % do not allow comments to exceed max
            com_time(com_time > size(raw_pmat,2)) = size(raw_pmat,2);
            
            for i = 1: length(com_time)-1
                if strcmp(sep_vector{i}, 'false') == 0
                    power_matrix = raw_pmat(:,com_time(i):com_time(i+1));
                    save([exp_name '_' sep_vector{i}],'power_matrix')
                end
            end
        end
               
        % Custom Plot user properties
        %  get user input & pass into cella struct
        function [answer, canceled,formats,cella]  = input_prompt(varargin)
            % [answer, canceled,formats]  = input_prompt(choice_list)
            % Generates gui user input from inputsdlg function
            % [answer, canceled,formats,cella]  = spectral_analysis_batch.input_prompt() 
%             [answer, canceled,formats,cella]  =
%             spectral_analysis_batch.input_prompt('Flow','Fhigh','par_var','removeNans')
            % 'Flow';'Fhigh';'par_var';'norms_v';'mean_v','ind_v';'cond'
            
            % enter name
            name = 'User Input';
            
            % set options
            options.Resize = 'on';
            options.Interpreter = 'tex';
            options.CancelButton = 'on';
            
            % format types included
            formats = struct('type', {}, 'style', {}, 'items', {}, ...
                'format', {}, 'limits', {}, 'size', {});
            
            % set cell size
            cell_size = [100, 30];
                                 
            % append all lists to an array
            % choice string
            prm_array(:,1) = {'Flow';'Fhigh';'par_var';'norms_v';'mean_v';'ind_v';'cond';'band1';'band2';'removeNans'};
            % prompt
            prm_array(:,2) = {'Low Frequency:';'High Frequency:';'Choose PSD parameter:';'Normalise to baseline?';...
                'Plot Mean?';'Plot Individual?';'Comparison conditions';'Band - 1:';'Band - 2:';'Paired'}; 
            % default answer
            prm_array(:,3) = {2; 80; 3; false; true; true;[1,2]; [3,6]; [6,12]; true};
            % type
            prm_array(:,4) = {'edit';'edit';'list';'check';'check';'check';'edit';'edit';'edit';'check'};
            % format
            prm_array(:,5) = {'float';'float';'';'logical';'logical';'logical';'vector';'vector';'vector';'logical'};
            % style
            prm_array(:,6) = {'';'';'popupmenu';'checkbox';'checkbox';'checkbox';'';'';'';'checkbox'};
            % items
            prm_array(:,7) = {'';'';{'Peak Power', 'Peak Frequency', 'Power Area'};'';...
                '';'';'';'';'';''};
    
            
            % take care of empty lists
            if isempty(varargin) == 0
                for ii = 1 :length(varargin)
                    choice_list(ii) = find(contains(prm_array(:,1),varargin{ii}));
                end
            else
                choice_list= 1:size(prm_array,1);
            end
            
            % loop through choice list and crate inputsdlg format
            for i = 1 : length(choice_list)
                k = choice_list(i);
                % append choices
                prompt{i} = prm_array{k,2};
                defaultanswer{i} = prm_array{k,3};
                formats(i,1).type   = prm_array{k,4};
                formats(i,1).format = prm_array{k,5};
                formats(i,1).style = prm_array{k,6};
                formats(i,1).items = prm_array{k,7};
                formats(i,1).size   = cell_size;
   
            end
            
            % remove empty rows from structure
            % formats = formats(~cellfun(@isempty,{formats.type}));
            
            % obtain answer and canceled var through user input
            [answer, canceled] = inputsdlg(prompt', name, formats, defaultanswer',options);
            
            % put data in structure
            cella=[];
            for i = 1 : length(choice_list)
                k = choice_list(i);
                eval(['cella.' prm_array{k,1} '= answer{i};'])
            end
        
        end
        
    end
    
    methods(Access = public, Static) % D - Plot related
        
        % title,x- ,y- labels for subplot
        function super_labels(title_str,xlabel_str,ylabel_str)
            % super_labels(title_str,xlabel_str,ylabel_str)
            % makes outer labels for subplot
            if isempty(title_str)==0
                axes( 'Position', [0, 0.93, 1, 0.05] ) ;
                text( 0.5, 0, title_str, 'FontSize', 18', 'FontWeight', 'Bold', ...
                    'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
                set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
            end
            if isempty(xlabel_str)==0
                axes( 'Position', [0, 0.01, 1, 0.05] ) ;
                text( 0.5, 0, xlabel_str, 'FontSize', 18', 'FontWeight', 'Bold', ...
                    'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
                set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
            end
            if isempty(ylabel_str)==0
                axes( 'Position', [0, 0, 0.04, 1] ) ;
                h = text(0.9, 0.5, ylabel_str, 'FontSize', 18', 'FontWeight', 'Bold', ...
                    'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
                set(h,'Rotation',90);
                set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
            end
        end       
           
        % set font and axis properties
        function prettify_o(axhand)
            % prettify_o(obj,line_width,font_size)
            %figure handle
            set(gcf,'color','w');
            
            %axis handle
            axhand.Box = 'off';
            axhand.TickDir = 'out';
            axhand.XColor = 'k';
            axhand.YColor = 'k';
            axhand.FontName = 'Calibri';
            axhand.FontSize = 14;
            axhand.FontWeight = 'bold';
            %             axhand.LineWidth = 1;
            
        end
        
        % get color for vectors
        function [col_mean,col_sem] = color_vec(col_idx)
            
            % create colour vectors
            col_mat = [...
                0 0 0;... 
                0,0.45,0.74;... 
                0.85, 0.325, 0.01;... 
                0.4940, 0.1840, 0.5560;...
                0.4706, 0.6706, 0.1882];
            % create shade
            col_sem_mat = [
                0.9 0.9 0.9;... 
                0.87, 0.95, 1;... 
                1, 0.95, 0.85;... 
                0.95, 0.85, 1;... 
                0.85,  0.95, 0.85];...
            
            % never allow color index to exceed length of color vector
            if col_idx > length(col_mat)
                col_idx = rem(col_idx,length(col_mat));
                if col_idx ==0
                    col_idx = 1;
                end
            end
            %return color vectors
            col_mean = col_mat(col_idx,:); col_sem = col_sem_mat(col_idx,:);
        end
        
        % get color for vectors
        function [col_face,col_edge,col_light] = color_vec2(col_idx)
            % create colour vectors
            col_mat = [0 0 0;0 0 1; 1 0 0 ;0.8 0.2 0.9 ; 1 0.6 0.2];
            col_sem_mat = [0.5 0.5 0.5; 0.5 0.5 1; 1 0.5 0.5; 0.8 0.5 0.9; 1 0.8 0.6];
            col_light = [0.8 0.8 0.8; 0.8 0.8 1; 1 0.8 0.8; 1 0.8 1; 1 0.9 0.8];
            %= col_mat; col_sem_mat(col_sem_mat==0)=0.9;
            
            % never allow color index to exceed length of color vector
            if col_idx > length(col_mat)
                col_idx = rem(col_idx,length(col_mat));
                if col_idx ==0
                    col_idx = 1;
                end
            end
            %return color vectors
            col_edge = col_mat(col_idx,:); col_face = col_sem_mat(col_idx,:);
            col_light = col_light(col_idx,:);
        end
        
        % get wave for fill plot
        function [mean_wave, xfill,yfill] = getmatrix_mean(feature_aver,t)
            % [mean_wave, xfill,yfill] = getmatrix_mean(feature_aver,t)
            
            [~,len] = size(feature_aver);
            %get mean line with sem
            mean_wave = nanmean(feature_aver,2)';
            sem_wave = nanstd(feature_aver,0,2)'/sqrt(len);
            mean_wave_plus = mean_wave + sem_wave;
            mean_wave_minus = mean_wave - sem_wave;
            
            %plot mean and shaded sem
            xfill = [t fliplr(t)];   %#create continuous x value array for plotting
            yfill = [mean_wave_plus fliplr(mean_wave_minus)];
        end
        
        % dot plot
        function dot_plot(inarray,conditions,indiv_var,mean_var,col_vec)
            % dot_plot(extr_feature,conditions,indiv_var,mean_var)
            % Inarray = m (conditions) by n (experiment number array
            % conditons = cell with condition labels
            % indiv_var = 1 plot individual;
            % mean_var = 1 plot mean
            
            % get condition and experiment number
            [conds, exps] = size(inarray);
            
            % Remove Nans
            if conds>1
                inarray = inarray(:,all(~isnan(inarray)));
            end
            
            % calculate sem
            sem_pkpower = std(inarray,0,2)/sqrt(exps);
            
            hold on;
            
            % plot individual experiments
            if indiv_var ==1
                plot(inarray,'o-','color',col_vec(1,:),'MarkerFaceColor', col_vec(1,:),'MarkerSize',4);               
            end
            
            % plot mean experiment
            if mean_var == 1
%                 plot(mean(inarray,2),'-','color',col_vec(2,:),'MarkerFaceColor', col_vec(2,:),'Linewidth',1.5)
                errorbar(mean(inarray,2),sem_pkpower,'color',col_vec(2,:),'Linewidth',1.5)
            end
            
            ax1= gca;
            spectral_analysis_batch.prettify_o(ax1)
            
            % set tick number
            x_ticks = 1:conds;
            
            % set x boundaries
            lim_boundary = conds*0.1; % xtickangle(45);
            xlim([x_ticks(1)-lim_boundary x_ticks(end)+ lim_boundary]);
            
            % set x tick labels
            xticks(x_ticks); xticklabels(strrep(conditions,'_',' '))
            
            % set y lim boundaries
            ylim([min(inarray(:))- mean(inarray(:))*0.1...
                max(inarray(:))+ mean(inarray(:))*0.1]);
            
        end
        
        % bar plot
        function bar_plot(inarray,conditions,indiv_var,mean_var)
            % bar_plot(extr_feature,conditions,indiv_var,mean_var)
            % Inarray = m (conditions) by n (experiment number array
            % conditons = cell with condition labels
            % indiv_var = 1 plot individual;
            % mean_var = 1 plot mean
            
            % get condition and experiment number
            [conds, exps] = size(inarray);
            
            % calculate sem
            sem_pkpower = nanstd(inarray,0,2)/sqrt(exps);
            hold on
            if indiv_var ==1
                if conds>1
                    plot(inarray,'o','color',[0.5 0.5 0.5],'MarkerFaceColor', [0.5 0.5 0.5],'MarkerSize',4);
                else
                    plot(ones(1),inarray,'o','color',[0.5 0.5 0.5],'MarkerFaceColor', [0.5 0.5 0.5],'MarkerSize',4);
                end
            end
            
            if mean_var == 1
                bar(nanmean(inarray,2),'FaceColor',  'none', 'EdgeColor', 'k','Linewidth',1.5)
                errorbar(nanmean(inarray,2),sem_pkpower,'.','color','k','Linewidth',1.5)
            end
            
            % format graph
            spectral_analysis_batch.prettify_o(gca)
            
            % set tick number
            x_ticks = 1:conds;
            
            % set x boundaries
            lim_boundary = conds*0.3; % xtickangle(45);
            xlim([x_ticks(1)-lim_boundary x_ticks(end)+ lim_boundary]);
            
            % set x tick labels
            xticks(x_ticks); xticklabels(strrep(conditions,'_',' '))
            
            % set y lim boundaries
            ylim([min(inarray(:))- nanmean(inarray(:))*0.1...
                max(inarray(:))+ nanmean(inarray(:))*0.1]);
            
        end
        
        % box plot
        function box_plot(inarray,conditions)
            % box_plot(extr_feature,conditions,indiv_var,mean_var)
            % Inarray = m (conditions) by n (experiment number array
            % conditons = cell with condition labels
            
            boxplot(inarray','Colors','k','Labels',strrep(conditions,'_',' '))
            ax1= gca;spectral_analysis_batch.prettify_o(ax1)
        end
        
        % plot voltage traces for exploration
        function x_idx = plot_traces(input,Fs,com,comtext,block)
            % x_idx = plot_traces(input,Fs,com,comtext,block)
            % Left click = Selet, Right click = preview zoomed version
            % input = voltage trace
            % Fs = sampling rate pet second
            % com = array containing comment times
            % comtext = text array containing comment strings
            % block = integer with current block
            
            
            % create figure to plot full input
            f1 = figure('units','normalized','position',[0.2 0.1 0.6 0.4]);
            % create figure to plot zoomed version
            f2 = figure('units','normalized','position',[0.15 0.6 0.4 0.3]);
            % create figure to plot zoomed version
            f3 = figure('units','normalized','position',[0.55 0.6 0.3 0.3]);
            
            % down sample rate
            new_Fs = 500;
            down_rate = Fs/new_Fs;
            down_data = decimate(input,down_rate,'fir');
            % down_data = BandPassFilter(down_data,new_Fs,30,80);
            
            freq  = 0:Fs/Fs:Fs/2;
            F1 = ceil(1*(Fs/Fs))+1; F2 = ceil(150*(Fs/Fs))+1;
            
            % create time
            t_or = 0:1/Fs:(length(input)-1)/Fs;
            t = 0:1/(Fs/down_rate):(length(down_data)-1)/(Fs/down_rate);
            t = t/60; % convert to minutes
            
            % plot full input figure
            figure(f1)
            plot(t,down_data,'k');ax1=gca;
            title ('Left click = Select, Right click = Preview')
            
            % add lines according to comments
            if nargin>4
                
                hold on;
                counter=0;
                %get normalised comment times in minutes
                com_time = round(com(:,3)./Fs/60);
                %     com_time = com(:,3)./Fs;
                %choose only com times in the analyzed block
                com_time = com_time(com(:,2)== block);
                Txt_com = comtext(com(:,2)== block,:);
                %add last point
                com_time(length(com_time)+1)= t(end);
                %Txt_com(length(com_time),:)='baseline                   ';
                C = jet(length(com_time));
                Yval = 1.1* max(input);
                
                for ii=1:length(com_time)
                    
                    plot([counter,com_time(ii)],[Yval,Yval],'color',C(ii,:),'linewidth',2)
                    %label
                    if(ii == 1)
                        text((com_time(ii)+counter)/2,Yval*1.05, 'baseline                   '...
                            ,'HorizontalAlignment','center','Color','black','FontSize',14)
                    else
                        text((com_time(ii)+counter)/2,Yval*1.05, Txt_com(ii-1,:),...
                            'HorizontalAlignment','center','Color','black','FontSize',14)
                    end
                    counter = com_time(ii);
                    
                end
                
            end
            
            
            % prettify
            set(gcf,'color','w');ax1.FontWeight = 'bold';
            ax1.XColor = 'k';ax1.YColor = 'k';ax1.FontSize = 14;
            xlabel(' Time (min.)')
            ylabel('Amp. (V)')
            
            % initiate button
            button = 3;
            
            while button == 3
                
                % get x coordinate
                [x,~,button] = ginput(1);
                
                % convert to x to sample number
                x_idx = round(x * Fs * 60);
                
                if button == 1 %left click selects
                    break
                end
                
                % limit boundaries
                if x_idx < Fs+1
                    x_idx = Fs+1;
                elseif x_idx> length(input)
                    x_idx = length(input)-1;
                end
                
                figure(f2)
                % plot filtered x zoomed version
                temp_data = input(x_idx-Fs/2+1:x_idx); %medfilt1(,10)
                plot(t_or(x_idx-Fs/2+1:x_idx),temp_data,'k')
                ax2 = gca;
                set(gcf,'color','w');ax2.Box = 'off';ax2.TickDir = 'out';
                ax2.XColor = 'k';ax2.YColor = 'k';ax2.FontSize = 14;ax2.FontWeight = 'bold';
                ylabel('Amp. (V)')
                xlabel('Time (Sec.)')
                
                figure(f3)
                temp_data = temp_data - mean(temp_data);
                xdft = (abs(fft(temp_data)).^2);
                xdft = 2*xdft*(1/(new_Fs*length(temp_data)));
                psdx = xdft((1:length(xdft)/2+1));
                plot(freq(F1:F2),psdx(F1:F2),'linewidth',1.5)
                ax3 = gca;
                set(gcf,'color','w');ax3.Box = 'off';ax3.TickDir = 'out';
                ax3.XColor = 'k';ax3.YColor = 'k';ax3.FontSize = 14;ax3.FontWeight = 'bold';
                ylabel('Power (V^2 Hz^{-1})')
                xlabel('Freq. (Hz)')
                
                
                figure(f1)
                
            end
            
            % x2 = x2/Fs/60
            close all;
        end
        
        
        
        
    end
    
    
    %%% Dynamic methods %%%
    
    methods % - Misc & object load %
        
        % class constructor used to obtain desktop path
        function obj = spectral_analysis_batch()
            % get path to dekstop
            obj.desktop_path = cd;
            obj.desktop_path = obj.desktop_path(1:strfind( obj.desktop_path,'Documents')-1);
            obj.desktop_path = fullfile( obj.desktop_path,'Desktop\');
        end
        
        % update object properies
        function obj = get_parameters(obj)
            
            obj.set_array.ext = 'mat'; % app.file_ext; % file_ext
            obj.set_array.channel_struct = 'bla';  % app.ChannelStructureEditField.Value; % Channel structure
            obj.set_array.channel =  'bla'; % app.ChannelAnalyzedEditField.Value; % Channel to be analyzed
            
            obj.set_array.file_format = 'single'; % app.BinaryFormatEditField.Value; % file format
            obj.set_array.norm = 1; % app.NormFactorEditField.Value; % normalization factor for values (y)
            obj.set_array.samplerate = 4000; % app.SamplerateEditField.Value; % sampling rate (samples per second)
            obj.set_array.start_time = 1; % app.Analysis_Duration.Value; % start time for file analysis
            obj.set_array.end_time = 12; % app.Analysis_Duration_2.Value; % end time for file analysis
            
            obj.set_array.fft_window_type = 'Hann'; %app.WindowTypeDropDown.Value; % spectral window type
            obj. set_array.fft_overlap = 50; % app.OverlapEditField.Value; % percentage overlap for fft window segments
            obj.set_array.fft_windowsz = 5; % app.SizesecEditField.Value; % duration (seconds) of fft window
        end
        
        % create analysis directory
        function obj = make_analysis_dir(obj)
            % obj.save_path,obj.raw_psd_path  = make_analysis_dir(obj.lfp_data_path)
            % get analysis and raw psd folder paths
            k = strfind(obj.lfp_data_path,'\');
            obj.save_path = fullfile(obj.lfp_data_path(1:k(end-1)-1),['analysis_' obj.channel_struct{obj.channel_No}]);
            obj.raw_psd_path = fullfile(obj.save_path,'raw_psd'); % ['raw_psd_' obj.channel_struct{obj.channel_No}]
            
            % create analysis folder
            mkdir(obj.save_path);
            % create raw PSD folder
            mkdir(obj.raw_psd_path);
        end
        
        % reload object
        function copy_obj = reload_object(obj)
            temp_path = uigetdir(obj.desktop_path);
            % check if user gave input
            if temp_path == 0
                copy_obj = 0;
                return
            end
            obj.save_path = temp_path;
            load(fullfile(obj.save_path,'psd_object'));%#ok
            copy_obj = copy(psd_object);
        end
        
        % reset paths of migrated object
        function reloc_obj = reset_migrated_obj_path(obj)
            
            new_path = uigetdir(obj.desktop_path);
            
            % check if user gave input
            if new_path == 0
                return
            end
            
            load(fullfile(new_path,'psd_object'));%#ok
            
            % change paths to match the new path
            
            psd_object.desktop_path = obj.desktop_path; % path to desktop
            
            psd_object.save_path = new_path; % analysed folder and object path
            
            psd_object.lfp_data_path = fullfile(new_path,'raw_data');% LFP folder path
            psd_object.raw_psd_path = fullfile(new_path,'raw_psd'); % raw psd folder path
            psd_object.raw_psd_user = fullfile(new_path,'raw_psd/raw_psd_user'); % raw unseparated psd path
            psd_object.proc_psd_path = fullfile(new_path,'processed_psd'); % processed psd path
            psd_object.excld_path = fullfile(new_path,'excluded'); % excluded psd path
            psd_object.export_path = fullfile(new_path,'exported');% path for exported tables and parameters
            
            % save_changes
            save(fullfile(new_path,'psd_object'),'psd_object')
            
            %copy object for use
            reloc_obj = copy(psd_object);
            
        end
        
        % create bin table to observe experiment length (hours)
        function [colnames, the_list,exp_table] = bin_table(obj)
            
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path, ['*.' obj.set_array.ext]));
            
            % loop through experiments and get exp name and length
            for ii = 1:length(lfp_dir)
                
                % get file path
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(ii).name);
                
                % map the file to memory
                m = memmapfile(Full_path,'Format', obj.set_array.file_format);
                
                % create list
                the_list{ii,1} = lfp_dir(ii).name;
                the_list{ii,2} = length(m.data)/obj.Tchannels/60/60/obj.Fs;
                
                %clear memmap object
                clear m
                
            end
            
            colnames = {'Exp-Name', 'Exp-Length-Hours'};
            exp_table = cell2table(the_list,'VariableNames', {'Exp_Name' 'Exp_Length_hours'});
        end
        
        % create mat table to observe experiment length (minutes)
        function [colnames, the_list,exp_table] = mat_raw_table(obj)
            % [colnames, the_list,exp_table] = mat_table(obj)
            % outputs list of file names and duration in minutes
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path ,'*.mat'));
            
            % loop through experiments and get exp name and length
            for ii = 1:length(lfp_dir)
                
                % get file path
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(ii).name);
                
                % map the file to memory
                load(Full_path,'data','samplerate');
                
                % create list
                the_list{ii,1} = lfp_dir(ii).name;
                
                % time
                the_list{ii,2} = length(data)/obj.Tchannels/60/samplerate;
                
            end
            
            colnames = {'Exp-Name', 'Exp_Length_mins'};
            exp_table = cell2table(the_list,'VariableNames', {'Exp_Name' 'Exp_Length_mins'});
        end
        
        % create mat table to observe experiment length (minutes)
        function [colnames, the_list,exp_table] = mat_table(obj)
            % [colnames, the_list,exp_table] = mat_table(obj)
            % outputs list of file names and duration in minutes
            % get lfp directory
            lfp_dir = dir(fullfile(obj.raw_psd_user,'*.mat'));
            
            % get unique conditions
            unique_conds = spectral_analysis_batch.isolate_condition({lfp_dir.name},1);
            
            % get exp list
            exp_list = spectral_analysis_batch.get_exp_array(lfp_dir,unique_conds,1);
            
            % init list
            the_list = cell(length(lfp_dir),2);
            % loop through experiments and get exp name and length
            for ii = 1:length(lfp_dir)
                
                % get file path
                Full_path = fullfile(obj.raw_psd_user, exp_list{ii});
                
                % map the file to memory
                load(Full_path,'power_matrix');
                
                % get length of power matrix
                [~,len] = size(power_matrix);
                
                % create list
                the_list{ii,1} = exp_list{ii};
                
                % get duration
                the_list{ii,2} = len*(obj.dur/2)/60;

                
            end
            
            colnames = {'Exp-Name', 'Exp_Length_mins'};
            exp_table = cell2table(the_list,'VariableNames', {'Exp_Name' 'Exp_Length_mins'});
        end
        
        % get times for conditions separated by user
        function get_cond_times(obj,path1)
            % get_cond_times(obj,obj.proc_psd_path)
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            exp_list = spectral_analysis_batch.cellfromstruct(mat_dir,1);
            
            try
                unique_conds = spectral_analysis_batch.isolate_condition(exp_list,1);
            catch
                disp('could not get unique conditions')
                unique_conds = {'cond1';'cond2';'cond3'};
            end
            
            % get conditions
            prompt = {'Enter conditions ( separated with ; ):'};
            Prompt_title = 'For observation only'; dims = [1 50];
            definput = {strjoin(unique_conds,';')};
            answer = inputmod(prompt,Prompt_title,dims,definput);
            
            obj.condition_id = strsplit(answer{1},';');
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(path1,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,0);
            
            % remove empty rows
            exp_list(any(cellfun(@isempty, exp_list), 2),:) = [];
            
            obj.condition_time=[];
            for ii = 1:length(obj.condition_id)
                load(fullfile(path1, exp_list{1,ii}),'proc_matrix');
                [~,lengthx] = size(proc_matrix);
                obj.condition_time = [obj.condition_time, lengthx];
            end
            
            
        end
        
        % exclude processed psds from analysis
        function exp_exclude(obj)
            % exp_exclude(obj)
            
            % set exluded path
            obj.excld_path = fullfile(obj.proc_psd_path,'excluded');
            
            % create exluded folder if it does not exists
            if exist(obj.excld_path,'dir')== 0
                mkdir(obj.excld_path)
            end
            
            % get file list
            [filelist,~] = uigetfile(fullfile(obj.proc_psd_path,'*.mat'),...
                'Select One or More Files','MultiSelect', 'on');
            
            % check if user has selected files
            if iscell(filelist)==0
                
                if filelist ==0
                    return
                end
                
                % move on file to exluded foler
                movefile(fullfile (obj.proc_psd_path,filelist),...
                    fullfile(obj.excld_path,filelist))
            else
                
                % loop through file list and exclude files from filelist
                for i = 1:length(filelist)
                    movefile(fullfile (obj.proc_psd_path,filelist{i}),...
                        fullfile(obj.excld_path,filelist{i}))
                end
            end
            
            %save psd_object
            psd_object = saveobj(obj); %#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
            
        end
        
        % Delete files
        function del_files(obj)
            % prompt files to delete
            list = {'All','Raw PSDs - Matlab Separated','Raw PSDs',...
                'Excluded PSDs','Processed PSDs'};
            [indx,~] = listdlg('ListString',list,'SelectionMode','single');
            
            if isempty(indx) % check if user entered input
                return
            else
                answer = questdlg('Would you like to delete files?', ...
                    'Menu', 'Yes','No','No');
                if strcmp(answer,'No')
                    return
                end
            end
            
            if indx == 1 || indx == 2 || indx == 3
                if exist(obj.raw_psd_user,'dir') > 0
                    % delete raw psds matlab separated
                    delete(fullfile(obj.raw_psd_user,'*.mat'))
                    rmdir (obj.raw_psd_user)
                end
            end
            
            if indx == 1 || indx == 3
                if exist(obj.raw_psd_path,'dir') > 0
                    % delete raw psds
                    delete(fullfile(obj.raw_psd_path,'*.mat'))
                    rmdir (obj.raw_psd_path)
                end
            end
            
            if indx == 1 || indx == 4 || indx == 5
                if exist(obj.excld_path,'dir') > 0
                    % delete excluded psds
                    delete(fullfile(obj.excld_path,'*.mat'))
                    rmdir (obj.excld_path)
                end
            end
            
            if  indx == 1 || indx == 5
                if exist(obj.proc_psd_path,'dir') > 0
                    % delete processed psds
                    delete( fullfile(obj.proc_psd_path,'*.mat'))
                    rmdir (obj.proc_psd_path)
                end
            end
            
            if indx == 1
                % delete main analysis folder with object
                if exist(obj.save_path,'dir') > 0
                    if exist(obj.export_path,'dir') > 0
                        delete(fullfile(obj.export_path,'*'))
                        rmdir (obj.export_path)
                    end
                    delete(fullfile(obj.save_path,'*.mat'))
                    rmdir (obj.save_path)
                end
            end
            
        end
        
    end
    
    methods % A - FFT analysis %
        
        %%% Manual Separation %%%
        
        % extract power matrix from mat file (files separated using comments)
        function extract_pmatrix_mat_user(obj)
            %Enter parameters for time-frequency analysis
            %extract power matrix for each experiment and save in save_path
            %.mat file saved from labchart with default setting
            
            % prompt user for input
            prompt = {'Enter observation frequency range','Enter conditions in sequence (separated by ;)'};
            Prompt_title = 'Input';
            dims = [1 40];
            definput = {'2 - 80','cond1;cond2;cond3'};
            cond_list = inputmod(prompt,Prompt_title,dims,definput);
            
            % get observation frequencies
            freq_range = split(cond_list{1}, '-');
            Flow = str2double(freq_range{1});
            Fhigh = str2double(freq_range{2});
            
            % create analysis folder
            make_analysis_dir(obj)
            
            % make raw exvivo folder
            obj.raw_psd_user =  fullfile(obj.raw_psd_path, 'raw_psd_user'); %['raw_psd_user_' obj.channel_struct{obj.channel_No}]
            mkdir(obj.raw_psd_user)
            
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path,'*.mat'));
            
            % get Fs
            Full_path = fullfile(obj.lfp_data_path, lfp_dir(1).name);
            load(Full_path,'samplerate')
            obj.Fs = samplerate(1);
            
            % get winsize
            obj.winsize = round(obj.Fs*obj.dur);
            
            % get freq vector
            freq = eval(obj.freq_cmd);
            
            % get index values of frequencies
            obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut);  % lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut); % upper boundary
            
            %initialise progress bar
            progressbar('Progress')
            
            % loop through experiments and perform fft analysis
            for i = 1:length(lfp_dir)
                
                % load file
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(i).name);
                load(Full_path,'data','datastart','dataend','com','comtext')
                
                % get channel and block number
                [~, blocks] = size(datastart);
                
                %set while loop conditionals
                curr_block = blocks;
                save_var = 0;
                
                % loop through blocks
                while curr_block >= 1 && save_var == 0
                    
                    % get data on desired channel and block
                    output = data(datastart(obj.channel_No,curr_block): dataend(obj.channel_No,curr_block));
                    
                    % obtain power matrix
                    power_matrix = obj.fft_hann(output,obj.winsize,obj.F1,obj.F2,obj.Fs);
                    
                    % get time index
                    [~,L] = size(power_matrix);
                    t =(1:L)*(obj.dur/2); % time in minutes
                    
                    % obtain power area and peak freq for initial analysis and
                    % remove outliers for plotting
                    [~, peak_freq, power_area] = obtain_pmatrix_params(obj,power_matrix,freq,Flow,Fhigh);
                    [power_area,out_vector] = obj.remove_outliers(power_area,5);
                    power_area(out_vector)=nan;
                    peak_freq(out_vector)=nan;
                    
                    %%%%Plot power area and peak frequency%%%%
                    figure('units','normalized','position',[0.2 0.4 .6 .4])
                    
                    subplot(2,1,1);
                    
                    plot(t,power_area,'k')
                    ylabel('Power Area (V^2)'); % y-axis label
                    % add baseline and drug time from labchart comments
                    [com_time,Txt_com] = obj.add_comments(obj.Fs,power_area,com,comtext,t,curr_block);%#ok
                    ax1 = gca; obj.prettify_o(ax1);
                    
                    subplot(2,1,2);
                    plot(t,peak_freq,'k')
                    xlabel('Time (sec)') % x-axis label
                    ylabel('Peak Freq. (Hz)'); % y-axis label
                    ax2 = gca; obj.prettify_o(ax2);title(strrep(lfp_dir(i).name,'_',' '));
                    
                    % get user input for data analysis
                    user_input = obj.separate_conds(com_time,cond_list{2});
                    
                    % check that user input has the correct format
                    if isempty(user_input)
                        answer = questdlg('Terminate current analysis?', ...
                            'Attention','Yes','No','No');
                        if strcmp(answer,'Yes')
                            % delete current analysis files
                            close all
                            delete(fullfile(obj.raw_psd_user,'*.mat'))
                            rmdir (obj.raw_psd_user)
                            rmdir (obj.raw_psd_path)
                            rmdir (obj.save_path)
                            return
                        elseif strcmp(answer,'No')
                            continue
                        end
                        
                    else
                        % get integer of save variable
                        save_var = str2double(user_input{1});
                        
                        if save_var == 0
                            % start from last block and remove 1 on each loop
                            curr_block = curr_block -1;
                            continue
                            
                        elseif save_var == 1
                            if length(strsplit(user_input{2},';')) ~= length(com_time)
                                % get user input for data analysis
                                disp ('input structure is not correct')
                                save_var = 0;
                                close all;
                                continue
                            else
                                curr_block = 0;
                            end
                            
                            % start from last block and remove 1 on each loop
                            curr_block = curr_block -1;
                            
                        end
                    end
                end
                
                % save files
                if (save_var == 1)
                    % get path to exp name without mat ending
                    exp_name = fullfile(obj.raw_psd_user,erase(lfp_dir(i).name,'.mat'));
                    % separate conditions
                    obj.separate_psd(power_matrix,exp_name,strsplit(user_input{2},';'),com_time,obj.dur)
                    close
                end
                
                % update progress bar
                progressbar(i/length(lfp_dir))
                
            end
            
            % save psd_object
            psd_object = saveobj(obj);%#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
            
            close all;
        end
        
        % analyse the whole file
        % extract power matrix from bin file (files separated using comments)
        function extract_pmatrix_bin_user(obj)
            
            % prompt user for input
            prompt = {'Enter observation frequency range','Enter conditions in sequence (separated by ;)'};
            Prompt_title = 'Input';
            dims = [1 40];
            definput = {'2 - 80','cond1;cond2;cond3'};
            cond_list = inputmod(prompt,Prompt_title,dims,definput);
            
            % get observation frequencies
            freq_range = split(cond_list{1}, '-');
            Flow = str2double(freq_range{1});
            Fhigh = str2double(freq_range{2});

            % create analysis folder
            make_analysis_dir(obj)
            
            % make raw exvivo folder
            obj.raw_psd_user =  fullfile(obj.raw_psd_path, 'raw_psd_user'); %['raw_psd_user_' obj.channel_struct{obj.channel_No}]
            mkdir(obj.raw_psd_user)
            
            % get lfp directory
            lfp_dir_bin = dir(fullfile(obj.lfp_data_path,'*.adibin'));
            lfp_dir_mat = dir(fullfile(obj.lfp_data_path,'*.mat'));
            
            % get Fs
            load(fullfile(obj.lfp_data_path, lfp_dir_mat(1).name),'samplerate')
            obj.Fs = samplerate(1);
            
            % get winsize
            obj.winsize = round(obj.Fs*obj.dur);
            
            % get freq vector
            freq = eval(obj.freq_cmd);
            
            % get index values of frequencies
            obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut);  % lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut); % upper boundary
            
            % get epoch in samples
            epoch = obj.period_dur * 60 * obj.Fs;
            
            %initialise progress bar
            progressbar('Total', 'Exp')
            
            % loop through experiments and perform fft analysis
            for ii = 1:length(lfp_dir_bin)
                
                % get file path and load file
                path_dir = fullfile(obj.lfp_data_path, lfp_dir_bin(ii).name);
                path_mat = fullfile(obj.lfp_data_path, lfp_dir_mat(ii).name);
                load(path_mat,'datastart','dataend','com','comtext')
                              
                % set data starting point for analysis
                data_start = obj.set_array.start_time * epoch;
                
                % initalise power matrix
                power_matrix = [];
                
                % map the whole file to memory
                m = memmapfile(path_dir,'Format',obj.set_array.file_format);
                
                % get file length for one channel and clear memmap
                len = length(m.Data)/obj.Tchannels; clear m;
                loop_n = floor(len/epoch);
                
                for i = obj.start_time:loop_n
                    
                    % update data_end
                    if i == obj.start_time
                        data_end = data_start + epoch;
                    elseif i == loop_n
                        data_end = len;
                    else  % get back winsize
                        data_end = data_start + epoch + obj.winsize/2;
                    end
                    
                    % map the whole file to memory
                    m = memmapfile(path_dir,'Format',obj.set_array.file_format);
                    
                    % get part of the channel
                    OutputChannel = double(m.Data(data_start*obj.Tchannels+obj.channel_No : obj.Tchannels ...
                        : data_end*obj.Tchannels));
                    
                    % clear memmap object
                    clear m;
                    
                    % set correct units to Volts
                    OutputChannel = OutputChannel/obj.set_array.norm;
                    
                    % obtain power matrix
                    power_matrix_single = obj.fft_hann(OutputChannel,obj.winsize,obj.F1,obj.F2,obj.Fs);
                    
                    % concatenate power matrix
                    power_matrix  = [power_matrix, power_matrix_single];
                    
                    % update data start
                    data_start = data_start + epoch - obj.winsize/2;
                    
                    % update progress bar
                    progressbar( [], i/loop_n)
                end
                              
                
                %%%% ----------- Plot power area ------------------ %%%%
                f = figure('units','normalized','position',[0.2 0.4 .6 .4]);
                [~, ~, power_area] = obtain_pmatrix_params(obj,power_matrix,freq,Flow,Fhigh);
                [power_area,out_vector] = obj.remove_outliers(power_area,5);
                power_area(out_vector)=nan;
                t =(1:size(power_matrix,2))*(obj.dur/2); % time in minutes
                plot(t,power_area,'k')
                ylabel('Power Area (V^2)'); % y-axis label
                %add baseline and drug time from labchart comments
                [com_time,Txt_com] = obj.add_comments(obj.Fs,power_area,com,comtext,t,1);%#ok
                ax1 = gca; obj.prettify_o(ax1);
                title(strrep(erase(lfp_dir_mat(ii).name,'.mat'),'_',' '))
                
                % get user input for data analysis
                user_input = obj.separate_conds(com_time,cond_list{1});
                
                % check that user input has the correct format
                if isempty(user_input)
                    answer = questdlg('Terminate current analysis?', ...
                        'Attention','Yes','No','No');
                    if strcmp(answer,'Yes')
                        % delete current analysis files
                        close all
                        delete(fullfile(obj.raw_psd_user,'*.mat'))
                        rmdir (obj.raw_psd_user)
                        rmdir (obj.raw_psd_path)
                        rmdir (obj.save_path)
                        return
                    elseif strcmp(answer,'No')
                        continue
                    end
                    
                else
                    % get integer of save variable
                    save_var = str2double(user_input{1});
                    
                    if save_var == 0
                        continue
                        
                    elseif save_var == 1
                        if length(strsplit(user_input{2},';')) ~= length(com_time)
                            % get user input for data analysis
                            disp ('input structure is not correct')
                            % reset vars
                            close(f)
                            save_var = 0;
                            ii  = ii -1;
                            continue
                        end
                        
                    end
                end            
            
            % save files
            if (save_var == 1)
                % get path to exp name without mat ending
                exp_name = fullfile(obj.raw_psd_user,erase(lfp_dir_mat(ii).name,'.mat'));
                % separate conditions
                obj.separate_psd(power_matrix,exp_name,strsplit(user_input{2},';'),com_time,obj.dur)
                close
            end
            
            % update progress bar
            progressbar( ii/length(lfp_dir_bin), [])
        
            end
        
            % save psd_object
            psd_object = saveobj(obj);%#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
            
            close all;
        end
        
        % consistent time separation across conditions and experiments
        function file_split_by_time(obj)
            % file_split_by_time(obj)
            
            % make all zeros except baseline to 1
            cond_time = obj.cond_orig_sep_time;
            
            % get separation vector
            cond_time = (cond_time * 60) / (obj.dur/2); % convert to blocks
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.raw_psd_user,'*.mat'));
            
            % separate conditions into a list
            new_array = obj.cellfromstruct(mat_dir,1);
            cond_list =  obj.filter_list(new_array,obj.condition_id) ;
            
            % get size for each condition
            [exps, conds] = size(cond_list);
            
            % make cond time to matrix for loop configuration
            cond_time = reshape(cond_time,2,conds)';
            
            % initialise progress bar
            progressbar('Separating files')
            
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through condiitons
                    
                    if isempty(cond_list{i,ii})==0 % check if file exists
                        
                        % file load
                        load(fullfile(obj.raw_psd_user , cond_list{i,ii}),'power_matrix');
                        
                        % get matrix size
                        [~,len] = size(power_matrix);
                        
                        if (cond_time(ii,2)-cond_time(ii,1))>len % check if extracted length exceeds bounds
                            disp(['the file' cond_list{i,ii} ' exceeds size bounds'])
                        else
                            
                            % get extracted times for each condition
                            if cond_time(ii,1) < 0
                                power_matrix = power_matrix(:,end + cond_time(ii,1)+1:end + cond_time(ii,2));
                            else
                                power_matrix = power_matrix(:,cond_time(ii,1)+1:cond_time(ii,2));
                            end
                            
                            % save file
                            save(fullfile(obj.raw_psd_path,cond_list{i,ii}),'power_matrix')
                        end
                    end
                end

                progressbar (i/exps) % update progress bar
            end
            
            % save psd_object
            psd_object = saveobj(obj);%#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
        end
        
        % time_locked_separation
        function time_locked_separation(obj)
            % time_locked_separation(obj)
            
            f = figure(); % init figure
            lfp_dir = dir(fullfile(obj.raw_psd_user,'*.mat')); % get lfp files
            unique_conds = spectral_analysis_batch.isolate_condition({lfp_dir.name},1); % get unique conditions
            exp_list = spectral_analysis_batch.get_exp_array(lfp_dir,unique_conds,1); % get exp list
            
            if strcmp(obj.box_or_table,'table') == 1
                
                % init list
                the_list = cell(length(lfp_dir),2);
                % loop through experiments and get exp name and length
                for ii = 1:length(lfp_dir)
                    
                    % get file path
                    Full_path = fullfile(obj.raw_psd_user, exp_list{ii});
                    
                    % map the file to memory
                    load(Full_path,'power_matrix');
                    
                    % get time bins of power matrix
                    [~,len] = size(power_matrix);
                    
                    % create list
                    the_list{ii,1} = exp_list{ii};
                    
                    % get duration
                    the_list{ii,2} = len*(obj.dur/2)/60;
                end
                
                colnames = {'Exp-Name', 'Exp_Length_mins'};
                exp_table = cell2table(the_list,'VariableNames', {'Exp_Name' 'Exp_Length_mins'});
                uitable(f, 'Data', the_list, 'ColumnName', colnames,'Units', 'Normalized', 'Position',[0.1, 0.1, 0.8, 0.8]);
                
            elseif strcmp(obj.box_or_table,'box_plot') == 1
                
                % create duration list
                dur_list = zeros(size(exp_list));
                
                % get size of condition list
                [exps, conds] = size(exp_list);
                
                for i = 1:exps % loop through experiments
                    for ii = 1:conds % loop through conditions
                        load(fullfile(obj.raw_psd_user,exp_list{i,ii}),'power_matrix') % load pmat
                        dur_list(i,ii) = size(power_matrix,2)*(obj.dur/2)/60; % get in minutes
                    end
                end
                
                boxplot(dur_list,'Labels',unique_conds,'color', [0 ,0, 0]);
                % spectral_analysis_batch.dot_plot(dur_list',unique_conds,1,0,[0.5 0.5 0.5 ;0 0 0])
                ylabel('Duration - Minutes')
                ax1 = gca; spectral_analysis_batch.prettify_o(ax1)
            end
            
            % wait for user to close
            uiwait(f);
            
            % get example vector for splitting file based on unique
            % conditions
            nums = mat2str(zeros(1,length(unique_conds)*2));nums(end)=[];nums(1)=[];
            
            % get user input
            prompt = {'Conditions (separated by ;) :',' Condition duration - mins (space separated) :'};
            Prompt_title = 'Input'; dims = [1 35];
            definput = {strjoin(unique_conds,';'),nums};%'-5 -0 0 8'
            answer = inputmod(prompt,Prompt_title,dims,definput);
            
            if isempty(answer) == 0 % proceed only if user clicks ok
                obj.condition_id = strsplit(answer{1},';');
                obj.cond_orig_sep_time = str2num(answer{2});
                file_split_by_time(obj);
            end
        end
        
        %%% Automatic Separation %%%
        
        % extract power matrix from mat file (files not separated)
        function extract_pmatrix_mat(obj)
            
            % create analysis folder
            make_analysis_dir(obj);
            
            % make raw exvivo folder
            obj.raw_psd_user =  fullfile(obj.raw_psd_path,'raw_psd_user'); % ['raw_psd_user_' obj.channel_struct{obj.channel_No}]
            mkdir(obj.raw_psd_user);
            
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path,'*.mat'));
            
            % Get Fs
            Full_path = fullfile(obj.lfp_data_path, lfp_dir(1).name);
            load(Full_path,'samplerate')
            obj.Fs = samplerate(obj.channel_No);
            
            % get winsize
            obj.winsize = round(obj.Fs*obj.dur);
            
            % get index values of frequencies
            obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut); % lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut); % upper boundary
            
            % initialise progress bar
            progressbar('Progress')
            
            % loop through experiments and perform fft analysis
            for i = 1:length(lfp_dir)
                % load file
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(i).name);
                load(Full_path,'data','datastart','dataend')
                
                % get data on desired channel and block
                output = data(datastart(obj.channel_No,obj.block_number): dataend(obj.channel_No,obj.block_number));
                
                % obtain power matrix
                power_matrix = obj.fft_hann(output,obj.winsize,obj.F1,obj.F2,obj.Fs);%#ok
                save(fullfile(obj.raw_psd_user,lfp_dir(i).name),'power_matrix')
                
                % update progress bar
                progressbar(i/length(lfp_dir))
            end
            
            % save psd_object
            psd_object = saveobj(obj);%#ok
            save(fullfile(obj.save_path,'psd_object.mat'),'psd_object');
        end
        
        % extract power matrix from bin file (files not separated)
        function extract_pmatrix_bin(obj)
            % create analysis folder
            make_analysis_dir(obj)
            
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path,['*.' obj.set_array.ext]));
            
            % get winsize
            obj.winsize = round(obj.Fs*obj.dur);
            
            % get index values of frequencies
            obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut); % lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut);% upper boundary
            
            % get epoch in samples
            epoch = obj.period_dur * 60 * obj.Fs;
            
            %initialise progress bar
            progressbar('Total', 'Exp')
            
            % loop through experiments and perform fft analysis
            for ii = 1:length(lfp_dir)
                
                % get file path
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(ii).name);
                
                % set data starting point for analysis
                data_start = obj.set_array.start_time * epoch;
                
                % initalise power matrix
                power_matrix = [];
                
                for i = obj.start_time:obj.Nperiods % loop across total number of periods
                    
                    % update data_end
                    if i == obj.start_time
                        data_end = data_start + epoch;
                    else  % get back winsize
                        data_end = data_start + epoch + obj.winsize/2;
                    end
                    
                    % map the whole file to memory
                    m = memmapfile(Full_path,'Format',obj.set_array.file_format);
                    
                    % get part of the channel
                    OutputChannel = double(m.Data(data_start*obj.Tchannels+obj.channel_No : obj.Tchannels ...
                        : data_end*obj.Tchannels));
                    
                    % clear  memmap object
                    clear m;
                    
                    % set correct units to Volts
                    OutputChannel = OutputChannel/obj.set_array.norm;
                    
                    % obtain power matrix
                    power_matrix_single = obj.fft_hann(OutputChannel,obj.winsize,obj.F1,obj.F2,obj.Fs);
                    
                    % concatenate power matrix
                    power_matrix  = [power_matrix, power_matrix_single];
                    
                    % update data start
                    data_start = data_start + epoch - obj.winsize/2;
                    
                    % update progress bar
                    progressbar( [], i/(obj.Nperiods-obj.start_time))
                end
                
                % save power matrix per experiment
                save(fullfile(obj.raw_psd_path,erase(lfp_dir(ii).name,['.' obj.set_array.ext])),'power_matrix')
                
                % update progress bar
                progressbar( ii/length(lfp_dir), [])
            end
            
            %save psd_object
            psd_object = saveobj(obj);%#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
            
        end
        
        
        %%% --------------------------------------------------------------- %%%
        
    end
    
    methods % B - PSD pre-processing %
        
        % general psd processing program
        function psd_processing(obj)
            
            % make processed psd directory
            obj.proc_psd_path = fullfile(obj.save_path,'processed_psd'); % ['processed_psd_' obj.channel_struct{obj.channel_No}]
            mkdir(obj.proc_psd_path)
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.raw_psd_path,'*.mat'));
            
            % get freq vector
            freq = eval(obj.freq_cmd);
            
            % initialise progress bar
            progressbar('Progress')
            
            % loop through experiments
            for i = 1:length(mat_dir)
                
                % load wild type and perform analysis
                strc_psd = load(fullfile(obj.raw_psd_path ,mat_dir(i).name)); %struct = rmfield(struct,'power_matrix')
                
                if obj.bin_size ~= -1 % merge bins
                    proc_matrix = merge_bins(obj,strc_psd.power_matrix);
                    
                else % create proc matrix if not merging
                    proc_matrix = strc_psd.power_matrix;
                    
                end
                
                %1%%  remove outliers from data
                if obj.outlier_var ~= -1
                    [proc_matrix,~]  = remove_outliers_pmatrix(obj,proc_matrix,freq,obj.LowFCut,obj.HighFCut);
                end
                
%                 % detect nans after filling
%                 if any(any(isnan(proc_matrix))) ==1
%                     disp(i)
%                 end
                
                %2%%  remove noise from data %%%
                if obj.noise_var ~= -1
                    proc_matrix = remove_noise_pmatrix(obj,proc_matrix);
                end
                
                %3%%  normalise data %%%
                if strcmp (obj.norm_var, 'log(e)')
                    proc_matrix  = log(proc_matrix);
                    
                elseif strcmp (obj.norm_var, 'log(10)')
                    proc_matrix  = log10(proc_matrix);
                    
                elseif strcmp (obj.norm_var, 'maxVal')
                    proc_matrix  = power_matrix / max(proc_matrix(:));
                    
                elseif strcmp(obj.norm_var, 'totalpower')
                    proc_matrix = obj.TotalPower_norm(proc_matrix);
                end
                
                %4%%  linear psd %%%
                if strcmp(obj.linear_var,'yes')
                    [proc_matrix,~]  = lin_p_matrix(obj,proc_matrix);
                end
                
                % save new file
                save(fullfile(obj.proc_psd_path,mat_dir(i).name),'proc_matrix')
                
                % update progress bar
                progressbar(i/length(mat_dir))
                
            end
            
            % get condition times
            get_cond_times(obj,obj.proc_psd_path);
            
            %save psd_object
            psd_object = saveobj(obj); %#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
            
        end
        
        %%% subroutines called by the general psd function %%%
        
        % merge bins for smoothing and long files
        function p_matrix_out = merge_bins(obj,power_matrix)
            % p_matrix_out = merge_bins(obj,power_matrix)
            
            % get elements to be averaged
            merge_factor = obj.bin_size/obj.dur;
            
            %get size of current matrix
            [W,L] = size(power_matrix);
            
            % get reps required to cover the file
            new_L = floor(L/merge_factor);
            
            % preallocate matrix
            p_matrix_out =  zeros(W,new_L);
            
            % new counter
            cntr = 1;
            
            for i = 1: merge_factor:L-merge_factor+1 %loop through psd bins
                % get mean
                p_matrix_out(:,cntr) = mean(power_matrix(:,i:i+merge_factor-1),2);
                
                % update counter
                cntr = cntr +1;
                
            end
            
        end
        
        % perform computations across all time steps of one power matrix
        % obtain power area, peak freq and peak power vectors (across time)
        function [peak_power, peak_freq, power_area] = obtain_pmatrix_params(obj,power_matrix,freq,Flow,Fhigh)
            %%% [peak_power, peak_freq, power_area] = obtain_pmatrix_params(obj,power_matrix,Flow,Fhigh)
            % obtain power spectral density parameters withing desired
            % power matrix and freq should alraedy be trimmed once
            % frequencies Flow to Fhigh in Hz
            
            [~,L] = size(power_matrix);
            %pre-allocate vectors
            peak_power = zeros(1,L);
            peak_freq = zeros(1,L);
            power_area = zeros(1,L);
            
            %frequency parameters
            freq_range(1) = obj.getfreq(obj.Fs,obj.winsize,Flow)- obj.F1+1; %new low boundary
            freq_range(2) = obj.getfreq(obj.Fs,obj.winsize,Fhigh) -obj.F1+1; %new high boundary
            
            for i = 1:L %loop through psd bins
                [peak,peak_freqx,p_area]= obj.psd_parameters(power_matrix(freq_range(1):freq_range(2),i),...
                    freq(freq_range(1):freq_range(2)));
                peak_power(i) = peak;
                peak_freq(i) = peak_freqx;
                power_area(i) = p_area;
            end
            
        end
        
        % remove noise from all psd time bins of one experimnet
        function p_matrix_out = remove_noise_pmatrix(obj,power_matrix)
            % p_matrix_out = remove_noise_pmatrix(obj,power_matrix)
            [W,L] = size(power_matrix);
            p_matrix_out =  zeros(W,L);
            for i = 1:L %loop through psd bins
                p_matrix_out(:,i) = obj.remove_noise_psd(power_matrix(:,i),obj.Fs,obj.winsize,obj.noise_var,obj.noisewidth_var,obj.F1);
            end
        end
        
        % remove outliers from all psd time bins of one experimnet
        function [p_matrix_out,outlier_index]  = remove_outliers_pmatrix(obj,power_matrix,freq,Flow,Fhigh)
            % p_matrix_out = remove_outliers_pmatrix(obj,power_matrix,median_filt,Flow,Fhigh)
            % Returns the the outlier free signal (outfree_signal) where
            % outliers are replaced by the median value
            % index_vec indicates by 1 which points are outliers
            
            % get power area
            [~, ~, power_area] = obtain_pmatrix_params(obj,power_matrix,freq,Flow,Fhigh);
            
            % get outlier index
            [~,outlier_index] = obj.remove_outliers(power_area,obj.outlier_var);
            
            % get output matrix
            p_matrix_out = power_matrix;
            
            for i = 1:length(outlier_index)
                % replace outlier value
                if outlier_index(i) == 1
                    p_matrix_out(:,i) =  NaN ;%median(power_matrix(:));
                end
            end
            
            % replace missing
            p_matrix_out = fillmissing(p_matrix_out,'nearest',2);
        end
        
        % linearise PSD
        function [p_matrix_out, back_pmat]  = lin_p_matrix(obj,power_matrix)
            % p_matrix_out = remove_noise_pmatrix(obj,power_matrix)
            [W,L] = size(power_matrix);
            p_matrix_out =  zeros(W,L);
            for i = 1:L %loop through psd bins
                p_matrix_out(:,i) = obj.linearise_fft(freqx',power_matrix(:,i));
            end
            
        end
        
        
        %%% --------------------------------------------------------------- %%%
    end
    
    methods % C - Plots %
        
        % Get time vector
        function [units,factor,t,x_condition_time] = getcond_realtime(obj,vectorx)
            % [units,factor,t,x_condition_time] = getcond_realtime(obj,feature_aver)
            
            if obj.bin_size ~= -1
                t =(0:length(vectorx)-1)*(obj.bin_size/2);
                [units,factor] = obj.autoscale_x(length(t)*(obj.bin_size/2)*obj.Fs,obj.Fs);
                t = t*factor; dt = t(2)-t(1);
            else
                t =(0:length(vectorx)-1)*(obj.dur/2);
                [units,factor] = obj.autoscale_x(length(t)*(obj.dur/2)*obj.Fs,obj.Fs);
                t = t*factor; dt = t(2)-t(1);
            end
            
            x_condition_time = cumsum(obj.condition_time)*dt; %/60; %/60/2
            %             x_condition_time(1) = x_condition_time(1) - dt;
            x_condition_time = x_condition_time - dt;
        end
        
        % Spectrogram - subplot for each experiment
        function spectrogram_subplot(obj,Flow,Fhigh,normlz,paired,sub_plot)
            % spectrogram_plot(obj,Flow,Fhigh)
            % spectrogram subplot within desired frequencies
            % from Flow to Fhigh (in Hz)
            % normlz: 0 = no normalization, 1 = norm to baseline
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,paired);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            
            freq = eval(obj.freq_cmd);
            freqx = freq(Flow:Fhigh);
            
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % create figure
            if sub_plot ==1
                figure()
            end
            
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            
            for i = 1:exps % loop through experiments
                
                if sub_plot == 1
                % subplot
                subplot(ceil(exps/3),3,i)
                elseif sub_plot == 0
                    figure()
                end
                    
                % pre-allocate conditions
                conc_pmat = []; wave_base = 1;
                for ii = 1:conds %concatenate conditions to one vector
                     % if file exists
                     if isempty(exp_list{i,ii})==0
                         % load file
                         load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                         
                         if ii == 1
                             % create concatanated wave
                             conc_pmat = proc_matrix(Flow:Fhigh,:);
                             wave_base = mean(proc_matrix(Flow:Fhigh,:));
                         else
                             %-% concatanate baseline & drug segments
                             conc_pmat = horzcat(conc_pmat,proc_matrix(Flow:Fhigh,:));
                         end
                         
                         if ii == conds && normlz == true % normalise
                             conc_pmat = conc_pmat/mean(wave_base);
                         end
                         
                     else
                         conc_pmat = horzcat(conc_pmat,NaN(length(Flow:Fhigh),obj.condition_time(ii)));                         
                     end
                    
                    end
                    % Get time vector
                    [units,~,t,x_condition_time] = getcond_realtime(obj,mean(conc_pmat,1));
                    x_condition_time = [0 x_condition_time];
                    
                    % Plot spectrogram
                    hold on
                    h = surf(t,freqx,conc_pmat,'EdgeColor','None'); z = get(h,'ZData');
                    
                    if isempty(exp_list{i,ii})==0
                        title(erase(strrep(exp_list{i,ii},'_',' '),'.mat'));
                    end
                    
                    % Format
                    axis1 = gca; obj.prettify_o(axis1); colormap jet;
                    colorbar;  axis tight; shading interp;
                    view(0,90) % make 2d  % view(20,50);
                    
                    % get index at which max occurs and plot max freq line
                    [max_y,idx] = max(conc_pmat);
                    plot3(t,smooth_v1(freqx(idx),10),max_y,'c')
                                       
                    % add conditions on plot
                    for iii = 1: conds
                        xarrow = [x_condition_time(iii+1) x_condition_time(iii+1)];
                        yarrow = [min(freqx) max(freqx)];
                        
                        % add arrow
                        plot3(xarrow,yarrow,[max(z(:)) max(z(:))],'color',[1 1 1],'linewidth',2)
                        
                        % add text
                        text((x_condition_time(iii+1)+ x_condition_time(iii))/2,0.9 * max(freqx),max(z(:)),...
                            strrep(obj.condition_id(iii),'_', ' '),'HorizontalAlignment','center',...
                            'Color','white','FontSize',14,'FontWeight','bold')
                    end                        
                
                if sub_plot == 0
                    xlabel(['Time '  '(' units ')'])
                    ylabel('Freq. (Hz)')
                end
            end
            if sub_plot == 1
                obj.super_labels([],['Time '  '(' units ')'],'Freq. (Hz)')
            end
            
        end    
        
        % Plot Aver PSD with SEM - subplot for each experiment in one figure
        function plot_subPSD(obj,Flow,Fhigh,paired,sub_plot)
            % plot_PSD_single(obj,Flow,Fhigh)
            % plot power spectral density within desired frequencies
            % from Flow to Fhigh
            
            % get title string
            ttl_string = [obj.channel_struct{obj.channel_No} ' ' num2str(Flow) ' - ' num2str(Fhigh) ' Hz'];
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,paired);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow); % low bound
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh); % high bound         
            freq = eval(obj.freq_cmd); freqx_bound = freq(Flow:Fhigh); % get frequencies
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % create figure
            if sub_plot ==1
                figure('Name',ttl_string);
            end
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            
            for i = 1 : exps %loop through experiments
                % subplot
                if sub_plot == 1
                    subplot(ceil(exps/3),3,i);hold on;
                elseif sub_plot == 0
                    figure();hold on;
                end
                
                for ii = 1:conds %loop through condiitons
                    
                    % get color vectors
                    [col_mean,col_sem] = obj.color_vec(ii);
                    
                    
                    % if file exists
                    if isempty(exp_list{i,ii})==0
                        
                        title_str{ii,i} = erase(strrep(exp_list{i,ii},'_',' '),'.mat');%#ok
                        
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                                            
                        % get mean and sem
                        mean_wave = mean(proc_matrix(Flow:Fhigh,:),2)';
                        sem_wave = std(proc_matrix(Flow:Fhigh,:),0,2)'/sqrt(size(proc_matrix,2));
                        mean_wave_plus = mean_wave+sem_wave;  mean_wave_minus = mean_wave-sem_wave;
                        
                        % plot mean and shaded sem
                        Xfill= horzcat(freqx_bound, fliplr(freqx_bound));   %#create continuous x value array for plotting
                        Yfill= horzcat(mean_wave_plus, fliplr(mean_wave_minus));
                        fill(Xfill,Yfill,col_sem,'LineStyle','none','DisplayName','SEM');
                        p(ii) = plot(freqx_bound,mean_wave,'color', col_mean,'Linewidth',1.5);
                        exp_name{ii} = title_str{ii,i};
                    else
                        title_str{ii,i} = 'NaN';%#ok
                        plot(NaN,'DisplayName',title_str{ii,i})
                        axis1 = gca;
                    end
                    
                    if ii == conds
                        if sub_plot == 0
                            xlabel('Freq. (Hz)') ;  ylabel ('Power (V^2 Hz^{-1})')
                        end
                        obj.prettify_o(gca) 
                    end         
                end
                legend(p,exp_name);
            end
            
            % add labels
            if sub_plot == 1
                obj.super_labels(ttl_string,'Freq. (Hz)','Power (V^2 Hz^{-1})')         
            end
        end
        
        % Plot Aver PSD with SEM - mean accross experiment
        function plot_meanPSD(obj,Flow,Fhigh,paired)
            % plot_meanPSD(obj,Flow,Fhigh)
            % plot power spectral density within desired frequencies
            % from Flow to Fhigh
            % paired = remove paris with NaNs
            
            % get title string
            ttl_string = [obj.channel_struct{obj.channel_No} ' ' num2str(Flow) ' - ' num2str(Fhigh) ' Hz'];
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,paired);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            freq = eval(obj.freq_cmd);
            freqx_bound = freq(Flow:Fhigh);
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % create figure
            figure(); hold on;legend()
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            for ii = 1:conds %loop through condiitons
                
                % get color vectors
                [col_mean,col_sem] = obj.color_vec(ii);
                temp_mean = [];
                for i = 1 : exps %loop through experiments to get mean
                    % if file exists
                    if isempty(exp_list{i,ii})==0
                        
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                        
                        % get mean and SEM
                        temp_mean(:,i) = mean(proc_matrix(Flow:Fhigh,:),2)';
                    else
                        disp([exp_list{i,ii} ' is empty'])
                    end
                end
                
                % get mean and sem
                mean_wave = mean(temp_mean,2)';
                sem_wave = std(temp_mean,0,2)'/sqrt(size(temp_mean,2));
                mean_wave_plus = mean_wave + sem_wave;  mean_wave_minus = mean_wave-sem_wave;
                
                % plot mean and shaded sem
                Xfill= horzcat(freqx_bound, fliplr(freqx_bound));   %#create continuous x value array for plotting
                Yfill= horzcat(mean_wave_plus, fliplr(mean_wave_minus));
                fill(Xfill,Yfill,col_sem,'LineStyle','none');
                p(ii) = plot(freqx_bound,mean_wave,'color', col_mean,'Linewidth',1.5); %
                name_array{ii} = strrep(obj.condition_id{ii},'_',' ');
            end
            % prettify and add title
            xlabel('Freq. (Hz)') ;  ylabel ('Power (V^2 Hz^{-1})')
            axis1 = gca;obj.prettify_o(axis1)
            title(ttl_string)
            legend(p, name_array)
            
        end
        
        
        % Aver PSD parameters vs time
        function feature_aver = psd_prm_time(obj,strct1)
            % param_vs_time(obj,strct1)
            % Plot PSD parameters vs time
            % strct1.Flow = 2  low boundary
            % strct1.Fhigh = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            % strct1.removeNans = true % remove nans
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % create figure
            figure(); hold on
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
            % get freq parameters
            freq = eval(obj.freq_cmd); freqx = (freq(obj.F1:obj.F2));
            
            % loop through experiments
            for i = 1:exps
                
                % pre allocate
                wave_temp=[] ; wave_base =1;
                for ii = 1:conds %concatenate conditions to one vector

                    % load file
                    load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                    
                    % get peak_power, peak_freq and power_area [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)]
                    [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)] = obtain_pmatrix_params(obj,proc_matrix,...
                        freqx,strct1.Flow,strct1.Fhigh);
                    
                    if ii == 1
                        % create concatanated wave
                        wave_temp = psd_prm(strct1.par_var,:);
                        wave_base = mean(psd_prm(strct1.par_var,:));
                    else
                        %-% concatanate baseline & drug segments
                        wave_temp = horzcat(wave_temp,psd_prm(strct1.par_var,:));
                    end
                    
                    if ii == conds && strct1.norms_v == true % normalise
                        wave_temp = wave_temp/wave_base;
                    end
                    
                    % empty psd_prm
                    psd_prm = [];
                    
                end               

                feature_aver(:,i) = wave_temp;
                
            end
            
            % remove rows containg NaNs
%             feature_aver = feature_aver(:,all(~isnan(feature_aver)));
            
            % get time
            [units,~,t,x_condition_time] = getcond_realtime(obj,feature_aver);
            
            % get mean and sem as filled
            [mean_wave, xfill,yfill] = obj.getmatrix_mean(feature_aver,t);
            
            % plot individual experiments
            if strct1.ind_v == true
                plot(t,feature_aver,'-','Color', [0.8 0.8 0.8])
            end
            
            % plot mean
            if strct1.mean_v == true
                fill(xfill,yfill,[0.4 0.4 0.4],'LineStyle','none')
                plot(t,mean_wave,'-ok','Linewidth',1.5)
            end
            
            % add arrow
            for iii = 1: conds-1
                xarrow = [x_condition_time(iii) x_condition_time(iii)];
                yarrow = [median(mean_wave)*1.2 median(mean_wave)*2];
                plot(xarrow,yarrow,'r','linewidth',2)
            end
            
            % choose label
            switch strct1.par_var
                case 1
                    ylabel('Peak Power (V^2 Hz^{-1})')
                case 2
                    ylabel('Peak Freq. (Hz)')
                case 3
                    ylabel('Power Area (V^2)')
            end
            
            % set x label
            xlabel(['Time (' units ')'])
            
            % format graph
            axis1 = gca; obj.prettify_o(axis1)
            title([obj.channel_struct{obj.channel_No} ' ' num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'])
            
            % set limits
            xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
            ylim([min(feature_aver(:))- mean(feature_aver(:))/20, max(feature_aver(:))+ mean(feature_aver(:))/20 ])
            
        end
        
        % Aver PSD parameter ratio vs time
        function feature_aver = prm_ratio_time(obj,strct1)
            % feature_aver = prm_ratio_time(obj,strct1)
            % Plot PSD parameters vs time
            % strct1.band1 = [2 4]  low boundary
            % strct1.band2 = [30 80] high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            
            % choose label
            switch strct1.par_var
                case 1
                    label_y = 'Peak Power (V^2 Hz^{-1})';
                case 2
                    label_y = 'Peak Freq. (Hz)';
                case 3
                    label_y = 'Power Area (V^2)';
            end
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % create figure
            figure(); hold on
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
            % get freq parameters
            freq = eval(obj.freq_cmd); freqx = (freq(obj.F1:obj.F2));
            
            % loop through experiments
            for i = 1:exps
                
                for ii = 1:conds %concatenate conditions to one vector
                    %                     if isempty(exp_list{i,ii})==1
                    %                         continue
                    %                     end
                    % load file
                    load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');  
                    
                    % Band 1 % get peak_power, peak_freq and power_area [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)]
                    [psd_prmA{1}, psd_prmA{2}, psd_prmA{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                        strct1.band1(1),strct1.band1(2));
                    % Band 2
                    [psd_prmB{1}, psd_prmB{2}, psd_prmB{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                        strct1.band2(1),strct1.band2(2));
                    
                    % get ratio
                    psd_prm1 = cell2mat(psd_prmA(strct1.par_var));
                    psd_prm2 = cell2mat(psd_prmB(strct1.par_var));
                    psd_prm = psd_prm1./ psd_prm2;
                    
                    if ii == 1
                        % create concatanated wave
                        wave_temp1 = psd_prm1;
                        wave_base1 = mean(psd_prm1);
                        wave_temp2 = psd_prm2;
                        wave_base2 = mean(psd_prm2);
                        wave_temp = psd_prm;
                        wave_base = mean(psd_prm);
                    else
                        %-% concatanate baseline & drug segments
                        wave_temp1 = horzcat(wave_temp1,psd_prm1);
                        wave_temp2 = horzcat(wave_temp2,psd_prm2);
                        wave_temp = horzcat(wave_temp,psd_prm);
                    end
                    
                    if ii == conds && strct1.norms_v == true % normalise
                        wave_temp1 = wave_temp1/wave_base1;
                        wave_temp2 = wave_temp2/wave_base2;
                        wave_temp = wave_temp/wave_base;
                    end
                    
                    % empty psd_prm
                    psd_prm1 = [];
                    psd_prm2 = [];
                    psd_prm = [];
                    
                end
                
                % save values to matrix for analysis and plotting
                %                 if isempty(exp_list{i,ii}) == 0
                feature_aver1(:,i) = wave_temp1;
                feature_aver2(:,i) = wave_temp2;
                feature_aver(:,i) = wave_temp;
                %                 else
                %                     [len,~ ] = size(feature_aver);
                %                     feature_aver1(:,i) = NaN(len,1);
                %                     feature_aver2(:,i) = NaN(len,1);
                %                     feature_aver(:,i) = NaN(len,1);
                %                 end
                
            end
            
            % remove rows containg NaNs
            feature_aver1 = feature_aver1(:,all(~isnan(feature_aver)));
            feature_aver2 = feature_aver2(:,all(~isnan(feature_aver)));
            feature_aver = feature_aver(:,all(~isnan(feature_aver)));
            
            % get time
            [units,~,t,x_condition_time] = getcond_realtime(obj,feature_aver);
            
            % get mean and sem as filled
            [mean_wave1, xfill1,yfill1] = obj.getmatrix_mean(feature_aver1,t);
            [mean_wave2, xfill2,yfill2] = obj.getmatrix_mean(feature_aver2,t);
            [mean_wave, xfill,yfill] = obj.getmatrix_mean(feature_aver,t);
            
            % plot individual experiments
            if strct1.ind_v == true
                subplot(3,1,1);hold on
                plot(t,feature_aver,'Color', [1 0.8 0.8])
                subplot(3,1,2);hold on
                plot(t,feature_aver1,'Color', [0.8 0.8 0.8])
                subplot(3,1,3);hold on
                plot(t,feature_aver2,'Color', [0.8 0.8 0.8])
            end
            
            % plot mean
            if strct1.mean_v == true
                subplot(3,1,1);hold on
                fill(xfill,yfill,[1 0.4 0.4],'LineStyle','none')
                plot(t,mean_wave,'color', [0.6 0 0]  ,'Linewidth',1.5)
                ylabel('Ratio')
                axis1 = gca; obj.prettify_o(axis1)
                xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
                ylim([min(feature_aver(:))- mean(feature_aver(:))/20, ...
                    max(feature_aver(:))+ mean(feature_aver(:))/20 ])
                
                subplot(3,1,2);hold on
                fill(xfill1,yfill1,[0.4 0.4 0.4],'LineStyle','none')
                plot(t,mean_wave1,'k','Linewidth',1.5)
                ylabel([num2str(strct1.band1(1)) ' - ' num2str(strct1.band1(2)) ' Hz'])
                xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
                ylim([min(feature_aver1(:))- mean(feature_aver1(:))/20, ...
                    max(feature_aver1(:))+ mean(feature_aver1(:))/20 ])
                axis1 = gca; obj.prettify_o(axis1)
                
                subplot(3,1,3);hold on
                fill(xfill2,yfill2,[0.4 0.4 0.4],'LineStyle','none')
                plot(t,mean_wave2,'k','Linewidth',1.5)
                ylabel([num2str(strct1.band2(1)) ' - ' num2str(strct1.band2(2)) ' Hz'])
                xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
                ylim([min(feature_aver2(:))- mean(feature_aver2(:))/20, ...
                    max(feature_aver2(:))+ mean(feature_aver2(:))/20 ])
                xlabel(['Time '  '(' units ')'])
                axis1 = gca; obj.prettify_o(axis1)
                
            end
            
            % add arrow
            for iii = 1: conds-1
                xarrow = [x_condition_time(iii) x_condition_time(iii)];
                yarrow = [median(mean_wave)*1.2 median(mean_wave)*2];
                subplot(3,1,1);hold on
                plot(xarrow,yarrow,'k','linewidth',2)
            end
            
            
            % format graph
            lim1 = mean(strct1.band1);
            lim2 = mean(strct1.band2);
            obj.super_labels([obj.channel_struct{obj.channel_No} ' ' num2str(lim1) ' / ' num2str(lim2) ' Hz'],[],label_y)
            
        end
        
        % Individual PSD parameters vs time
        function psd_prm_time_ind(obj,strct1)
            % param_vs_time(obj,strct1)
            % Plot PSD parameters vs time
            % strct1.Flow = 2  low boundary
            % strct1.Fhigh = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            % strct1.removeNans = true % remove nans
            
            
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
            % remove empty rows
            exp_list(any(cellfun(@isempty, exp_list), 2),:) = [];
            
            % get freq parameters
            freq = eval(obj.freq_cmd); freqx = (freq(obj.F1:obj.F2));
            
            % loop through experiments
            for i = 1:exps
                
                for ii = 1:conds %concatenate conditions to one vector
                    if isempty(exp_list{i,ii})==0
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                        
                        % get peak_power, peak_freq and power_area [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)]
                        [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)] = obtain_pmatrix_params(obj,proc_matrix,...
                            freqx,strct1.Flow,strct1.Fhigh);
                        
                        if ii == 1
                            % create concatanated wave
                            wave_temp = psd_prm(strct1.par_var,:);
                            wave_base = mean(psd_prm(strct1.par_var,:));
                        else
                            %-% concatanate baseline & drug segments
                            wave_temp = horzcat(wave_temp,psd_prm(strct1.par_var,:));
                        end
                        
                        if ii == conds && strct1.norms_v == true %normalise
                            wave_temp = wave_temp/wave_base;
                        end
                        
                        % empty psd_prm
                        psd_prm = [];
                    end
                end
                
                
                % get time
                [units,~,t,x_condition_time] = getcond_realtime(obj,wave_temp);
                
                % get title string
                ttl_string = [obj.channel_struct{obj.channel_No} ' ' num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'];
                
                
                % create figure
                figure();legend;
                title(ttl_string)
                plot(t,wave_temp,'k','Linewidth',1,'DisplayName',...
                    erase(strrep(exp_list{i,ii},'_',' '),'.mat'));hold on;
                
                % add arrow
                for iii = 1: conds-1
                    xarrow = [x_condition_time(iii) x_condition_time(iii)];
                    yarrow = [median(wave_temp)*1.2 median(wave_temp)*2];
                    plot(xarrow,yarrow,'r','linewidth',2)
                end
                
                % choose label
                switch strct1.par_var
                    case 1
                        ylabel('Peak Power (V^2 Hz^{-1})')
                    case 2
                        ylabel('Peak Freq. (Hz)')
                    case 3
                        ylabel('Power Area (V^2)')
                end
                
                % set x label
                xlabel(['Time (' units ')'])
                
                % format graph
                axis1 = gca; obj.prettify_o(axis1)
                title([strrep(erase(exp_list{i,ii},'.mat'),'_',' ') ' ' obj.channel_struct{obj.channel_No} ...
                    ' ' num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'])
                
                % set limits
                xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
                ylim([min(wave_temp)- mean(wave_temp)/20, max(wave_temp)+ mean(wave_temp)/20 ])
                
            end
        end
        
        
        % Aver PSD parameter
        function extr_feature = aver_psd_prm(obj,strct1,plot_type)
            % extr_feature = aver_psd_prm(obj,strct1,plot_type)
            % Plot aver psd parameters
            % strct1.Flow = 2  low boundary
            % strct1.High = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            % strct1.removeNans = % remove pairs with nans
            % plot_type ,1  = dot plot, 2 = bar plot, 3 = box plot
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,strct1.Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,strct1.Fhigh);
            freq = eval(obj.freq_cmd); freqx = freq(Flow:Fhigh);
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % create figure
            figure(); hold on;
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            % preallocate vector
            extr_feature = zeros(conds,exps);
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through conditions
                    
                    if isempty(exp_list{i,ii})==0 % check if file is present
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                        
                        % get mean and sem
                        mean_psd = mean(proc_matrix(Flow:Fhigh,:),2);
                        
                        % get parameters from each experiment
                        [psd_prm(1),psd_prm(2),psd_prm(3)] = obj.psd_parameters(mean_psd,freqx);
                        
                        extr_feature(ii,i) = psd_prm(strct1.par_var);
                    else
                        extr_feature(ii,i) = NaN;
                    end
                    
                end
                
            end
            

            % perform box plot before normalisation
            if plot_type == 3
                spectral_analysis_batch.box_plot(extr_feature,obj.condition_id)
            end
            
            if  strct1.norms_v == true % normalise
                extr_feature = extr_feature./extr_feature(1,:);
            end
            
            % choose between dot or bar plot
            if plot_type == 1
                spectral_analysis_batch.dot_plot(extr_feature,obj.condition_id,strct1.ind_v,strct1.mean_v...
                    ,[0.5 0.5 0.5 ;0 0 0])
            end
            
            if plot_type == 2
                spectral_analysis_batch.bar_plot(extr_feature,obj.condition_id,strct1.ind_v,strct1.mean_v)
            end
            
            % graph title
            title([obj.channel_struct{obj.channel_No} ' ' num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'])
            
            % choose label
            switch strct1.par_var
                case 1
                    ylabel('Peak Power (V^2 Hz^{-1})')
                case 2
                    ylabel('Peak Freq. (Hz)')
                case 3
                    ylabel('Power Area (V^2)')
            end
            
        end
        
        % Aver PSD parameter ratio
        function extr_feature = aver_psd_prm_ratio(obj,strct1,plot_type)
            % extr_feature = aver_psd_prm_ratio(obj,strct1,plot_type)
            % Plot average PSD parameters
            % strct1.band1 = [5 10]  low boundary
            % strct1.band2 = [10 20] high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            % strct1.removeNans = % remove pairs with nans
            % plot_type ,1  = dot plot, 2 = bar plot, 3 = box plot
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);

            % get freq parameters
            f_range(1) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1(1)) - obj.F1+1;
            f_range(2) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1(2)) - obj.F1+1;
            f_range(3) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2(1)) - obj.F1+1;
            f_range(4) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2(2)) - obj.F1+1;
            
            % get frequency vector
            freq = eval(obj.freq_cmd); freq = freq(obj.F1:end);
            
            % create figure
            figure(); hold on;
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            % preallocate vector
            extr_feature = zeros(conds,exps);
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through conditions
                    
                    if isempty(exp_list{i,ii})==0 % check if file is present
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                        
                        % get parameters from each experiment
                        [psd_prmA(1),psd_prmA(2),psd_prmA(3)] = obj.psd_parameters(mean(proc_matrix(f_range(1):f_range(2),:),2)...
                            ,freq(f_range(1):f_range(2)));
                        [psd_prmB(1),psd_prmB(2),psd_prmB(3)] = obj.psd_parameters(mean(proc_matrix(f_range(3):f_range(4),:)...
                            ,2),freq(f_range(3):f_range(4)));
                        
                        extr_feature(ii,i) = psd_prmA(strct1.par_var)/psd_prmB(strct1.par_var);
                    else
                        extr_feature(ii,i) = NaN;
                    end
                    
                end
                
            end
            
%             % remove rows containg NaNs
%             if conds >1
%                 extr_feature = extr_feature(:,all(~isnan(extr_feature)));
%             end
            
            % perform box plot before normalisation
            if plot_type == 3
                obj.box_plot( extr_feature,obj.condition_id)
            end
            
            if  strct1.norms_v == true %normalise
                extr_feature = extr_feature./extr_feature(1,:);
            end
            
            % choose between dot or bar plot
            if plot_type == 1
                obj.dot_plot(extr_feature,obj.condition_id,strct1.ind_v,strct1.mean_v...
                    ,[0.5 0.5 0.5 ;0 0 0])
            end
            
            if plot_type == 2
                obj.bar_plot(extr_feature,obj.condition_id,strct1.ind_v,strct1.mean_v)
            end
            
            % graph title
            title([obj.channel_struct{obj.channel_No} ' ' num2str(mean(strct1.band1)) ' / ' num2str(mean(strct1.band2)) ...
                ' Hz'])
            
            % choose label
            switch strct1.par_var
                case 1
                    ylabel('Peak Power (V^2 Hz^{-1})')
                case 2
                    ylabel('Peak Freq. (Hz)')
                case 3
                    ylabel('Power Area (V^2)')
            end
            
        end
        
    end
    
    methods % D - Export & Statistics %
        
        % missing excluded files
        
        % Get matrix with aver PSD parameters vs time
        function [feature_aver, conds] = psd_prm_matrix(obj,strct1)
            % param_vs_time(obj,strct1)
            % inputs: 1) LFP object , 2) structure with 4 fields
            % strct1.Flow = 2  low boundary
            % strct1.Fhigh = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise            
            % Outputs
            % feature aver is a matrix: rows = time bins, columns =           
            % feature aver  is a matrix = rows(exps) * col(conds)
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
            % get freq parameters
            freq = eval(obj.freq_cmd); freqx = (freq(obj.F1:obj.F2));
            
            % loop through experiments
            for i = 1:exps
                
                for ii = 1:conds %concatenate conditions to one vector
                    if isempty(exp_list{i,ii})==0
                        
                        %
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                        
                        % get peak_power, peak_freq and power_area [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)]
                        [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)] = obtain_pmatrix_params(obj,proc_matrix,...
                            freqx,strct1.Flow,strct1.Fhigh);
                        
                        if ii == 1
                            % create concatanated wave
                            wave_temp = psd_prm(strct1.par_var,:);
                            wave_base = mean(psd_prm(strct1.par_var,:));
                        else
                            %-% concatanate conditions (baseline & drug segments)
                            wave_temp = horzcat(wave_temp,psd_prm(strct1.par_var,:));
                        end
                        
                        
                        % empty psd_prm
                        psd_prm = [];
                        
                    else
                        % get nans
                        if ii == 1
                            % create concatanated wave
                            wave_temp = NaN(obj.condition_time(i),1);
                        else
                            %-% concatanate conditions (baseline & drug segments)
                            wave_temp = horzcat(wave_temp, NaN(1,obj.condition_time(i)));
                        end                                               
                        
                    end
                    
                    if ii == conds && strct1.norms_v == true % normalise
                        wave_temp = wave_temp/wave_base;
                    end
                end
                
                % append to feature aver
                feature_aver(:,i) = wave_temp;
                
            end
            
            % remove rows containg NaNs
            % feature_aver = feature_aver(:,all(~isnan(feature_aver)));
            
        end
        
        % Get matrix with aver PSD parameter ratio vs time
        function [feature_aver, conds] = prm_prm_ratio_matrix(obj,strct1)
            % feature_aver = prm_ratio_time(obj,strct1)
            % Plot PSD parameters vs time
            % strct1.band1 = [3 6]  low boundary
            % strct1.band2 = [40 80] high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
            % get freq parameters
            freq = eval(obj.freq_cmd); freqx = (freq(obj.F1:obj.F2));
            
            % loop through experiments
            for i = 1:exps
                
                for ii = 1:conds %concatenate conditions to one vector
                    %                     if isempty(exp_list{i,ii})==1
                    %                         continue
                    %                     end
                    % load file
                    load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                    
                    % get peak_power, peak_freq and power_area [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)]
                    
                    % Band 1
                    [psd_prmA{1}, psd_prmA{2}, psd_prmA{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                        strct1.band1(1), strct1.band1(2));
                    % Band 2
                    [psd_prmB{1}, psd_prmB{2}, psd_prmB{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                        strct1.band2(1),strct1.band2(2));
                    
                    % get ratio
                    psd_prm1 = cell2mat(psd_prmA(strct1.par_var));
                    psd_prm2 = cell2mat(psd_prmB(strct1.par_var));
                    psd_prm = psd_prm1./ psd_prm2;
                    
                    if ii == 1
                        % create concatanated wave
                        wave_temp = psd_prm;
                        wave_base = mean(psd_prm);
                    else
                        %-% concatanate baseline & drug segments
                        wave_temp = horzcat(wave_temp,psd_prm);
                    end
                    
                    if ii == conds && strct1.norms_v == true % normalise
                        wave_temp = wave_temp/wave_base;
                    end
                    
                    % empty psd_prm
                    psd_prm = [];
                    
                end
                
                feature_aver(:,i) = wave_temp;
            end
            feature_aver = feature_aver(:,all(~isnan(feature_aver)));
            
        end
        
        % Aver PSD parameter
        function [extr_feature,conds] = aver_psd_prm_matrix(obj,strct1)
            % extr_feature = aver_psd_prm_matrix(obj,strct1)
            % Plot aver psd parameters
            % strct1.Flow = 2  low boundary
            % strct1.High = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.removeNans = 0 % do not remove Nans
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,strct1.Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,strct1.Fhigh);
            freq = eval(obj.freq_cmd); freqx = freq(Flow:Fhigh);
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            % preallocate vector
            extr_feature = zeros(conds,exps);
            
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through conditions
                    
                    if isempty(exp_list{i,ii})==0 % check if file is present
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                        
                        % get mean and sem
                        mean_psd = mean(proc_matrix(Flow:Fhigh,:),2);
                        
                        % get parameters from each experiment
                        [psd_prm(1),psd_prm(2),psd_prm(3)] = obj.psd_parameters(mean_psd,freqx);
                        
                        extr_feature(ii,i) = psd_prm(strct1.par_var);
                    else
                        extr_feature(ii,i) = NaN;
                    end
                    
                end
                
            end
            
%             % remove rows containg NaNs
%             if conds >1 && strct1.removeNans == 1
%                 extr_feature = extr_feature(:,all(~isnan(extr_feature)));
%             end
            
            if  strct1.norms_v == true %normalise
                extr_feature = extr_feature./extr_feature(1,:);
            end
            
            
        end
        
        % Aver PSD parameter ratio
        function [extr_feature,conds] = aver_psd_prm_matrix_ratio(obj,strct1)
            % extr_feature = aver_psd_prm_matrix_ratio(obj,strct1)
            % Plot average PSD parameters
            % strct1.band1 = 5  low boundary
            % strct1.band2 = 10 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get freq parameters
            f_range(1) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1(1)) - obj.F1+1;
            f_range(2) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1(2)) - obj.F1+1;
            f_range(3) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2(1)) - obj.F1+1;
            f_range(4) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2(2)) - obj.F1+1;
            
            % get frequency vector
            freq = eval(obj.freq_cmd); freq = freq(obj.F1:end);
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            % preallocate vector
            extr_feature = zeros(conds,exps);
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through conditions
                    
                    if isempty(exp_list{i,ii})==0 % check if file is present
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix');
                        
                        % get parameters from each experiment
                        [psd_prmA(1),psd_prmA(2),psd_prmA(3)] = obj.psd_parameters(mean(proc_matrix(f_range(1):f_range(2),:),2)...
                            ,freq(f_range(1):f_range(2)));
                        [psd_prmB(1),psd_prmB(2),psd_prmB(3)] = obj.psd_parameters(mean(proc_matrix(f_range(3):f_range(4),:)...
                            ,2),freq(f_range(3):f_range(4)));
                        
                        extr_feature(ii,i) = psd_prmA(strct1.par_var)/psd_prmB(strct1.par_var);
                    else
                        extr_feature(ii,i) = NaN;
                    end
                end
            end
            
        end
        
        
        
        % Export mean psd parameters to excel table
        function excel_meanprms(obj,strct1,ratio)
            % excel_meanprms(obj,strct1)
            % export data in excel format
            % strct1.Flow = 2  low boundary
            % strct1.Fhigh = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            
            % create exported folder
            if exist(obj.export_path,'dir') == 0
                obj.export_path = fullfile(obj.save_path,'exported');
                mkdir(obj.export_path)
            end
            
            % get parameters
            param_array = {'peak_power';'peak_freq';'power_area'};
            param_units =  {' (V^2/Hz)'; ' (Hz)'; ' (V^2)'};
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get sorted unique list
            exp_list = spectral_analysis_batch.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get matrix with psd properties
            for i = 1:3
                strct1.par_var = i;
                if ratio == 0
                    [mat_psd_var,conds] = aver_psd_prm_matrix(obj,strct1); %#ok
                elseif ratio == 1
                    [mat_psd_var,conds] = aver_psd_prm_matrix_ratio(obj,strct1);%#ok
                end
                eval([param_array{i} ' = transpose(mat_psd_var);'])
            end
            
            % convert matrix to cell
            extr_features = num2cell([ peak_power NaN(size(exp_list,1),1) peak_freq NaN(size(exp_list,1),1) power_area]);
            
            % add exp id
            table_array = [exp_list extr_features];
            
            % create empty table name vector
            [~,W] = size(table_array);
            
            % add table names
            table_names = cell(1,W); table_names{1} = 'Exp_ID';
            prm_names = cell(1,W);
            
            % add conditions and variable names
            k = 1+length(obj.condition_id);
            prm_names(k:conds+1:(W-conds+1)) = cellfun(@(x,y) [x  ' ' y],param_array,param_units,'un',0);
            table_names(k:k+conds-1) = obj.condition_id;      
                    
            % merge main array with column names
            merged_array = [prm_names; table_names;  table_array];
            
            % get spreadsheet file name
            if ratio == 0
                flow_str = strrep(num2str(strct1.Flow),'.',','); fhigh_str = strrep(num2str(strct1.Fhigh),'.',',');
                file_name = [obj.channel_struct{obj.channel_No} '_' flow_str '_' fhigh_str ' Hz_aver'];
            elseif ratio == 1
                flow_str1 = strrep(num2str(strct1.band1(1)),'.',','); fhigh_str1 = strrep(num2str(strct1.band1(2)),'.',',');
                flow_str2 = strrep(num2str(strct1.band2(1)),'.',','); fhigh_str2 = strrep(num2str(strct1.band2(2)),'.',',');
                file_name = [flow_str1 '-' fhigh_str1 ' _ '...
                    flow_str2 '-' fhigh_str2 ' Hz'];             
            end
                       
            % save table to excel file
            xlswrite(fullfile(obj.export_path, file_name),merged_array);
            
        end
        
        % need to add ratios to time !!!
        % Export psd parameters vs time to excel table
        function excel_psd_time(obj,strct1,ratio)
            % excel_psd_time(obj,strct1)
            % export data in excel format
            % strct1.Flow = 2  low boundary
            % strct1.Fhigh = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            
            % create exported folder
            if exist(obj.export_path,'dir') == 0
                obj.export_path = fullfile(obj.save_path,'exported');
                mkdir(obj.export_path)
            end
            
            % get spreadsheet file name
            file_name = [obj.channel_struct{obj.channel_No} '_' num2str(strct1.Flow) '_' num2str(strct1.Fhigh) ' Hz_time'];
            
            % array with parameters
            param_array = {'peak_power','peak_freq','power_area'};
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.lfp_data_path,'*.mat'));
            
            % get sorted list
            exp_list = spectral_analysis_batch.get_exp_array(mat_dir,obj.condition_id,strct1.removeNans);
            
            % get matrix with psd properties
            for i = 1:3
                strct1.par_var = i;
                if ratio ==0
                    [mat_psd_var,conds] = psd_prm_matrix(obj,strct1);
                else
                    [mat_psd_var,conds] = prm_prm_ratio_matrix(obj,strct1);
                end
                
                % get times for each condition
                [time_units,~,t,x_condition_time] = getcond_realtime(obj,mat_psd_var);
                
                % get index in samples
                idx =[];
                
                for ii = 1:conds
                    idx = [idx find(t==x_condition_time(ii))];
                end
                
                % convert psd parameter matrix to cell array
                temp_array = num2cell(mat_psd_var);
                % add time
                temp_array = [num2cell(t') temp_array];
                
                array_names = [{['time (' time_units ')']} exp_list'];
                merged_array = [array_names; temp_array];
                
                xlswrite(fullfile(obj.export_path, file_name),merged_array,i);
                
            end
            
            
            
            % Rename excel sheets
            %             e = actxserver('Excel.Application'); % # open Activex server
            %             ewb = e.Workbooks.Open(fullfile(obj.export_path, file_name)); % # open file (enter full path!)
            %             for i = 1 : conds
            %                 ewb.Worksheets.Item(i).Name = param_array{i}; % # rename 1st sheet
            %                 ewb.Save % # save to the same file
            %             end
            %             ewb.Close(false)
            %             e.Quit
        end
        
        % Export psd parameters vs time to .mat file
        function psd_to_mat_time(obj,strct1,ratio)
            % psd_to_mat_time(obj,strct1)
            
            % create exported folder
            if exist(obj.export_path,'dir') == 0
                obj.export_path = fullfile(obj.save_path,'exported');
                mkdir(obj.export_path)
            end
            
            param_array = {'peak_power','peak_freq','power_area'};
            
            % get matrix with psd properties
            for i = 1:3
                strct1.par_var = i;
                if ratio ==0
                    [mat_psd_var,conds] = psd_prm_matrix(obj,strct1);
                else
                    [mat_psd_var,conds] = prm_prm_ratio_matrix(obj,strct1);
                end
                eval([param_array{i} ' = transpose(mat_psd_var);'])
            end
            
            % get times for each condition
            [time_units,~,t,x_condition_time] = getcond_realtime(obj,mat_psd_var);
            
            % get index in samples
            index_samples =[];
            
            for i = 1:conds
                index_samples = [index_samples find(t==x_condition_time(i))];
            end
            
            % change to appropriate names
            index_time = x_condition_time;  conditions = obj.condition_id; time = t;
            
            % create appropriate file name
            flow_str = strrep(num2str(strct1.Flow),'.',','); fhigh_str = strrep(num2str(strct1.Fhigh),'.',',');
            file_name = [obj.channel_struct{obj.channel_No} '_' flow_str '_' fhigh_str ' Hz_time'];
            
            % save .mat file
            save(fullfile(obj.export_path, file_name),'peak_power','peak_freq','power_area','index_samples','index_time'...
                ,'time','time_units','conditions');
        end
        
        % Export mean psd parameters to .mat file
        function psd_to_mat_aver(obj,strct1,ratio)
            % psd_to_mat_aver(obj,strct1)
            
            % create exported folder
            if exist(obj.export_path,'dir') == 0
                obj.export_path = fullfile(obj.save_path,'exported');
                mkdir(obj.export_path)
            end
            
            % get matrix with psd properties
            param_array = {'peak_power','peak_freq','power_area'};
            
            % get matrix with psd properties
            for i = 1:3
                strct1.par_var = i;
                if ratio == 0
                    [mat_psd_var,~] = aver_psd_prm_matrix(obj,strct1); %#ok
                else
                    [mat_psd_var,~] = aver_psd_prm_matrix_ratio(obj,strct1);%#ok
                end
                eval([param_array{i} ' = transpose(mat_psd_var);'])
            end
            
            % give appropriate names to saved arrays
            conditions = obj.condition_id;
            
            % get file name
            flow_str = strrep(num2str(strct1.Flow),'.',','); fhigh_str = strrep(num2str(strct1.Fhigh),'.',',');
            file_name = [obj.channel_struct{obj.channel_No} '_' flow_str '_' fhigh_str ' Hz_aver'];
            
            % save .mat file
            save(fullfile(obj.export_path,file_name),'peak_power','peak_freq','power_area','conditions');
        end
        
        % matlab stats - needs completion !!!
        function psd_prm_stat(obj,strct1)
            % psd_prm_stat(obj,strct1)
            % Plot aver psd parameters
            % strct1.Flow = 2  low boundary
            % strct1.High = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,strct1.Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,strct1.Fhigh);
            freq = eval(obj.freq_cmd); freqx = freq(Flow:Fhigh);
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % create figure
            figure(); hold on;
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            if cond>2
                return
            end
            
            % preallocate vector
            extr_feature = zeros(conds,exps);
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through conditions
                    
                    if isempty(exp_list{i,ii})==0 % check if file is present
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                        
                        % get mean and sem
                        mean_psd = mean(proc_matrix(Flow:Fhigh,:),2);
                        
                        % get parameters from each experiment
                        [psd_prm(1),psd_prm(2),psd_prm(3)] = obj.psd_parameters(mean_psd,freqx);
                        
                        extr_feature(ii,i) = psd_prm(strct1.par_var);
                    else
                        extr_feature(ii,i) = NaN;
                    end
                    
                end
                
            end
            
            % remove rows containg NaNs
            extr_feature = extr_feature(:,all(~isnan(extr_feature)));
            
            
            if  strct1.norms_v == true %normalise
                extr_feature = extr_feature./extr_feature(1,:);
            end
            
            
            % graph title
            title([obj.channel_struct{obj.channel_No} ' ' num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'])
            
            % choose label
            switch strct1.par_var
                case 1
                    ylabel('Peak Power (V^2 Hz^{-1})')
                case 2
                    ylabel('Peak Freq. (Hz)')
                case 3
                    ylabel('Power Area (V^2)')
            end
            
            
        end
            
        
    end
    
    methods % E - Comparative Plots and statistics %
        
        % Plot between conditions (time)
        function cmp_psd_prm_time(obj,strct1,folder_list,ratio)
            % cmp_psd_prm_time(obj,strct1,folder_list,ratio)
            
            if ratio == false
                % get title string
                ttl_string = [num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'];
            else
                ttl_string = [num2str(mean(strct1.band1)) ' / ' num2str(mean(strct1.band2)) ...
                    ' Hz'];
            end
            
            % choose label
            switch strct1.par_var
                case 1
                    y_label = 'Peak Power (V^2 Hz^{-1})';
                case 2
                    y_label = 'Peak Freq. (Hz)';
                case 3
                    y_label = 'Power Area (V^2)';
            end
            
            
            % create figure
            figure();
            % add figure title
            obj.super_labels(ttl_string,[],y_label)
            obj.prettify_o(gca)
            
            for i  = 1: length(folder_list)
                
                load(fullfile(folder_list{i},'psd_object'));%#ok
                psd_object = copy(psd_object);
                
                if ratio == false
                    [feature_aver, conds] = psd_prm_matrix(psd_object,strct1);
                else
                    [feature_aver, conds] = prm_prm_ratio_matrix(psd_object,strct1);
                end
                
                [col_face,col_edge,col_light] = spectral_analysis_batch.color_vec2(i);
                
                subplot(length(folder_list),1,i)
                %                 mean_wave = mean(feature_aver,2);
                
                % get time
                [units,~,t,x_condition_time] = getcond_realtime(psd_object,feature_aver);
                
                % get mean and sem as filled
                [mean_wave, xfill,yfill] = obj.getmatrix_mean(feature_aver,t);
                
                k = strfind(psd_object.save_path,'\');
                hold on;
                
                %plot ind
                if strct1.ind_v ==1
                    plot(t,feature_aver','Color',[0.9, 0.9, 0.9],'LineWidth',0.5)
                end
                
                % shaded region with SEM
                if strct1.mean_v ==1
                    fill(xfill,yfill,col_light,'LineStyle','none')
                    plot(t,mean_wave,'o','Color',col_edge,'MarkerEdgeColor',col_edge,'MarkerFaceColor',col_face,...
                        'MarkerSize',4,'DisplayName',strrep(psd_object.save_path(k(end-1)+1:k(end)-1),'_',' '));
                end
                % set x label
                xlabel(['Time (' units ')'])
                
                % format graph
                obj.prettify_o(gca)
                
                % set limits
                xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
                %                 ylim([min(feature_aver(:))- mean(feature_aver(:))/20, max(feature_aver(:))+ mean(feature_aver(:))/20 ])
                
                % add arrow
                for iii = 1: conds-1
                    xarrow = [x_condition_time(iii) x_condition_time(iii)];
                    yarrow = [median(mean_wave) max(mean_wave(:))*0.9];
                    plot(xarrow,yarrow,'k','linewidth',3)
                    text(x_condition_time(iii),max(mean_wave(:)),...
                        strrep(psd_object.condition_id{iii+1},'_',' '),'FontSize',14,'FontWeight','bold')
                end
                % add legend
                % legend(h(1))
                
            end
            
        end
        
        % Plot between conditions (average)
        function [bar_vector,pval] = cmp_psd_prm_aver(obj,strct1,folder_list,ratio)
            % [bar_vector,pval] = cmp_psd_prm_aver(obj,strct1,folder_list,ratio)
            % strct inputs
            % strct1.par_vec choose which conditions to compare
            
            
            if ratio == false
                % get title string
                ttl_string = [num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'];
            else
                ttl_string = [num2str(mean(strct1.band1)) ' / ' num2str(mean(strct1.band2)) ...
                    ' Hz'];
            end
            
            % remove all rows with even 1 condition empty
            strct1.removeNans = 0;
            
            % parameter difference
            par_vec = strct1.cond;
            
            % choose label
            switch strct1.par_var
                case 1
                    y_label = 'Peak Power (V^2 Hz^{-1})';
                case 2
                    y_label = 'Peak Freq. (Hz)';
                case 3
                    y_label = 'Power Area (V^2)';
            end
            
            %create _figure
            figure()
            obj.super_labels(ttl_string,[],y_label)
            for i  = 1: length(folder_list)
                
                % load psd object for each set of experiments
                load(fullfile(folder_list{i},'psd_object'));%#ok
                psd_object = copy(psd_object);
                
                % get aver parameters across conditions
                if ratio == false
                    extr_feature = aver_psd_prm_matrix(psd_object,strct1);
                    ttl_string = [psd_object.channel_struct{obj.channel_No} ' ' num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'];
                else
                    extr_feature = aver_psd_prm_matrix_ratio(psd_object,strct1);
                    ttl_string = [psd_object.channel_struct{psd_object.channel_No} ' ' num2str(mean(strct1.band1)) ' / ' num2str(mean(strct1.band2))  ' Hz'];
                end
                % get color
                [col_face,col_edge] = spectral_analysis_batch.color_vec2(i);
                % create subplot
                subplot(1,length(folder_list),i)
                
                % choose between dot or bar plot
                spectral_analysis_batch.dot_plot(extr_feature(par_vec,:),psd_object.condition_id(par_vec),strct1.ind_v,strct1.mean_v...
                    ,[col_face;col_edge])
                    
                % format graph
                spectral_analysis_batch.prettify_o(gca)
                bar_vector{i} = extr_feature(par_vec(2),:) - extr_feature(par_vec(1),:);
                [~, pval(i)] = ttest(extr_feature(par_vec(2),:),extr_feature(par_vec(1),:));
                bar_face_col(i,:) = col_face;
                bar_edge_col(i,:) = col_edge;
                bar_labels{i} = psd_object.condition_id{par_vec(2)};
                
            end
            
            
            figure();hold on
            title(ttl_string)
            % get mean and SEM
            mean_vec = cellfun(@nanmean,bar_vector);
            std_vec = cellfun(@nanstd,bar_vector);
            len_vec = cellfun(@length,bar_vector);
            sem_vec = std_vec./sqrt(len_vec);
            
            % set tick number
            x_ticks = 1:length(mean_vec);
            
            
            for i  = 1: length(folder_list)                
                % add bars
                bar(i, mean_vec(:,i),'FaceColor',bar_face_col(i,:),'EdgeColor',bar_edge_col(i,:));
                % add significance
                mysigstar(gca, i, (mean_vec(:,i)+ sem_vec(:,i))*1.1, pval(i));               
            end
            
            % add SEM error
            errorbar(mean_vec,sem_vec,'.','color','k','Linewidth',1);        
            xticks(x_ticks); xticklabels(strrep(bar_labels,'_',' '));
            
            % set x boundaries
            lim_boundary = length(mean_vec)*0.3; % xtickangle(45);
            xlim([x_ticks(1)-lim_boundary x_ticks(end)+ lim_boundary]);
            ylabel(['Delta ' y_label]);
            obj.prettify_o(gca)
        end
        
    end
    
end