classdef spectral_analysis_batch < matlab.mixin.Copyable
    % x = spectral_analysis_batch
    %
    % x = reload_object(x)
    % folder_list = x.obtain_dir_list(x.desktop_path)
    %
    % extract_pmatrix_mat_user(x,5,80), extract_pmatrix_mat(x), extract_pmatrix_bin(x)
    %
    % 
    % psd_processing(x)
    %
    % plot_subPSD(x,2,20)
    %
    % psd_prm_time(x,a), aver_psd_prm(x,a)
    % 1- peak power, 2 - peak freq, 3. power area
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
    
    properties
        % wave properties
        dur  = 5  % Window duration of the fft transform (in seconds)
        Fs = 4000;% sampling rate
        winsize % fft window size
        block_number = 1;
        channel_No = 1;
        channel_struct = {'BLA'}
        
        % for multichannel channel long recordings
        Tchannels = 3;
        Nperiods = 12; % in hours
        period_dur = 60; % in minutes  
        
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
        
        % psd processing variables
        norm_var = 'no'; % log(10), log(e), max_val
        linear_var = 'no'; %yes/no
        noise_var = -1; % in Hz
        noisewidth_var = 2; % in Hz
        outlier_var = -1; % median multiple
        bin_size = -1; % new bin for merging (in seconds)
        
        % condition identifier
        condition_id = [];
        condition_time = [];
        
        % abort variable
        abort_var = false;
        
    end
    
    % Static methods
    
    methods(Static) % MISC % 
        
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
            axhand.FontName = 'Arial';
            axhand.FontSize = 14;
            axhand.FontWeight = 'bold';
%             axhand.LineWidth = 1;
            
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
        
        
        
    end
        
    methods(Static) % Array sorting and filtering %
        
        % create sorted array
        function exp_list = get_exp_array(mat_dir,conditions)
            
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
        end
        
        % extract column from struct
        function new_list = cellfromstruct(struct_mat,col)
            % cell_col = cellfromstruct(struct_mat,col)
            
            %convert structure to cell
            s = struct2cell(struct_mat);
            
            %assign column to new array
            new_list = s(col,:)';
        end
             
        % filter list based on conditions
        function filtered_list = filter_list(raw_list,str_conds)
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
        function num_array = get_exp_id(raw_list,start_str,condition_list)
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
        function sorted_list = sorted_array(filt_array,str_conds,num_list)
            % sorted_list = sorted_array(filt_array,str_conds,num_list)
            % builds a sorted array
            % filt_array = m x n array  where m is experiment number and n is
            % condition number
            % str_conds = conditions
            % num_list unique exp identifier list
            
            % get list length and width
            [len, wid]= size(filt_array);
            
            % pre-allocate list
            sorted_list = cell(len,wid);
            
            for ii = 1:wid % loop through conditions            
                for i = 1:len % loop across experiments and find match            
                    % check if exp is present
                    x = find(contains(filt_array,['_' num2str(num_list(i)) '_' str_conds{ii}]));
                    if x~=0           
                        sorted_list{i,ii} = filt_array{x};
                    else   
                        sorted_list{i,ii} = [];
                    end    
                end           
            end
            
        end
        
        %not used currently%
        
        % sort list % needs completion
        function num_list = sort_list(raw_list,start_str,end_str)
            % out_list = sort_list(raw_list,start_str,end_str)
            
            for i = 1: length(raw_list)
                % check if condition exists
                if isempty(raw_list{i}) == 0
                    num_list(i) = str2double(cell2mat(extractBetween(...
                        raw_list{i},start_str,end_str)));
                else
                    
                    num_list(i) = nan;
                end
            end
            
            
        end
        
        % filter list based on conditions
        function filtered_list = filter_array(raw_list,str_conds)
            % filtered_list = filter_list(raw_list,str_conds)
            % returns a filtered array separated by conditions
            
            % get list length and width 
            wid = length(str_conds);
            len = round(length(raw_list)/wid);
            
            % create filtered list
            %filtered_list = cell(len,wid);
            
            for ii = 1:wid
                % init loop counter
                cntr = 1;
                
                for i = 1:length(raw_list)
                    % loop across experiments and find match
                    if isempty(strfind(raw_list{i},str_conds{ii}))==0
                        
                        filtered_list{cntr,ii} = raw_list(i);
                        cntr = cntr+1;
                    end
                    
                    
                end
                
            end
            
        end
            
        % merge cell array into one string
        function outarray = merge_cells(arrayin,str)
            % outarray = merge_cells(arrayin,str)
            % arrayin = input cell array
            % separating string
            outarray = arrayin{1};
            for i= 1:length(arrayin)-1
                outarray = [outarray str arrayin{i+1}];
            end
        end
        
    end
    
    methods(Static) % PSD related %
               
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
            
            %ensure that input matches vector format of hann window
            if isrow(input_wave) == 1
                winvec = winvec';
            end
            % get correct channel length for analysis
            Channel_length = length(input_wave)-(rem(length(input_wave),overlap));
            % fprintf('%d samples were discarded to produce equal size windows\n',(rem(length(input_wave),winsize)))
                        
            % preallocate waves
            power_matrix = zeros(F2 - F1+1,(Channel_length/overlap)-2);
            
            % removes dc component
            input_wave = input_wave - mean(input_wave);
            
            % initialise counter
            Cntr = 1;
            for i=1:overlap:Channel_length -(overlap) %loop through signal segments with overlap
                %%get segment
                signal = input_wave(i:i+winsize-1);
                
                %%multiply the fft by hanning window
                signal = signal.*winvec; 
                
                %%get normalised power spectral density
                xdft = (abs(fft(signal)).^2);
                xdft = 2*xdft*(1/(Fs*length(signal)));
                
                %%2 .* to conserve energy across
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
            
            %get threshold
            threshold_p = median(input) * median_mult;
            threshold_m = median(input) / median_mult;
            
            %find outliers
            out_vector = input > threshold_p | input < threshold_m;
            
            %replace with nan if value exceeds threshold according to index array
%             input(out_vector)= NaN; %- originally with Nans - NaN;
        end
        
        % noise removal
        function psd_out = remove_noise_psd(psd,Fs,winsize,noise_freq,width_f)
            %psd_noise_remove(band_freq,width_f,psd)
            
            %removing boise boundaries
            Fnoisemin = spectral_analysis_batch.getfreq(Fs,winsize,noise_freq - width_f); %Fnoisemin= Fnoisemin+1-F1;
            Fnoisemax = spectral_analysis_batch.getfreq(Fs,winsize,noise_freq + width_f); %Fnoisemax= Fnoisemax+1-F1;
            
            %replace noise band range with nans
            psd(Fnoisemin:Fnoisemax) = nan;
            
            %replace nans with pchip interpolatipjn
            psd_out = fillmissing(psd,'pchip');
            
        end 
        
        % extract power area, peak power and peak freq
        function [Peak,peak_freqx,p_area] = psd_parameters(psd_raw,freqx) % ,peak_width
            %%[Peak,peak_freqx,p_area]= psd_parameters(psd_raw,freqx)
            %input parameters are PSD and frequency vectors
            
            %smooth psd curve (smooth factor of 5)
            smth_psd = smooth_v1(psd_raw);
            
            %find peak power and index at which the peak power occurs
            [Peak, x_index] = max(smth_psd);
            peak_freqx = freqx(x_index);
            
            %get power area
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
            
            plot(y_fit); hold on; plot(psd_log)
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
        
    end
    
    methods(Static) % user & plot related %
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
            %add lines according to comments
            hold on;
            counter=0;
            %get normalised comment times in minutes
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
                    text((com_time(ii)+counter)/2,Yval*1.05, 'baseline                   '...
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
            
            for i = 1: length(com_time)-1
                if strcmp(sep_vector{i}, 'false') == 0
                    power_matrix = raw_pmat(:,com_time(i):com_time(i+1));
                    save([exp_name '_' sep_vector{i}],'power_matrix')
                end
            end
        end
        
        %%% plot assisting scripts%%%
        function [mean_wave, xfill,yfill] = getmatrix_mean(feature_aver,t)
            % [mean_wave, xfill,yfill] = getmatrix_mean(feature_aver,t)
            
           [~,len] = size(feature_aver);
            %get mean line with sem
            mean_wave = mean(feature_aver,2)';
            sem_wave = std(feature_aver,0,2)'/sqrt(len);
            mean_wave_plus = mean_wave + sem_wave;
            mean_wave_minus = mean_wave - sem_wave;
            
            %plot mean and shaded sem
            xfill=[t fliplr(t)];   %#create continuous x value array for plotting
            yfill=[mean_wave_plus fliplr(mean_wave_minus)];
        end
        
        %  get user input for Psd parameter plot properties
        function [answer, canceled,formats]  = input_prompt()
            
            name = 'Analysis Input';
            prompt = {'Low Frequency:';'High Frequency:';'Choose PSD parameter:';'Normalise to baseline?';...
                'Plot Type'};
            
            
            % format types included
            formats = struct('type', {}, 'style', {}, 'items', {}, ...
                'format', {}, 'limits', {}, 'size', {});
            
            % new field
            formats(1,1).type   = 'edit';
            formats(1,1).format = 'integer';
            formats(1,1).limits = [0.4 200];
            %
            formats(2,1).type   = 'edit';
            formats(2,1).format = 'integer';
            formats(2,1).limits = [0.4 200];
            %
            formats(3,1).type   = 'list';
            formats(3,1).style  = 'popupmenu';
            formats(3,1).items  = {'Peak Power', 'Peak Frequency', 'Power Area'};
            %
            formats(4,1).type   = 'list';
            formats(4,1).style  = 'popupmenu';
            formats(4,1).items  = {'true', 'false'};
            %
            formats(5,1).type   = 'list';
            formats(5,1).style  = 'popupmenu';
            formats(5,1).items  = {'Mean', 'Individual', 'Mean & Ind'};
            
            % set default answer
            defaultanswer = {2,80,1,2,3};
            
            % obtain answer and canceled var through user input
            [answer, canceled] = inputsdlg(prompt, name, formats, defaultanswer);
            
            % get back values
            % eg. test = formats(3,1).items{answer{3}};
            
            
        end  
        
        %  get user input for psd parameter plot properties
        function [answer, canceled,formats]  = input_prompt_ratio()
            
            name = 'Analysis Input';
            prompt = {'Band - 1:';'Band - 2:';'Band Width';'Choose PSD parameter:';'Normalise to baseline?';...
                'Plot Type'};

            % format types included
            formats = struct('type', {}, 'style', {}, 'items', {}, ...
                'format', {}, 'limits', {}, 'size', {});
            
            % new field
            formats(1,1).type   = 'edit';
            formats(1,1).format = 'integer';
            formats(1,1).limits = [0.4 200];
            %
            formats(2,1).type   = 'edit';
            formats(2,1).format = 'integer';
            formats(2,1).limits = [0.4 200];   
            %
            formats(3,1).type   = 'edit';
            formats(3,1).format = 'integer';
            formats(3,1).limits = [0.4 200];
            %
            formats(4,1).type   = 'list';
            formats(4,1).style  = 'popupmenu';
            formats(4,1).items  = {'Peak Power', 'Peak Frequency', 'Power Area'};
            %
            formats(5,1).type   = 'list';
            formats(5,1).style  = 'popupmenu';
            formats(5,1).items  = {'true', 'false'};
            %
            formats(6,1).type   = 'list';
            formats(6,1).style  = 'popupmenu';
            formats(6,1).items  = {'Mean', 'Individual', 'Mean & Ind'};
            
            % set default answer
            defaultanswer = {5,10,4,3,2,3};
            
            % obtain answer and canceled var through user input
            [answer, canceled] = inputsdlg(prompt, name, formats, defaultanswer);
            
            % get back values
            % eg. test = formats(3,1).items{answer{3}};
      
        end  
          
         %  get user input for psd parameter plot properties
        function [answer, canceled,formats]  = input_prompt_plot()
            
            name = 'Analysis Input';
            prompt = {'Band - 1:';'Band - 2:';'Band Width';'Choose PSD parameter:';'Normalise to baseline?';...
                'Plot parameters';'Plot Type'};

            % format types included
            formats = struct('type', {}, 'style', {}, 'items', {}, ...
                'format', {}, 'limits', {}, 'size', {});
            
            % new field
            formats(1,1).type   = 'edit';
            formats(1,1).format = 'integer';
            formats(1,1).limits = [0.4 200];
            %
            formats(2,1).type   = 'edit';
            formats(2,1).format = 'integer';
            formats(2,1).limits = [0.4 200];   
            %
            formats(3,1).type   = 'edit';
            formats(3,1).format = 'integer';
            formats(3,1).limits = [0.4 200];
            %
            formats(4,1).type   = 'list';
            formats(4,1).style  = 'popupmenu';
            formats(4,1).items  = {'Peak Power', 'Peak Frequency', 'Power Area'};
            %
            formats(5,1).type   = 'list';
            formats(5,1).style  = 'popupmenu';
            formats(5,1).items  = {'true', 'false'};
            %
            formats(6,1).type   = 'list';
            formats(6,1).style  = 'popupmenu';
            formats(6,1).items  = {'Mean', 'Individual', 'Mean & Ind'};            
            %
            formats(7,1).type   = 'list';
            formats(7,1).style  = 'popupmenu';
            formats(7,1).items  = {'Dot Plot', 'Bar Plot','Box Plot'};
            
            % set default answer
            defaultanswer = {5,10,4,3,2,3,1};
            
            % obtain answer and canceled var through user input
            [answer, canceled] = inputsdlg(prompt, name, formats, defaultanswer);
            
            % get back values
            % eg. test = formats(3,1).items{answer{3}};
      
        end  
        
    end
    
    methods(Static) % Plot related
        
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
        
        % get color for vectors
        function [col_mean,col_sem] = color_vec(col_idx)
             % create colour vectors
            col_mat = [0 0 0; 1 0 0; 0 0 0.8; 0 0.7 0; 0 0.7 0.7];
            col_sem_mat = [0.8 0.8 0.8; 1 0.8 0.8; 0.8 0.85 1; 0.75 0.9 0.75; 0.75 0.9 0.9]; 
            %= col_mat; col_sem_mat(col_sem_mat==0)=0.9;
            
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
        function [col_face,col_edge] = color_vec2(col_idx)
             % create colour vectors
            col_mat = [0 0 0;0 0 1; 1 0 0 ;0.8 0.2 0.9 ; 1 0.6 0.2];
            col_sem_mat = [0.5 0.5 0.5; 0.5 0.5 1; 1 0.5 0.5; 0.8 0.5 0.9; 1 0.8 0.6]; 
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
            
            % calculate sem
            sem_pkpower = std(inarray,0,2)/sqrt(exps);
            
            hold on;
            
            % plot individual experiments
            if indiv_var ==1
                plot(inarray,'o-','color',col_vec(1,:),'MarkerFaceColor', col_vec(1,:),'MarkerSize',4);
                
            end
            
            % plot mean experiment
            if mean_var == 1
                plot(mean(inarray,2),'-','color',col_vec(2,:),'MarkerFaceColor', col_vec(2,:),'Linewidth',1.5)
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
            sem_pkpower = std(inarray,0,2)/sqrt(exps);
            
            if indiv_var ==1
                if conds>1
                    plot(inarray,'o','color',[0.5 0.5 0.5],'MarkerFaceColor', [0.5 0.5 0.5],'MarkerSize',4);
                else
                    plot(ones(1),inarray,'o','color',[0.5 0.5 0.5],'MarkerFaceColor', [0.5 0.5 0.5],'MarkerSize',4);
                end
            end
            
            if mean_var == 1
                bar(mean(inarray,2),'FaceColor',  'none', 'EdgeColor', 'k','Linewidth',1.5)
                errorbar(mean(inarray,2),sem_pkpower,'.','color','k','Linewidth',1.5)
            end
            
            % format graph
            ax1= gca;spectral_analysis_batch.prettify_o(ax1)
            
            % set tick number
            x_ticks = 1:conds;
            
            % set x boundaries
            lim_boundary = conds*0.3; % xtickangle(45);
            xlim([x_ticks(1)-lim_boundary x_ticks(end)+ lim_boundary]);
            
            % set x tick labels
            xticks(x_ticks); xticklabels(strrep(conditions,'_',' '))
            
            % set y lim boundaries
            ylim([min(inarray(:))- mean(inarray(:))*0.1...
                max(inarray(:))+ mean(inarray(:))*0.1]);
            
        end
        
        % box plot
        function box_plot(inarray,conditions)
            % box_plot(extr_feature,conditions,indiv_var,mean_var)
            % Inarray = m (conditions) by n (experiment number array 
            % conditons = cell with condition labels
            
           boxplot(inarray','Colors','k','Labels',strrep(conditions,'_',' '))
           ax1= gca;spectral_analysis_batch.prettify_o(ax1)
        end
        
    end
    
    % Object methods
    
    methods % - Misc & object load %  
   
        % class constructor
        function obj = spectral_analysis_batch()
            % get path to dekstop
            obj.desktop_path = cd;
            obj.desktop_path = obj.desktop_path(1:strfind( obj.desktop_path,'Documents')-1);
            obj.desktop_path = fullfile( obj.desktop_path,'Desktop\');              
        end
                
        % create analysis directory
        function obj = make_analysis_dir(obj)
            % obj.save_path,obj.raw_psd_path  = make_analysis_dir(obj.lfp_data_path)
            % get analysis and raw psd folder paths
            k = strfind(obj.lfp_data_path,'\');
            obj.save_path = fullfile(obj.lfp_data_path(1:k(end-1)-1),['analysis_' obj.channel_struct{obj.channel_No}]);
            obj.raw_psd_path = fullfile(obj.save_path,['raw_psd_' obj.channel_struct{obj.channel_No}]);
            
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
        
        % create bin table to observe experiment length (hours)
        function [colnames, the_list,exp_table] = bin_table(obj)
            
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path,'*.adibin'));        
  
            % loop through experiments and get exp name and length
            for ii = 1:length(lfp_dir)

                % get file path
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(ii).name);
                               
                % map the file to memory
                m = memmapfile(Full_path,'Format','int16');  
                
                % create list
                the_list{ii,1} = lfp_dir(ii).name;
                the_list{ii,2} = length(m.data)/3/60/60/obj.Fs;
                
                %clear memmap object
                clear m
            
            end
            
            colnames = {'Exp-Name', 'Exp-Length-Hours'};
            exp_table = cell2table(the_list,'VariableNames', {'Exp_Name' 'Exp_Length_hours'});     
        end
        
        % create mat table to observe experiment length (minutes)
        function [colnames, the_list,exp_table] = mat_table(obj)
            % [colnames, the_list,exp_table] = mat_table(obj)
            % outputs list of file names and duration in minutes
            % get lfp directory
            lfp_dir = dir(fullfile(obj.raw_psd_user,'*.mat'));
            
            % loop through experiments and get exp name and length
            for ii = 1:length(lfp_dir)
                
                % get file path
                Full_path = fullfile(obj.raw_psd_user, lfp_dir(ii).name);
                
                % map the file to memory
                load(Full_path,'power_matrix');
                
                %get length of power matrix
                [~,len] = size(power_matrix);
                
                % create list
                the_list{ii,1} = lfp_dir(ii).name;
                
                if obj.bin_size >0
                    the_list{ii,2} = len*(obj.bin_size/2)/60;
                else
                    the_list{ii,2} = len*(obj.dur/2)/60;
                end
                
            end
            
            colnames = {'Exp-Name', 'Exp_Length_mins'};
            exp_table = cell2table(the_list,'VariableNames', {'Exp_Name' 'Exp_Length_mins'});
        end  
        
        % get times for conditions separated by user
        function get_cond_times(obj,path1)
            % get_cond_times(obj,obj.proc_psd_path)
            
            % get conditions
            prompt = {'Enter conditions ( separated with ; ):'};
            Prompt_title = 'For observation only'; dims = [1 50];
            definput = {'wt_base;wt_veh;wt_allo'};
            answer = inputmod(prompt,Prompt_title,dims,definput);
            
            obj.condition_id = strsplit(answer{1},';');
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(path1,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
          
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
    
    methods % A - fft analysis %
          
        % extract power matrix from mat file (files separated using matlab)
        function extract_pmatrix_mat_user(obj,Flow,Fhigh) 
            %Enter parameters for time-frequency analysis
            %extract power matrix for each experiment and save in save_path
            %.mat file saved from labchart with default setting
            
            
            % prompt user for input
            prompt = {'Enter conditions in sequence (separated by ;)'};
            Prompt_title = 'Input';
            dims = [1 40];
            definput = {'cond1;cond2;cond3'};
            cond_list = inputmod(prompt,Prompt_title,dims,definput);
            
            
            % create analysis folder
            make_analysis_dir(obj)

            % make raw exvivo folder
            obj.raw_psd_user =  fullfile(obj.raw_psd_path,...
                ['raw_psd_user_' obj.channel_struct{obj.channel_No}]); 
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
            obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut); %lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut);%upper boundary
            
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
                while curr_block >= 1 || save_var == 0
                    
                    % get data on desired channel and block
                    output = data(datastart(obj.channel_No,curr_block): dataend(obj.channel_No,curr_block));
                    
                    % obtain power matrix
                    power_matrix = obj.fft_hann(output,obj.winsize,obj.F1,obj.F2,obj.Fs);
                    
                    % get time index
                    [~,L] = size(power_matrix);
                    t =(1:L)*(obj.dur/2) ;%/60; % time in minutes
                    
                    % obtain power area and peak freq for initial analysis and
                    % remove outliers for plotting
                    [~, peak_freq, power_area] = obtain_pmatrix_params(obj,power_matrix,freq,Flow,Fhigh);
                    [power_area,~] = obj.remove_outliers(power_area,3);
                    
                    %%%%Plot power area and peak frequency%%%%
                    figure ()
                    
                    subplot(2,1,1); 
                    
                    plot(t,power_area,'k')
                    ylabel('Power Area (V^2)'); % y-axis label
                    %add baseline and drug time from labchart comments
                    [com_time,Txt_com] = obj.add_comments(obj.Fs,power_area,com,comtext,t,curr_block);%#ok
                    ax1 = gca; obj.prettify_o(ax1);
                    
                    subplot(2,1,2);
                    plot(t,peak_freq,'k')
                    xlabel('Time (Sec)') % x-axis label
                    ylabel('Peak Freq. (Hz)'); % y-axis label
                    ax2 = gca; obj.prettify_o(ax2);title(strrep(lfp_dir(i).name,'_',' '));
                                       
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
                            % start from last block and remove 1 on each loop
                            curr_block = curr_block -1;
                            continue      
                            
                        elseif save_var == 1                           
                            if length(strsplit(user_input{2},';')) ~= length(com_time)
                                % get user input for data analysis
                                disp ('input structure is not correct')
                                continue
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
        
        % consistent time separation across conditions and experiments
        function extract_psd_pharm(obj)
            % plot power spectral density within desired frequencies
            % from Flow to Fhigh % 
            cond_time = (obj.condition_time * 60) / (obj.dur/2);% convert to blocks
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.raw_psd_user,'*.mat'));
            
            % separate conditions into a list
            new_array = obj.cellfromstruct(mat_dir,1);
            cond_list =  obj.filter_list(new_array,obj.condition_id) ;       
            
            % get size for each condition
            [exps, conds] = size(cond_list);
                          
            for i = 1 : exps %loop through experiments
                
                for ii = 1:conds %loop through condiitons
                    
                    if isempty(cond_list{i,ii})==0 % check if file exits
                        
                        % file load
                        load(fullfile(obj.raw_psd_user , cond_list{i,ii}),'power_matrix');
                        % get matrix size
                        [~,len] = size(power_matrix);
                        
                        if cond_time(ii)>len %check if extracted length exceeds bounds
                            cond_list{i,ii}
                        else
                            % get extracted times for each condition
                            if ii == 1
                                power_matrix = power_matrix(:,end - cond_time(ii)+1:end);
                            else
                                power_matrix = power_matrix(:, 1:cond_time(ii));
                            end
                            
                            %save file
                            save(fullfile(obj.raw_psd_path,cond_list{i,ii}),'power_matrix')
                        end
                    end
                end
            end
            
            % save psd_object
            psd_object = saveobj(obj);%#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
        end
        
        % extract power matrix from mat file (files per condition separated)
        function extract_pmatrix_mat(obj)
            
            % create analysis folder
            make_analysis_dir(obj);
            
            % make raw exvivo folder
            obj.raw_psd_user =  fullfile(obj.raw_psd_path,...
                ['raw_psd_user_' obj.channel_struct{obj.channel_No}]); 
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
            
            %save psd_object
            psd_object = saveobj(obj);%#ok
            save(fullfile(obj.save_path,'psd_object.mat'),'psd_object');
        end
        
        % extract power matrix from bin file (files per condition separated)
        function extract_pmatrix_bin(obj)
            % create analysis folder
            make_analysis_dir(obj)
            
            % get lfp directory
            lfp_dir = dir(fullfile(obj.lfp_data_path,'*.adibin'));
            
            % get winsize
            obj.winsize = round(obj.Fs*obj.dur);
            
            % get index values of frequencies
            obj.F1 = obj.getfreq(obj.Fs,obj.winsize,obj.LowFCut); %lower boundary
            obj.F2 = obj.getfreq(obj.Fs,obj.winsize,obj.HighFCut);%upper boundary
            
            % get epoch in seconds
            epoch = obj.period_dur * 60 * obj.Fs;
            
            %initialise progress bar
            progressbar('Total', 'Exp')
            
            % loop through experiments and perform fft analysis
            for ii = 1:length(lfp_dir)

                % get file path
                Full_path = fullfile(obj.lfp_data_path, lfp_dir(ii).name);
                               
                % set data starting point for analysis
                data_start = 0;
                
                %initalise power matrix
                power_matrix = [];
                
            for i = 1:obj.Nperiods % loop across total number of periods
                
                % update data end
                if i == 1
                    data_end = data_start + epoch;  
                else  % get back winsize
                    data_end = data_start + epoch + obj.winsize/2;
                end
                
                % map the whole file to memory
                m = memmapfile(Full_path,'Format','int16');  
                
                %get part of the channel
                OutputChannel = double(m.Data(data_start*obj.Tchannels+obj.channel_No : obj.Tchannels : data_end*obj.Tchannels));
                
                % clear  memmap object
                clear m;
                
                %set correct units to Volts
                OutputChannel = OutputChannel/320000;
                
                % obtain power matrix
                power_matrix_single = obj.fft_hann(OutputChannel,obj.winsize,obj.F1,obj.F2,obj.Fs);
                
                % concatenate power matrix
                power_matrix  = [power_matrix, power_matrix_single];
                
                % update data start 
                data_start = data_start + epoch - obj.winsize/2;
                               
                % update progress bar
                progressbar( [], i/ (obj.Nperiods))
            end
            
            % save power matrix
            save(fullfile(obj.raw_psd_path,erase(lfp_dir(ii).name,'.adibin')),'power_matrix')
            
             % update progress bar
             progressbar( ii/length(lfp_dir), [])
            end
            
            %save psd_object
            psd_object = saveobj(obj);%#ok
            save (fullfile(obj.save_path,'psd_object.mat'),'psd_object')
            
        end
        %%% --------------------------------------------------------------- %%%
         
    end
    
    methods % B - PSD processing
        
        % general psd processing program
        function psd_processing(obj)

            % make processed psd directory
            obj.proc_psd_path = fullfile(obj.save_path,['processed_psd_' obj.channel_struct{obj.channel_No}]);
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
            % [peak_power, peak_freq, power_area] = obtain_pmatrix_params(obj,power_matrix,Flow,Fhigh)
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
                 p_matrix_out(:,i) = obj.remove_noise_psd(power_matrix(:,i),obj.Fs,obj.winsize,obj.noise_var,obj.noisewidth_var);
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
                    p_matrix_out(:,i) =  median(power_matrix(:));
                end
            end

        end

        % linearise PSD --needs completion!!!
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
             
    methods % C - Plots 
        
        % get time vector
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
            x_condition_time(1) = x_condition_time(1) - dt;
        end

        % Spectrogram - subplot for each experiment
        function ret_str = spectrogram_subplot(obj,Flow,Fhigh)
            % spectrogram_plot(obj,Flow,Fhigh)
            % spectrogram subplot within desired frequencies
            % from Flow to Fhigh
            
            if length(obj.condition_id)~= 1
                ret_str = 'Only one condition is allowed. Please re enter conditions';
                return
            else
                ret_str = [];
            end
             % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow); 
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            
            freq = eval(obj.freq_cmd);
            freqx_bound = freq(Flow:Fhigh);
         
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary

            % create figure
            figure()
           
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
     
               for i = 1 : exps %loop through experiments
                   
                   % subplot
                   subplot(ceil(exps/3),3,i)
                   
                   % load file
                   load(fullfile(obj.proc_psd_path , exp_list{i,1}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                   % z normalise
%                    proc_matrix = zscore(proc_matrix);
                   if i == 1
                       [freq, time_bins] = size(proc_matrix);
                       % get time vector in seconds
                       if obj.bin_size~= -1
                           t =(0:time_bins-1)*(obj.bin_size/2);
                           [units,factor] = obj.autoscale_x(length(t)*(obj.bin_size/2)*obj.Fs,obj.Fs);
                           t = t*factor;
                       else
                           t =(0:time_bins-1)*(obj.dur/2);
                           [units,factor] = obj.autoscale_x(length(t)*(obj.dur/2)*obj.Fs,obj.Fs);
                           t = t*factor;
                       end
                   end
                   
                   surf(t,freqx_bound,proc_matrix(Flow:Fhigh,:),'EdgeColor','None');
                   title(erase(strrep(exp_list{i,1},'_',' '),'.mat'))
                   axis1=gca; obj.prettify_o(axis1); colormap jet;
                   colorbar;  axis tight; shading interp;
                   % colormap hsv;% choose coloring scheme
                   view(0,90)% view(20,50);%make 2d
                  
                   
               end
               obj.super_labels([],['Time '  '(' units ')'],'Freq. (Hz)')             
        end
        
        % Spectrogram - separate plot for each experiment
        function ret_str = spectrogram_indplot(obj,Flow,Fhigh)
            % spectrogram_plot(obj,Flow,Fhigh)
            % spectrogram subplot within desired frequencies
            % from Flow to Fhigh
            
            if length(obj.condition_id)~= 1
                ret_str = 'Only one condition is allowed. Please re enter conditions';
                return
            else
                ret_str = [];
            end
             % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow); 
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            
            freq = eval(obj.freq_cmd);
            freqx = freq(Flow:Fhigh);
         
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
               
            % get size for each condition
            [exps, conds] = size(exp_list);
            
     
               for i = 1 : exps %loop through experiments
                   % create figure
                   figure()
                   
                   % load file
                   load(fullfile(obj.proc_psd_path , exp_list{i,1}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                   
                   % z normalise
%                    proc_matrix = zscore(proc_matrix);
                   
                   if i == 1
                       [freq, time_bins] = size(proc_matrix);
                       % get time vector in seconds
                       if obj.bin_size~= -1
                           t =(0:time_bins-1)*(obj.bin_size/2);
                           [units,factor] = obj.autoscale_x(length(t)*(obj.bin_size/2)*obj.Fs,obj.Fs);
                           t = t*factor;
                       else
                           t =(0:time_bins-1)*(obj.dur/2);
                           [units,factor] = obj.autoscale_x(length(t)*(obj.dur/2)*obj.Fs,obj.Fs);
                           t = t*factor;
                       end
                   end
                   
                   surf(t,freqx,proc_matrix(Flow:Fhigh,:),'EdgeColor','None');
                   title(erase(strrep(exp_list{i,1},'_',' '),'.mat'))
                   xlabel(['Time' ' (' units ')']); ylabel('Freq (Hz)');
                   axis1=gca; obj.prettify_o(axis1); colormap jet;
                   colorbar;  axis tight; shading interp;
                   view(0,90) % make 2d  % view(20,50);
                  
                   
               end
                
        end
        
        
        % Plot Aver PSD with SEM - separate figure for each experiment
        function plot_indPSD(obj,Flow,Fhigh)
            % plot_PSD_single(obj,Flow,Fhigh)
            % plot power spectral density within desired frequencies
            % from Flow to Fhigh
            
            % get title string
            ttl_string = [obj.channel_struct{obj.channel_No} ' ' num2str(Flow) ' - ' num2str(Fhigh) ' Hz'];
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow); 
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            
            freq = eval(obj.freq_cmd);
            freqx_bound = freq(Flow:Fhigh);
            
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary        
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            for i = 1 : exps %loop through experiments
                
                % create figure
                figure();hold on;
                legend();
                
                for ii = 1:conds %loop through condiitons
                    
                    % get color vectors
                    [col_mean,col_sem] = obj.color_vec(ii);
 
                    % if file exists
                    if isempty(exp_list{i,ii})==0
                        
                        title_str{ii,i} = erase(strrep(exp_list{i,ii},'_',' '),'.mat');%#ok
                        
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                        
                        %get mean and SEM
                        [~, nperiods] = size(proc_matrix);
                        
                        %get mean and sem
                        mean_wave = mean(proc_matrix(Flow:Fhigh,:),2)';
                        sem_wave = std(proc_matrix(Flow:Fhigh,:),0,2)'/sqrt(nperiods);
                        mean_wave_plus = mean_wave+sem_wave;  mean_wave_minus = mean_wave-sem_wave;
                        
                        %plot mean and shaded sem
                        Xfill= horzcat(freqx_bound, fliplr(freqx_bound));   %#create continuous x value array for plotting
                        Yfill= horzcat(mean_wave_plus, fliplr(mean_wave_minus));
                        fill(Xfill,Yfill,col_sem,'LineStyle','none');
                        plot(freqx_bound,mean_wave,'color', col_mean,'Linewidth',1.5,'DisplayName',title_str{ii,i});                       
                    else
                        title_str{ii,i} = 'NaN';%#ok
                        plot(NaN,'DisplayName',title_str{ii,i})                  
                    end

                end
                
                % prettify and add title
                axis1 = gca;
                xlabel('Freq. (Hz)') ;  ylabel ('Power (V^2 Hz^{-1})')
                title(ttl_string)
                obj.prettify_o(axis1)
            end
            
            
           
        end
        
        % Plot Aver PSD with SEM - subplot for each experiment in one figure
        function plot_subPSD(obj,Flow,Fhigh)
            % plot_PSD_single(obj,Flow,Fhigh)
            % plot power spectral density within desired frequencies
            % from Flow to Fhigh
            
            % get title string
            ttl_string = [obj.channel_struct{obj.channel_No} ' ' num2str(Flow) ' - ' num2str(Fhigh) ' Hz'];
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow); 
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            
            freq = eval(obj.freq_cmd);
            freqx_bound = freq(Flow:Fhigh);
         
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % create figure
            figure('Name',ttl_string);
            obj.super_labels(ttl_string,'Freq. (Hz)','Power (V^2 Hz^{-1})')
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            for ii = 1:conds %loop through condiitons

               % get color vectors
               [col_mean,col_sem] = obj.color_vec(ii);
               
               for i = 1 : exps %loop through experiments
                   % subplot
                   subplot(ceil(exps/3),3,i);hold on;
                   legend();
                   % if file exists
                   if isempty(exp_list{i,ii})==0
                       
                       title_str{ii,i} = erase(strrep(exp_list{i,ii},'_',' '),'.mat');%#ok
                        
                       % load file
                       load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                       
                       %get mean and SEM
                       [~, nperiods] = size(proc_matrix);
                       
                       %get mean and sem
                       mean_wave = mean(proc_matrix(Flow:Fhigh,:),2)';
                       sem_wave = std(proc_matrix(Flow:Fhigh,:),0,2)'/sqrt(nperiods);
                       mean_wave_plus = mean_wave+sem_wave;  mean_wave_minus = mean_wave-sem_wave;
                       
                       %plot mean and shaded sem
                       Xfill= horzcat(freqx_bound, fliplr(freqx_bound));   %#create continuous x value array for plotting
                       Yfill= horzcat(mean_wave_plus, fliplr(mean_wave_minus));
%                        fill(Xfill,Yfill,col_sem,'LineStyle','none');
                       plot(freqx_bound,mean_wave,'color', col_mean,'Linewidth',1.5,'DisplayName',title_str{ii,i});
%                        axis1 = gca;
                       
                       
                   else
                       title_str{ii,i} = 'NaN';%#ok
%                        plot(NaN,'DisplayName',title_str{ii,i})
%                        axis1 = gca;
                   end
                   
                       if ii == conds
%                            xlabel('Freq. (Hz)') ;  ylabel ('Power (V^2 Hz^{-1})')
                           obj.prettify_o(gca)
                           
                       end

               end
            end
           
           
%            sgtitle(ttl_string) 
        end
        
        % Plot Aver PSD with SEM - mean accross experiment
        function plot_meanPSD(obj,Flow,Fhigh)
            % plot_meanPSD(obj,Flow,Fhigh)
            % plot power spectral density within desired frequencies
            % from Flow to Fhigh
            
            % get title string
            ttl_string = [obj.channel_struct{obj.channel_No} ' ' num2str(Flow) ' - ' num2str(Fhigh) ' Hz'];
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
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
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                        
                        % get mean and SEM
                        [~, nperiods] = size(proc_matrix);
                        temp_mean(:,i) = mean(proc_matrix(Flow:Fhigh,:),2)';
                    else
                        disp([exp_list{i,ii} ' is empty'])
                    end
                end 
                
                % get mean and sem
                mean_wave = mean(temp_mean,2)';
                sem_wave = std(temp_mean,0,2)'/sqrt(nperiods);
                mean_wave_plus = mean_wave + sem_wave;  mean_wave_minus = mean_wave-sem_wave;
                
                % plot mean and shaded sem
                Xfill= horzcat(freqx_bound, fliplr(freqx_bound));   %#create continuous x value array for plotting
                Yfill= horzcat(mean_wave_plus, fliplr(mean_wave_minus));
                fill(Xfill,Yfill,col_sem,'LineStyle','none','DisplayName','SEM');
                plot(freqx_bound,mean_wave,'color', col_mean,'Linewidth',1.5,'DisplayName',strrep(obj.condition_id{ii},'_',' ')); % 
                            
            end
            % prettify and add title
            xlabel('Freq. (Hz)') ;  ylabel ('Power (V^2 Hz^{-1})')
            axis1 = gca;obj.prettify_o(axis1)
            title(ttl_string)

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
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
                                
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % remove empty rows
            exp_list(any(cellfun(@isempty, exp_list), 2),:) = []; 
            
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
%                         break
%                     end
                    
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
                
                % save values to matrix for analysis and plotting
%                 if isempty(exp_list{i,ii}) == 0   
                    feature_aver(:,i) = wave_temp;
%                 else
%                     [len,~ ] = size(feature_aver);
%                     feature_aver(:,i) = NaN(len,1);
%                 end
                
            end
            
            % remove rows containg NaNs
            feature_aver = feature_aver(:,all(~isnan(feature_aver)));
           
            % get time
            [units,~,t,x_condition_time] = getcond_realtime(obj,feature_aver);
            
            % get mean and sem as filled
            [mean_wave, xfill,yfill] = obj.getmatrix_mean(feature_aver,t);
            
            % plot individual experiments
            if strct1.ind_v == true
                plot(t,feature_aver,'Color', [0.8 0.8 0.8])
            end
            
            % plot mean
            if strct1.mean_v == true
                fill(xfill,yfill,[0.4 0.4 0.4],'LineStyle','none')
                plot(t,mean_wave,'k','Linewidth',1.5)
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
            % strct1.band1 = 5  low boundary
            % strct1.band2 = 10 high boundary
            % strct1.band_width = 5 high boundary
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
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);     
            
            % remove empty rows
             exp_list(any(cellfun(@isempty, exp_list), 2),:) = []; 
            
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
                        
                        % get peak_power, peak_freq and power_area [psd_prm(1,:), psd_prm(2,:), psd_prm(3,:)]
                        
                        % Band 1
                        [psd_prmA{1}, psd_prmA{2}, psd_prmA{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                            strct1.band1 - strct1.band_width,strct1.band1 + strct1.band_width);
                        % Band 2
                        [psd_prmB{1}, psd_prmB{2}, psd_prmB{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                            strct1.band2 - strct1.band_width,strct1.band2 + strct1.band_width);
                        
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
                ylabel([num2str(strct1.band1 - strct1.band_width) ' - ' num2str(strct1.band1 + strct1.band_width) ' Hz'])
                xlim([t(1) - t(end)/20, t(end)+ t(end)/20])
                ylim([min(feature_aver1(:))- mean(feature_aver1(:))/20, ...
                    max(feature_aver1(:))+ mean(feature_aver1(:))/20 ])
                axis1 = gca; obj.prettify_o(axis1)
                
                subplot(3,1,3);hold on
                fill(xfill2,yfill2,[0.4 0.4 0.4],'LineStyle','none')
                plot(t,mean_wave2,'k','Linewidth',1.5)
                ylabel([num2str(strct1.band2 - strct1.band_width) ' - ' num2str(strct1.band2 + strct1.band_width) ' Hz'])
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
            obj.super_labels([obj.channel_struct{obj.channel_No} ' ' num2str(strct1.band1 ) ' / ' num2str(strct1.band2)  ' Hz'],[],label_y)      
         
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
            

            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
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
            % plot_type ,1  = dot plot, 2 = bar plot, 3 = box plot
            
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
            if conds >1
                extr_feature = extr_feature(:,all(~isnan(extr_feature)));
            end
            
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
            % strct1.band1 = 5  low boundary
            % strct1.band2 = 10 high boundary
            % strct1.band_width = 5 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise 
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            % plot_type ,1  = dot plot, 2 = bar plot, 3 = box plot
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            f_range(1) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1 - strct1.band_width) - obj.F1+1;
            f_range(2) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1 + strct1.band_width) - obj.F1+1;
            f_range(3) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2 - strct1.band_width) - obj.F1+1;
            f_range(4) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2 + strct1.band_width) - obj.F1+1;
            
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

            % remove rows containg NaNs
            if conds >1
                extr_feature = extr_feature(:,all(~isnan(extr_feature)));
            end
            
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
            title([obj.channel_struct{obj.channel_No} ' ' num2str(strct1.band1) ' / ' num2str(strct1.band2) ...
                ' (' num2str(strct1.band_width) ') Hz'])
            
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
    
    methods % D - Export & Statistics
        
        % Export excel table mean psd parameters
        function excel_meanprms(obj,Flow,Fhigh)
            % excel_meanprms(obj,Flow,Fhigh)
            % export data in excel format
            % Flow = 2  low boundary
            % High = 15 high boundary
            
            % create exported folder
            if exist(obj.export_path,'dir') == 0
                obj.export_path = fullfile(obj.save_path,'exported');
                mkdir(obj.export_path)
            end

            
            % get parameters
            psd_prms = {'Peak_Power';'Peak_Freq';'Power_Area'};
            
            % get spreadsheet file name
            xcl_file = [obj.channel_struct{obj.channel_No} '_' num2str(Flow) '_' num2str(Fhigh) ' Hz'];
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            Flow = obj.getfreq(obj.Fs,obj.winsize,Flow);
            Fhigh = obj.getfreq(obj.Fs,obj.winsize,Fhigh);
            freq = eval(obj.freq_cmd); freqx = freq(Flow:Fhigh);
            Flow = Flow - obj.F1+1; % new low boundary
            Fhigh = Fhigh - obj.F1+1; % new high boundary
            
            % get size for each condition
            [exps, conds] = size(exp_list);
            
            % preallocate vector
            extr_features = zeros(exps,conds*length(psd_prms));
            
            for i = 1 : exps %loop through experiments
                cntr=1;
                for ii = 1:conds %loop through conditions
                    
                    if isempty(exp_list{i,ii})==0 % check if file is present
                        % load file
                        load(fullfile(obj.proc_psd_path , exp_list{i,ii}),'proc_matrix'); %struct = rmfield(struct,'power_matrix')
                        
                        % get mean and sem
                        mean_psd = mean(proc_matrix(Flow:Fhigh,:),2);
                        
                        % get parameters from each experiment
                        [psd_prm(1),psd_prm(2),psd_prm(3)] = obj.psd_parameters(mean_psd,freqx);
                        
                        extr_features(i,cntr:cntr-1+length(psd_prms)) = psd_prm;
                    else
                        extr_features(i,cntr:cntr-1+length(psd_prms)) = NaN;
                    end
                    
                    % update counter
                    cntr = cntr + length(psd_prms);
                end
                
            end
            
            %%% Transform data to create an excel table %%%
            
            % convert matrix 2 cell
            extr_features = num2cell(extr_features);
            
            % create empty table name vector
            table_names =['Exp_ID', psd_prms'];
            exp_id = {};
            table_features = {};
            cntr = 1;
            for i = 1: conds
                % get table names
                exp_id = [exp_id; ['Exp_Names_' obj.condition_id{i}]];
                exp_id = [exp_id; exp_list(:,i)];
                exp_id = [exp_id; cell(1,1)];
                
                % get table features
                table_features = [table_features; cell(1,length(psd_prms))];
                table_features = [table_features; extr_features(:,cntr:cntr-1+length(psd_prms))];
                table_features = [table_features; cell(1,length(psd_prms))];
                
                % update counter
                cntr = cntr + length(psd_prms);
            end
            
            % create array for table storage
            tablearray = [exp_id table_features];
             
            % create table
            T = cell2table(tablearray ,'VariableNames',table_names);
            
            % save table to excel file
            writetable(T,fullfile(obj.export_path, xcl_file),'FileType','spreadsheet','Sheet',1)
        end
        
        % Aver PSD parameter
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
    
    methods % E - Correlation
        function corr_psd_prm(obj,Flow,Fhigh)
        end
    end
    
    methods % F - Comparative Plots and statistics
        
        % get matrix with aver PSD parameters vs time
        function [feature_aver, conds] = psd_prm_matrix(obj,strct1)  
            % param_vs_time(obj,strct1)
            % Plot PSD parameters vs time
            % strct1.Flow = 2  low boundary
            % strct1.Fhigh = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise 
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
                                
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % remove empty rows
            exp_list(any(cellfun(@isempty, exp_list), 2),:) = []; 
            
            % get size of condition list
            [exps, conds] = size(exp_list);
            
            % get freq parameters
            freq = eval(obj.freq_cmd); freqx = (freq(obj.F1:obj.F2));
            
            % loop through experiments           
            for i = 1:exps
                
                for ii = 1:conds %concatenate conditions to one vector
%                     if isempty(exp_list{i,ii})==1
%                         break
%                     end
                    
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
                
                % save values to matrix for analysis and plotting
%                 if isempty(exp_list{i,ii}) == 0   
                    feature_aver(:,i) = wave_temp;
%                 else
%                     [len,~ ] = size(feature_aver);
%                     feature_aver(:,i) = NaN(len,1);
%                 end
                
            end
            
            % remove rows containg NaNs
            feature_aver = feature_aver(:,all(~isnan(feature_aver)));        
     
        end 
        
        % get matrix with aver PSD parameter ratio vs time
        function [feature_aver, conds] = prm_prm_ratio_matrix(obj,strct1)  
            % feature_aver = prm_ratio_time(obj,strct1) 
            % Plot PSD parameters vs time
            % strct1.band1 = 5  low boundary
            % strct1.band2 = 10 high boundary
            % strct1.band_width = 5 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise 
            
            % get mat files in load_path directory
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
                                
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);     
            
            % remove empty rows
            exp_list(any(cellfun(@isempty, exp_list), 2),:) = [];
                     
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
                            strct1.band1 - strct1.band_width,strct1.band1 + strct1.band_width);
                        % Band 2
                        [psd_prmB{1}, psd_prmB{2}, psd_prmB{3}] = obtain_pmatrix_params(obj,proc_matrix,freqx,...
                            strct1.band2 - strct1.band_width,strct1.band2 + strct1.band_width);
                        
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
        function extr_feature = aver_psd_prm_matrix(obj,strct1)
            % extr_feature = aver_psd_prm_matrix(obj,strct1)
            % Plot aver psd parameters
            % strct1.Flow = 2  low boundary
            % strct1.High = 15 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise
            
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

            % remove rows containg NaNs
            if conds >1
                extr_feature = extr_feature(:,all(~isnan(extr_feature)));
            end
            
            if  strct1.norms_v == true %normalise
                extr_feature = extr_feature./extr_feature(1,:);
            end
            

        end
        
        % Aver PSD parameter ratio
        function extr_feature = aver_psd_prm_matrix_ratio(obj,strct1)
            % extr_feature = aver_psd_prm_matrix_ratio(obj,strct1)
            % Plot average PSD parameters
            % strct1.band1 = 5  low boundary
            % strct1.band2 = 10 high boundary
            % strct1.band_width = 5 high boundary
            % strct1.par_var = 1 % 1- peak power, 2 - peak freq, 3. power area
            % strct1.norms_v = true % normalise 
            % strct1.ind_v = true % plot individual experiments
            % strct1.mean_v = true % plot mean
            
            % get matlab directory for processed psds
            mat_dir = dir(fullfile(obj.proc_psd_path,'*.mat'));
            
            % get exp list
            exp_list = obj.get_exp_array(mat_dir,obj.condition_id);
            
            % get freq parameters
            f_range(1) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1 - strct1.band_width) - obj.F1+1;
            f_range(2) = obj.getfreq(obj.Fs,obj.winsize,strct1.band1 + strct1.band_width) - obj.F1+1;
            f_range(3) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2 - strct1.band_width) - obj.F1+1;
            f_range(4) = obj.getfreq(obj.Fs,obj.winsize,strct1.band2 + strct1.band_width) - obj.F1+1;
            
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
        
        % Plot between conditions (time)
        function cmp_psd_prm_time(obj,strct1,folder_list,ratio)
            if ratio == false
                % get title string
                ttl_string = [num2str(strct1.Flow) ' - ' num2str(strct1.Fhigh) ' Hz'];
            else
                ttl_string = [num2str(strct1.band1) ' / ' num2str(strct1.band2) ...
                    ' Width (' num2str(strct1.band_width) ') Hz'];
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

                [col_face,col_edge] = spectral_analysis_batch.color_vec2(i);
                
                subplot(length(folder_list),1,i)
                mean_wave = mean(feature_aver,2);
                
                % get time
                [units,~,t,x_condition_time] = getcond_realtime(psd_object,feature_aver);
                
                k = strfind(psd_object.save_path,'\');
                
                h(1) =  plot(t,mean_wave,'o','Color',col_edge,'MarkerEdgeColor',col_edge,'MarkerFaceColor',col_face,...
                    'MarkerSize',5,'DisplayName',strrep(psd_object.save_path(k(end-1)+1:k(end)-1),'_',' '));
                
                hold on;plot(t,smooth_v1(mean_wave,15),'Color',col_edge,'LineWidth',4)
                
                
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
%                 legend(h(1))

            end

        end
        
        % Plot between conditions (average)
        function [bar_vector,pval] = cmp_psd_prm_aver(obj,strct1,folder_list,ratio)

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
                     ttl_string = [psd_object.channel_struct{psd_object.channel_No} ' ' num2str(strct1.band1) ' / ' num2str(strct1.band2)  ' Hz'];
                 end
                 % get color
                 [col_face,col_edge] = spectral_analysis_batch.color_vec2(i);
                 
                 subplot(1,length(folder_list),i)
                 % choose between dot or bar plot
                 obj.dot_plot(extr_feature(2:3,:),psd_object.condition_id(2:3),strct1.ind_v,strct1.mean_v...
                     ,[col_face;col_edge])
                 
                 % format graph
                 obj.prettify_o(gca)
                 
                 bar_vector{i} = extr_feature(3,:) - extr_feature(2,:);
                 [~, pval(i)] = ttest(extr_feature(3,:),extr_feature(2,:));
                 bar_face_col(i,:) = col_face;
                 bar_edge_col(i,:) = col_edge;
                 bar_labels{i} = psd_object.condition_id{3};
                 
             end
            
             
             figure();hold on
             % get mean and SEM
             mean_vec = cellfun(@mean,bar_vector);
             std_vec = cellfun(@std,bar_vector);
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