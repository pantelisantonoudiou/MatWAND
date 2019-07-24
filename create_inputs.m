classdef create_inputs < matlab.mixin.Copyable
    
    properties
        Fs = 1000; % sampling rate      
        desktop_path;
        gen_path = 'C:\Users\panton01\Desktop\INvivo_data\unit_test'; % path to store new files
        save_path
        test_name = 'SineTest'     
        cond = {'cond1';'cond2'};
    end
    
    methods
        
        % class constructor
        function obj = create_inputs()
            % get path to dekstop
            obj.desktop_path = cd;
            obj.desktop_path = obj.desktop_path(1:strfind( obj.desktop_path,'Documents')-1);
            obj.desktop_path = fullfile( obj.desktop_path,'Desktop\');
        end
        
        % create sine_wave
        function sine_wave = sine_gen(obj,amp,freq,randvar,mins)
            % sine_wave = sine_gen(obj,amp,freq,randvar,mins)
            % sine_wave = sine_gen(x,1,10,5,1)
            % freq = frequency of sine wave
            % mins = length of generated wave in mins
            
            T = 1/obj.Fs;             % Sampling period
            L = obj.Fs*60*mins;             % Length of signal
            t = (0:T:(L-1)/obj.Fs);        % Time vector
            
            sine_wave = amp*sin(2*pi*freq*t) + randvar * rand(1,L);
            
        end
        
        % create triangular wave
        function saw_wave = saw_gen(obj,freq,mins,randvar)
            % saw_wave = saw_gen(obj,freq,mins,randvar)
            % sine_wave = sine_gen(x,10,5,1)
            % freq = frequency of saw wave
            % mins = length of generated wave in mins
            
            T = 1/obj.Fs;             % Sampling period
            L = obj.Fs*60*mins;             % Length of signal
            t = (0:T:(L-1)/obj.Fs);        % Time vector
            
            saw_wave = sawtooth(2*pi*freq*t,0.25) + randvar * rand(1,L);
            
        end
        
        % save batch waves
        function batch_wave(obj,cond_name,freq,amp)
                    
            % create analysis folder
            obj.save_path = fullfile(obj.gen_path, 'raw_data');
            
            if ~exist(obj.save_path, 'dir')
                mkdir(obj.save_path)
            end
            
            %initialise progress bar
            progressbar('Progress')
            
            file_num = 5;% number of files
            epochs = 30; % in minutes
            epoch_size = 1; % in minutes         
           
            for ii = 1:file_num % create multiple files

                % set parameters
                data = [];
                noise = 0;

                for i = 1 : epochs
                    % normrnd(freq,5,1)
                    % normrnd(amp,1,1)
                    temp_wave = sine_gen(obj,amp,freq,noise,epoch_size);
                    data = [data temp_wave]; %#ok
%                     freq = freq -5;
%                     amp = amp - 0.2;
                end
                
                % save files
                samplerate = obj.Fs; %#ok
                datastart = 1; %#ok
                dataend = length(data); %#ok
                save(fullfile(obj.save_path,[obj.test_name '_' num2str(ii) '_' cond_name '.mat']),'data','samplerate','datastart','dataend')
                
                % update progressbar
                progressbar(ii/file_num)
            end
            
        end
        
    end
    
    
end