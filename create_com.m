function create_com()

% user input
prompt={'Number of Channels:','Sample Rate"','Unique Comments:','Comment Times (Mins):','Folder Path'};
name='Recreate labchart comments';
numlines=1;
defaultanswer={'2','4000','veh;drug','127;236','C:\user\Desktop\eeg_1\raw_data'};

% get answer
answer = inputmod(prompt,name,numlines,defaultanswer);
if isempty(answer)==1
    return
end

% set main path to folder
main_path = answer{5};

% get parameters to correct format
ch_num = str2double(answer{1});
com_times = str2num(answer{4});
comtext = char(split(answer{3},';'));

% set samplerate
samplerate = ones(1,ch_num) * str2double(answer{2});

% pre allocate arrays
com = ones(ch_num,5); com(:,3) = com_times*samplerate(1)*60;
datastart = zeros(ch_num,1);
dataend = zeros(ch_num,1);

% get files
file_dir = dir(fullfile(main_path,'*.adibin'));

for ii = 1:length(file_dir)
% load files
s = memmapfile(fullfile(main_path,file_dir(ii).name),'Format','single');

% get file length
file_len = length(s.Data);
clear s;

datastart(1) = 1;
dataend(1) = file_len/ch_num;

for i = 1:ch_num-1   
    datastart(i+1) = dataend(i) + 1;
    dataend(i+1) = datastart(i+1) + file_len/ch_num -1;
end

% save matlab file
save(fullfile(main_path,strrep(file_dir(ii).name,'adibin','mat')),'samplerate','datastart','dataend','com','comtext')

end



