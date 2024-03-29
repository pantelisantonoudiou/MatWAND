% get file path
% -> file_path = 'C:\Users\panton01\Desktop\paper_data\sgGabrd\raw_data_bla';
% -> add_unique_id(file_path,'.mat')
function add_unique_id(file_path, ext)
% add_unique_id(file_path,ext)
% add_unique_id(file_path,'.mat') or add_unique_id([],'.mat'); add_unique_id([],'.adibin')

if  isempty(file_path) % if file path get user dir
    file_path = uigetdir();
end
% get file list
file_list = dir(fullfile(file_path, ['*', ext]));

disp('Renaming...')
disp('_____________________________________')

for i = 1:length(file_list) % loop through files

    temp_file = erase(file_list(i).name, ext); % remove ending
    k =  strfind(temp_file,'_'); % find underscore position
    
    if length(k) == 0 %#ok no underscores
        new_name = horzcat(temp_file, ['_', num2str(i), '_'], ext);
    elseif length(k) == 1 % one underscore for condition
        new_name = horzcat(temp_file(1:k), [num2str(i), '_'], temp_file(k+1:end) ,ext);
    elseif length(k) > 1
        new_name = 'More than one underscore detected: File name will not be changed.';
    end
    
    if contains(new_name, ext) % rename file
        movefile(fullfile(file_path,file_list(i).name),fullfile(file_path,new_name))
    end
    fprintf('from %s -----> %s \n',file_list(i).name,new_name)

end
disp('Files have been renamed with unique IDs.')
end

