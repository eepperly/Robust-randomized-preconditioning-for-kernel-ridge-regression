function folderpath = createFolderForExecution(name)
    date = datestr(now, 'dd-mm-yy_HH_MM');
    folderpath = fullfile('../results', name, string(date));
    mkdir(folderpath);
end