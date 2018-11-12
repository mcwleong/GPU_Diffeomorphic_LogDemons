function saveToBin(var,filename)


FileID = fopen(filename,'w');
%size header
fwrite(FileID, size(var,1), 'short');
fwrite(FileID, size(var,2), 'short');
fwrite(FileID, size(var,3), 'short');

fwrite(FileID, var, 'float32');

fclose(FileID);


end