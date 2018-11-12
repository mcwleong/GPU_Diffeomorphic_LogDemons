function data = loadbin(filename)
    fileID = fopen(filename);

    dim = fread(fileID,3, 'short');
    data = fread(fileID,prod(dim),'float32');
    fclose('all');
    data = reshape(data,  dim(1), dim(2), dim(3));
    
    %data = data(:,:,100);
end