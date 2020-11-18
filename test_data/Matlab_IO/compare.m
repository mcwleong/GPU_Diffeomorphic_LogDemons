function compare( var1, var2, slice )
if (nargin <3) slice = 70; end;
var1 = getslice(var1, slice);
var2 = getslice(var2, slice);
%var1, var2 should be 2d images
var1 = var1(1:end, 1:end);
var2 = var2(1:end, 1:end);

figure
subplot(1,3,1)
imagesc(var1);
subplot(1,3,2)
imagesc(var2);
subplot(1,3,3)
imagesc(var2-var1);

end

