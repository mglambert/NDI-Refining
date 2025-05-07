function im_final = imagen_papers2(imgs, voxel_size, rot, range, orientation, save, nombre)
if nargin < 3
    rot = [90,90,90];
    range = [-0.1, 0.1];
    orientation = 'vertical';
    save = 0;
    nombre = '';
end
if nargin < 4
    range = [-0.1, 0.1];
    orientation = 'vertical';
    save = 0;
    nombre = '';
end
if nargin < 5
    orientation = 'vertical';
    save = 0;
    nombre = '';
end

pos = round(size(imgs{1})/2) + 1;
% pos(3) = pos(3) - 3;
% pos(1) = pos(1) + 1;

% % pos = [176,172,69];
% pos(3) = 68;
% pos(3) = 73;
% pos(1) = 136;




im_final = [];

for i = 1:length(imgs)

    im = [];
    img = imgs{i};

    im1 = squeeze(img(pos(1),:,:));
    im1 = imresize(im1, size(im1)./voxel_size(2:end));
    im1 = imrotate(im1, rot(1));

    im2 = squeeze(img(:, pos(2),:));
    im2 = imresize(im2, size(im2)./voxel_size([1, 3]));
    im2 = imrotate(im2, rot(2));

    im3 = squeeze(img(:, :, pos(3)));
    im3 = imresize(im3, size(im3)./voxel_size(1:2));
    im3 = imrotate(im3, rot(3));

    if strcmp(orientation, 'vertical')
        col = max([size(im1, 2), size(im2, 2), size(im3, 2)]);
        
        [ii, ~] = find(im1~=0);
        im1 = im1(min(ii):max(ii), :);
        aux = zeros([size(im1, 1), col]);
        a = (col - size(im1, 2))/2;
        aux(:, (1+floor(a)):(col-ceil(a))) = im1;
%         aux = aux(16-15:16+15, 75-15:75+15);
%         ry = 16-15:16+15;
%         rx = 75-15:75+15;
%         for index = 1:length(ry)
%             if index == 1 || index == length(ry)
%                 for idx = rx
%                     aux(ry(index), idx) = 100;
%                 end
%             else
%                 for idx = rx([1, end])
%                     aux(ry(index), idx) = 100;
%                 end
%             end
%         end
%         
        
        im = [im; aux];



        [ii, ~] = find(im2~=0);
        im2 = im2(min(ii):max(ii), :);
        aux = zeros([size(im2, 1), col]);
        a = (col - size(im2, 2))/2;
        aux(:, (1+floor(a)):(col-ceil(a))) = im2;
%         aux = aux(26-15:26+15, 105-15:105+15);
%         ry = 26-15:26+15;
%         rx = 105-15:105+15;
%         for index = 1:length(ry)
%             if index == 1 || index == length(ry)
%                 for idx = rx
%                     aux(ry(index), idx) = 100;
%                 end
%             else
%                 for idx = rx([1, end])
%                     aux(ry(index), idx) = 100;
%                 end
%             end
%         end
%         im = [im; aux];


        [ii, ~] = find(im3~=0);
        im3 = im3(min(ii):max(ii), :);
        aux = zeros([size(im3, 1), col]);
        a = (col - size(im3, 2))/2;
        aux(:, (1+floor(a)):(col-ceil(a))) = im3;
%         aux = aux(43-15:43+15, 78-15:78+15);
%         ry = 43-15:43+15;
%         rx = 78-15:78+15;
%         for index = 1:length(ry)
%             if index == 1 || index == length(ry)
%                 for idx = rx
%                     aux(ry(index), idx) = 100;
%                 end
%             else
%                 for idx = rx([1, end])
%                     aux(ry(index), idx) = 100;
%                 end
%             end
%         end
%         im = [im; aux];
        
        [~, jj] = find(im~=0);
        im = im(:, min(jj):max(jj));
        
        im_final = [im_final im];

    elseif strcmp(orientation, 'horizontal')
        row = max([size(im1, 1), size(im2, 1), size(im3, 1)]);
        [~, jj] = find(im1~=0);
        im1 = im1(:, min(jj):max(jj));
        aux = zeros([row, size(im1, 2)]);
        a = (row - size(im1, 1))/2;
        aux((1+floor(a)):(row-ceil(a)), :) = im1;
        im = [im aux];
       

        [~, jj] = find(im2~=0);
        im2 = im2(:, min(jj):max(jj));
        aux = zeros([row, size(im2, 2)]);
        a = (row - size(im2, 1))/2;
        aux((1+floor(a)):(row-ceil(a)), :) = im2;
        im = [im aux];


        [~, jj] = find(im3~=0);
        im3 = im3(:, min(jj):max(jj));
        aux = zeros([row, size(im3, 2)]);
        a = (row - size(im3, 1))/2;
        aux((1+floor(a)):(row-ceil(a)), :) = im3;
        im = [im aux];
        
        [ii, ~] = find(im~=0);
        im = im(min(ii):max(ii), :);
        
        im_final = [im_final; im];
    end
end

figure;
imshow(im_final, range);

if save
    im_final = imresize(im_final, size(im_final)*3, 'bilinear');
    im_final(im_final > range(2)) = range(2);
    im_final(im_final < range(1)) = range(1);
    im_final = im_final - range(1);
    im_final = im_final/(2*range(2));
    im_final = uint8(255 * im_final);
    imwrite(im_final, [nombre, '.png']);
end

end
