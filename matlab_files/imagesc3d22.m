function iiiiiiiim = imagesc3d22(img, title_fig, rot, range, pos)
if nargin < 3
    rot = false;
    range = [];
    pos = round(size(img)/2) + 1;
end
if nargin < 4
    range = [];
    pos = round(size(img)/2) + 1;
end
if nargin < 5
    pos = round(size(img)/2) + 1;
end

% pos(3) = 33;
% pos(2) = 288-162;
% pos(1) = 136;


figure;
if rot == false
    subplot(131), imshow(squeeze(img(pos(1),:,:)), range)
    subplot(132), imshow(squeeze(img(:,pos(2),:)), range)
    subplot(133), imshow(img(:,:,pos(3)), range)
else
    subplot(131), imshow(imrotate(squeeze(img(pos(1),:,:)), rot(1)), range)
    iiiiiiiim = imrotate(squeeze(img(pos(1),:,:)), rot(1));
    subplot(132), imshow(imrotate(squeeze(img(:,pos(2),:)), rot(2)), range)
    subplot(133), imshow(imrotate(img(:,:,pos(3)), rot(3)), range)
end
l = sgtitle(title_fig);
set(l, 'FontSize', 40, 'Interpreter', 'latex')
drawnow;
end


