function [imagesOut bbsOut] = dataPreprocessing( imageIn, bbsIn_all, cropSize, translationShiftNumber, roatateAngleNumber)
% dataPreprocessing function perfroms
% 1) croping
% 2) padding
% 3) rotatation
% 4) shifting
%
% for a input image with a bbs as 4 points, 
% dataPreprocessing outputs a set of images with corresponding bbs.
%
%
% Inputs: 
%   imageIn: input image (480 by 640 by 3)
%   bbsIn: bounding box (2 by 4)
%   cropSize: output image size
%   shift: shifting offset
%   rotate: rotation angle
%
% Outputs:
%   imagesOut: output images (n images)
%   bbsOut: output bbs according to shift and rotation
%
%% created by Fu-Jen Chu on 09/15/2016

debug_dev = 0;
debug = 0;
%% show image and bbs
if(debug_dev)
figure(1); imshow(imageIn); hold on;
x = bbsIn_all(1, [1:3]);
y = bbsIn_all(2, [1:3]);
plot(x,y); hold off;
end

%% crop image and padding image
% cropping to 321 by 321 from center
imgCrop = imcrop(imageIn, [145 65 351 351]); 

% padding to 501 by 501
imgPadding = padarray(imgCrop, [75 75], 'replicate', 'both');

count = 1;
for i_rotate = 1:roatateAngleNumber*translationShiftNumber*translationShiftNumber
    % random roatateAngle
    theta = randi(360)-1;
    %theta = 0;

    % random translationShift
    dx = randi(101)-51;
    %dx = 0;

    %% rotation and shifting
    % random translationShift
    dy = randi(101)-51;
    %dy = 0;

    imgRotate = imrotate(imgPadding, theta);
    if(debug_dev)figure(2); imshow(imgRotate);end
    imgCropRotate = imcrop(imgRotate, [size(imgRotate,1)/2-160-dx size(imgRotate,1)/2-160-dy 320 320]);
    if(debug_dev)figure(3); imshow(imgCropRotate);end 
    imgResize = imresize(imgCropRotate, [cropSize cropSize]);
    if(debug)figure(4); imshow(imgResize); hold on;end

    %% modify bbs
    [m, n] = size(bbsIn_all);
    bbsNum = n/4;
    
    countbbs = 1;
    for idx = 1:bbsNum
      bbsIn =  bbsIn_all(:,idx*4-3:idx*4); 
      if(sum(sum(isnan(bbsIn)))) continue; end
      
      bbsInShift = bbsIn - repmat([320; 240], 1, 4);
      R = [cos(theta/180*pi) -sin(theta/180*pi); sin(theta/180*pi) cos(theta/180*pi)];
      bbsRotated = (bbsInShift'*R)';
      bbsInShiftBack = (bbsRotated + repmat([160; 160], 1, 4) + repmat([dx; dy], 1, 4))*cropSize/320;
      if(debug)
        figure(4)
        x = bbsInShiftBack(1, [1:4 1]);
        y = bbsInShiftBack(2, [1:4 1]);
        plot(x,y); hold on; pause(0.01);
      end
      bbsOut{count}{countbbs} = bbsInShiftBack;
      countbbs = countbbs + 1;
    end
    
    imagesOut{count} = imgResize;
    count = count +1;

   
end



end
