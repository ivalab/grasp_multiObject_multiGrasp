%% script to test dataPreprocessing
%% created by Fu-Jen Chu on 09/15/2016

close all
clear

%parpool(4)
addpath('/media/fujenchu/home3/data/grasps/')


% generate list for splits
list = [100:949 1000:1034];
list_idx = randperm(length(list));
train_list_idx = list_idx(length(list)/5+1:end);
test_list_idx = list_idx(1:length(list)/5);
train_list = list(train_list_idx);
test_list = list(test_list_idx);


for folder = 1:10
display(['processing folder ' int2str(folder)])

imgDataDir = ['/home/fujenchu/projects/deepLearning/tensorflow-finetune-flickr-style-master/data/grasps/' sprintf('%02d',folder) '_rgd'];
txtDataDir = ['/home/fujenchu/projects/deepLearning/tensorflow-finetune-flickr-style-master/data/grasps/' sprintf('%02d',folder)];

%imgDataOutDir = ['/home/fujenchu/projects/deepLearning/tensorflow-finetune-flickr-style-master/data/grasps/' sprintf('%02d',folder) '_Cropped320_rgd'];
imgDataOutDir = '/media/fujenchu/home3/fasterrcnn_grasp/rgd_multibbs_5_5_5_tf/data/Images';
annotationDataOutDir = '/media/fujenchu/home3/fasterrcnn_grasp/rgd_multibbs_5_5_5_tf/data/Annotations';
imgSetTrain = '/media/fujenchu/home3/fasterrcnn_grasp/rgd_multibbs_5_5_5_tf/data/ImageSets/train.txt'; 
imgSetTest = '/media/fujenchu/home3/fasterrcnn_grasp/rgd_multibbs_5_5_5_tf/data/ImageSets/test.txt'; 

imgFiles = dir([imgDataDir '/*.png']);
txtFiles = dir([txtDataDir '/*pos.txt']);

logfileID = fopen('log.txt','a');
mainfileID = fopen(['/home/fujenchu/projects/deepLearning/deepGraspExtensiveOffline/data/grasps/scripts/trainttt' sprintf('%02d',folder) '.txt'],'a');
for idx = 1:length(imgFiles) 
    %% display progress
    tic
    display(['processing folder: ' sprintf('%02d',folder) ', imgFiles: ' int2str(idx)])
    
    %% reading data
    imgName = imgFiles(idx).name;
    [pathstr,imgname] = fileparts(imgName);
    
    filenum = str2num(imgname(4:7));
    if(any(test_list == filenum))
        file_writeID = fopen(imgSetTest,'a');
        fprintf(file_writeID, '%s\n', [imgDataDir(1:end-3) 'Cropped320_rgd/' imgname '_preprocessed_1.png' ] );
        fclose(file_writeID);
        continue;
    end
    
    txtName = txtFiles(idx).name;
    [pathstr,txtname] = fileparts(txtName);

    img = imread([imgDataDir '/' imgname '.png']);
    fileID = fopen([txtDataDir '/' txtname '.txt'],'r');
    sizeA = [2 100];
    bbsIn_all = fscanf(fileID, '%f %f', sizeA);
    fclose(fileID);
    
    %% data pre-processing
    [imagesOut bbsOut] = dataPreprocessing_fasterrcnn(img, bbsIn_all, 227, 5, 5);
    
    % for each augmented image
    for i = 1:1:size(imagesOut,2)
        
        % for each bbs
        file_writeID = fopen([annotationDataOutDir '/' imgname '_preprocessed_' int2str(i) '.txt'],'w');
        printCount = 0;
        for ibbs = 1:1:size(bbsOut{i},2)
          A = bbsOut{i}{ibbs};  
          xy_ctr = sum(A,2)/4; x_ctr = xy_ctr(1); y_ctr = xy_ctr(2);
          width = sqrt(sum((A(:,1) - A(:,2)).^2)); height = sqrt(sum((A(:,2) - A(:,3)).^2));
          if(A(1,1) > A(1,2))
              theta = atan((A(2,2)-A(2,1))/(A(1,1)-A(1,2)));
          else
              theta = atan((A(2,1)-A(2,2))/(A(1,2)-A(1,1))); % note y is facing down
          end  
    
          % process to fasterrcnn
          x_min = x_ctr - width/2; x_max = x_ctr + width/2;
          y_min = y_ctr - height/2; y_max = y_ctr + height/2;
          %if(x_min < 0 || y_min < 0 || x_max > 227 || y_max > 227) display('yoooooooo'); end
          if((x_min < 0 && x_max < 0) || (y_min > 227 && y_max > 227) || (x_min > 227 && x_max > 227) || (y_min < 0 && y_max < 0)) display('xxxxxxxxx'); break; end
          cls = round((theta/pi*180+90)/10) + 1;
          
          % write as lefttop rightdown, Xmin Ymin Xmax Ymax, ex: 261 109 511 705  (x水平 y垂直)
          fprintf(file_writeID, '%d %f %f %f %f\n', cls, x_min, y_min, x_max, y_max );   
          printCount = printCount+1;
        end
        if(printCount == 0) fprintf(logfileID, '%s\n', [imgname '_preprocessed_' int2str(i) ]);end
        
        fclose(file_writeID);
        imwrite(imagesOut{i}, [imgDataOutDir '/' imgname '_preprocessed_' int2str(i) '.png']); 
        
        % write filename to imageSet 
        file_writeID = fopen(imgSetTrain,'a');
        fprintf(file_writeID, '%s\n', [imgname '_preprocessed_' int2str(i) ] );
        fclose(file_writeID);

    end
    

    toc
end
fclose(mainfileID);

end
