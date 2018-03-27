close all;
clear all;
clc;

%-------
% Paths.
%-------
addpath('modelspecific');
addpath('mexfiles');
addpath('multigs');

%-------------------
% Compile Mex files.
%-------------------
cd multigs;
if exist('computeIntersection','file')~=3
    mex computeIntersection.c; % <-- for multigs
end
cd ..;

cd mexfiles;
if exist('imagewarping','file')~=3
    mex ../imagewarping.cpp; 
end
if exist('wsvd','file')~=3
    mex ../wsvd.cpp; % We make use of eigen3's SVD in this file.
end
cd ..;

%----------------------
% Setup VLFeat toolbox.
%----------------------
cd vlfeat-0.9.14/toolbox;
feval('vl_setup');
cd ../..;

%---------------------------------------------
% Check if we are already running in parallel.
%---------------------------------------------
poolsize = parpool('local');
if poolsize == 0 %if not, we attempt to do it:
    parpool open;
end

%-------------------------
% User defined parameters.
%-------------------------
% Global model specific function handlers.
clear global;
global fitfn resfn degenfn psize numpar
fitfn = 'homography_fit';
resfn = 'homography_res';
degenfn = 'homography_degen';
psize   = 4;
numpar  = 9;

M     = 500;  % Number of hypotheses for RANSAC.
thr   = 0.1;  % RANSAC threshold.

C1 = 100; % Resolution/grid-size for the mapping function in MDLT (C1 x C2).
C2 = 100;

%%%%%%%%%%%%%%%%%%%
% *** IMPORTANT ***
%%%%%%%%%%%%%%%%%%%
% If you want to try with your own images and make use of the VLFEAT
% library for SIFT keypoint detection and matching, **comment** the 
% previous IF/ELSE STATEMENT and **uncomment** the following code:
 
 gamma = 0.1; % Normalizer for Moving DLT. (0.0015-0.1 are usually good numbers).
 sigma = 8.5;  % Bandwidth for Moving DLT. (Between 8-12 are good numbers).   
 scale = 1;    % Scale of input images (maybe for large images you would like to use a smaller scale).
 
% %------------------
% % Images to stitch.
% %------------------
 path1 = 'output_images1';
 path2 = 'output_images2';
 outputPath1 = 'output_images1\'; 
 outputPath2 = 'output_images2\'; 
 output = 'video';
 fileName = '1.avi'; 
 obj = VideoReader(fileName);
 numFrames = obj.NumberOfFrames;% 帧的总数
 for k = 1 : numFrames% 读取数据
     frame = read(obj,k);
     %imshow(frame);%显示帧
     imwrite(frame,strcat(sprintf('%s',outputPath1),num2str(k),'.jpg'),'jpg');% 保存帧
 end
 fprintf('> Finished!.\n');
 fileName2 = '2.avi'; 
 obj2 = VideoReader(fileName2);
 numFrames = obj2.NumberOfFrames;% 帧的总数
 for k = 1 : numFrames% 读取数据
     frame = read(obj2,k);
     %imshow(frame);%显示帧
     imwrite(frame,strcat(sprintf('%s',outputPath2),num2str(k),'_.jpg'),'jpg');% 保存帧
 end 
 fprintf('> Finished!.\n');
 if obj.NumberOfFrames<obj2.NumberOfFrames
     number = obj.NumberOfFrames;
 else
     number = obj2.NumberOfFrames;
 end
 CH = 100000;
 CW = 200000;
 
 for nnumber = 1:number
    % %-------------
    % % Read images.
    % %-------------
    fprintf('Read images and SIFT matching\n');tic;
    fprintf('> Reading images...');tic;
    img1 = imresize(imread(strcat((sprintf('%s',path1)),'\',sprintf('%d',nnumber),'.jpg')),scale);
    img2 = imresize(imread(strcat((sprintf('%s',path2)),'\',sprintf('%d',nnumber),'_.jpg')),scale);
    fprintf('done (%fs)\n',toc);
 
    % %--------------------------------------
    % % SIFT keypoint detection and matching.
    % %--------------------------------------
    fprintf('  Keypoint detection and matching...');tic;
    [ kp1,ds1 ] = vl_sift(single(rgb2gray(img1)),'PeakThresh', 0,'edgethresh',500);
    [ kp2,ds2 ] = vl_sift(single(rgb2gray(img2)),'PeakThresh', 0,'edgethresh',500);
    matches   = vl_ubcmatch(ds1,ds2);
    fprintf('done (%fs)\n',toc);

    % Normalise point distribution.
    fprintf('  Normalising point distribution...');tic;
    data_orig = [ kp1(1:2,matches(1,:)) ; ones(1,size(matches,2)) ; kp2(1:2,matches(2,:)) ; ones(1,size(matches,2)) ];
    [ dat_norm_img1,T1 ] = normalise2dpts(data_orig(1:3,:));
    [ dat_norm_img2,T2 ] = normalise2dpts(data_orig(4:6,:));
    data_norm = [ dat_norm_img1 ; dat_norm_img2 ];
    fprintf('done (%fs)\n',toc);

    %if size(img1,1) == size(img2,1)    
        % Show input images.
     %   fprintf('  Showing input images...');tic;
     %   figure;
      %  imshow([img1,img2]);
      %  title('Input images');
      %  fprintf('done (%fs)\n',toc);
    %end

    %-----------------
    % Outlier removal.
    %-----------------
    fprintf('Outlier removal\n');tic;
    % Multi-GS
    rng(0);
    [ ~,res,~,~ ] = multigsSampling(100,data_norm,M,10);
    con = sum(res<=thr);
    [ ~, maxinx ] = max(con);
    inliers = find(res(:,maxinx)<=thr);
    
 

    %-----------------------
    % Global homography (H).
    %-----------------------
    fprintf('DLT (projective transform) on inliers\n');
    %  Refine homography using DLT on inliers.
    fprintf('> Refining homography (H) using DLT...');tic;
    [ h,A,D1,D2 ] = feval(fitfn,data_norm(:,inliers));
    Hg = T2\(reshape(h,3,3)*T1);
    fprintf('done (%fs)\n',toc);

    %----------------------------------------------------
    % Obtaining size of canvas (using global Homography).
    %----------------------------------------------------
    fprintf('Canvas size and offset (using global Homography)\n');
    fprintf('> Getting canvas size...');tic;
    % Map four corners of the right image.
    TL = Hg\[1;1;1];
    TL = round([ TL(1)/TL(3) ; TL(2)/TL(3) ]);
    BL = Hg\[1;size(img2,1);1];
    BL = round([ BL(1)/BL(3) ; BL(2)/BL(3) ]);
    TR = Hg\[size(img2,2);1;1];
    TR = round([ TR(1)/TR(3) ; TR(2)/TR(3) ]);
    BR = Hg\[size(img2,2);size(img2,1);1];
    BR = round([ BR(1)/BR(3) ; BR(2)/BR(3) ]);

    % Canvas size.
    cw = max([1 size(img1,2) TL(1) BL(1) TR(1) BR(1)]) - min([1 size(img1,2) TL(1) BL(1) TR(1) BR(1)]) + 1;
    ch = max([1 size(img1,1) TL(2) BL(2) TR(2) BR(2)]) - min([1 size(img1,1) TL(2) BL(2) TR(2) BR(2)]) + 1;
    if cw<CW
        CW = cw;
    end
    if ch<CH
        CH = ch;
    end
    fprintf('done (%fs)\n',toc);

    % Offset for left image.
    fprintf('> Getting offset...');tic;
    off = [ 1 - min([1 size(img1,2) TL(1) BL(1) TR(1) BR(1)]) + 1 ; 1 - min([1 size(img1,1) TL(2) BL(2) TR(2) BR(2)]) + 1 ];
    fprintf('done (%fs)\n',toc);

    %-------------------------
    % Moving DLT (projective).
    %-------------------------
    fprintf('As-Projective-As-Possible Moving DLT on inliers\n');

    % Image keypoints coordinates.
    Kp = [data_orig(1,inliers)' data_orig(2,inliers)'];

    % Generating mesh for MDLT.
    fprintf('> Generating mesh for MDLT...');tic;
    [ X,Y ] = meshgrid(linspace(1,cw,C1),linspace(1,ch,C2));
    fprintf('done (%fs)\n',toc);

    % Mesh (cells) vertices' coordinates.
    Mv = [X(:)-off(1), Y(:)-off(2)];

    % Perform Moving DLT
    fprintf('  Moving DLT main loop...');tic;
    Hmdlt = zeros(size(Mv,1),9);
    parfor i=1:size(Mv,1)
    
        % Obtain kernel    
        Gki = exp(-pdist2(Mv(i,:),Kp)./sigma^2);   

        % Capping/offsetting kernel
        Wi = max(gamma,Gki); 
    
        % This function receives W and A and obtains the least significant 
        % right singular vector of W*A by means of SVD on WA (Weighted SVD).
        v = wsvd(Wi,A);
        h = reshape(v,3,3)';        
    
        % De-condition
        h = D2\h*D1;

        % De-normalize
        h = T2\h*T1;
    
        Hmdlt(i,:) = h(:);
    end
    fprintf('done (%fs)\n',toc);

%---------------------------------
% Image stitching with Moving DLT.
%---------------------------------
    fprintf('As-Projective-As-Possible Image stitching with Moving DLT and linear blending\n');
% Warping images with Moving DLT.
    fprintf('> Warping images with Moving DLT...');tic;
    warped_img1 = uint8(zeros(ch,cw,3));
    warped_img1(off(2):(off(2)+size(img1,1)-1),off(1):(off(1)+size(img1,2)-1),:) = img1;
    [warped_img2] = imagewarping(double(ch),double(cw),double(img2),Hmdlt,double(off),X(1,:),Y(:,1)');
    warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
    fprintf('done (%fs)\n',toc);

% Blending images by averaging (linear blending)
    fprintf('  Moving DLT linear image blending (averaging)...');tic;
    linear_mdlt = imageblending(warped_img1,warped_img2);
    fprintf('done (%fs)\n',toc);
%figure;
%imshow(linear_mdlt);
    fileName=sprintf('%d',nnumber); 
    imwrite(linear_mdlt,strcat(sprintf('%s',output),'\',fileName,'.jpg'));
%title('As-Projective-As-Possible Image Stitching with Moving DLT');
 end
 framesPath = 'video';       %图像序列所在路径，同时要保证图像大小相同 ，如果不同，可用 resize()
    videoName = 'output.avi';                     %表示将要创建的视频文件的名字  
      fps = obj.FrameRate;                                          %帧率  
    startFrame = 1;                                %从哪一帧开始 (玩爱剪辑的IT朋友，好好看)
    endFrame = number;                             %哪一帧结束    
                  %生成视频的参数设定  
    aviobj=VideoWriter(videoName);       %创建一个avi视频文件对象，开始时其为空  
    aviobj.FrameRate=fps;   
      open(aviobj);                                  %打开文件写入视频数据
                 %读入图片  
        for  i=startFrame:endFrame    % for循环，从哪一帧到哪一帧
            fileName=sprintf('%d',i);      %根据文件名而定 
            frames=imresize(imread(strcat((sprintf('%s',framesPath)),'\',fileName,'.jpg')),[1080,1920]);  
            writeVideo(aviobj,frames);  
        end  
    close(aviobj);                         % 关闭创建视频
    fprintf('> Finished!.\n');
