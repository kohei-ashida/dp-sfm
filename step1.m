
function foldername = step1(imgname, image_resize_val, stride, patch_size, ker_size, border)
    addpath(genpath('functions'));
    select_folder = 'Qualitative'; % input folder
    save_path = 'Results_qualitative'; % results folder
    save_path = 'blur_results';
    ext = "JPG";
;

%    imgname = '0001_B.png'; % selected image

                            % should end with '_B.png'
    % Note:
    % [imgname(1:end-6) '_L.png'] and [imgname(1:end-6) '_R.png'] are expected
    % to be present inside select_folder

    fig_flag=0; %0 -> no figure pop ups indicating progress

%    image_resize_val = 0.5; % downsample the image by this factor
                            % downsampling (or resampling) is generally NOT recommended
                            % since this can destroy dual-pixel disparity
                            % information. However, optimization applied
                            % patch-wise is very slow. This is a running time
                            % versus accuracy trade-off.
%    image_resize_val =1;
        
    %patch_size = 111; %111; ALWAYS ODD, after resizing

    %ker_size = 41; %ALWAYS ODD, size of the kernel.
    %ker_size = 51; %  Make sure kernel is big enough. If selected image has patches with
                %  too much blur, then kernel size should be adjusted accordingly.

%    stride = 33;%33; % ALWAYS ODD,
%                    % the bigger this number, the faster the algorithm, and
%                    % the coarser the output map

    %border = 25; % border pixels to leave out during optimization cost computation
            % should be greater than half of kernel size


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % user inputs end here
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    extraval = ['_p_' num2str(patch_size) '_k_' num2str(ker_size) '_s_' num2str(stride) '_r_' num2str(image_resize_val)];
    fprintf('%s \n',fullfile(select_folder,[imgname(1:end-6) extraval]))

    %imwrite([0, 0],fullfile(save_path,ext +'test.png'));

    foldername = [imgname(1:end-6) extraval];

    % read left and right images
    imgl=imread(fullfile(select_folder,[imgname(1:end-6) '_L.']+ext));
    imgr=imread(fullfile(select_folder,[imgname(1:end-6) '_R.']+ext));
    % and the combined image
    imgc=imread(fullfile(select_folder,imgname));

    % resize the images
    imgl = imresize(imgl,image_resize_val,'bicubic');
    imgr = imresize(imgr,image_resize_val,'bicubic');
    imgc = imresize(imgc,image_resize_val,'bicubic');

    % if img is 16-bit, convert to 8-bit
    if max(imgc(:))>255
        imgl=uint8(double(imgl)/256);
        imgr=uint8(double(imgr)/256);
        imgc=uint8(double(imgc)/256);
    end
    % if img is color, convert to grayscale
    % convert to grayscale
    if size(imgc,3)>1
        imgcg=double(rgb2gray(imgc));
    else
        imgcg=double(imgc);
    end
    if size(imgl,3)>1
        imglg=double(rgb2gray(imgl));
        imgrg=double(rgb2gray(imgr));
    else
        imglg=double(imgl);
        imgrg=double(imgr);
    end



    % run optimization algorithm
    [out_map,out_fval,out_sobel] = run_optimization_translating_disk_kernel(patch_size,ker_size,imglg,imgrg,imgcg,border,stride,fig_flag);

    % compute confidenc map and save results in a manner suited to run Step 2
    % which is the bilateral solver
    [target,confidence,reference] = prep_for_step_two(imgc,out_map,out_fval,out_sobel,stride,patch_size,ker_size);

    % save for Step 2
    mkdir(fullfile(save_path,[imgname(1:end-6) extraval]))
    save(fullfile(save_path,[imgname(1:end-6) extraval '/raw.mat']),'target','confidence', 'out_sobel', 'out_fval');


    imwrite(confidence,fullfile(save_path,[imgname(1:end-6) extraval '/confidence.png'])); % just for display
    target=(target-min(target(:)))/(max(target(:))-min(target(:)));
    imwrite(uint8(target*255),fullfile(save_path,[imgname(1:end-6) extraval '/target.png'])); % just for display
    imwrite(reference,fullfile(save_path,[imgname(1:end-6) extraval '/reference.png']))
    save(fullfile(save_path,[imgname(1:end-6) extraval '/res.mat']),'target','confidence'); % used in Step 2

    fprintf('\nFinished execution \n \n');
    fprintf('In Steps 2 and 3, \nuse directory_name = ''%s'' \nand img_name = ''%s''\n',save_path,[imgname(1:end-6) extraval]);

end