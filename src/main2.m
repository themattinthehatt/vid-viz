%% import data

% import
% myimg = imread('data/test.jpg');

%% crop to have 16:9 ratio

% find out if one or both dims need to be resized
[im_height, im_width, im_channels] = size(myimg);

if im_width/im_height < 16/9 + 0.03 && im_width/im_height > 16/9 - 0.03
    im_resize = 0;
else
    im_resize = 1;
end

if im_resize
        
    if im_width < 16/9*im_height
        % keep width, crop height
        target_pixels = 9/16*im_width;
        if mod(im_height,2) == 0
            % make target pixels even
            if mod(ceil(target_pixels),2) == 0
                target_pixels = ceil(target_pixels);
            else
                target_pixels = floor(target_pixels);
            end
            myimg = myimg((im_height/2 - target_pixels/2 + 1) : ...
                          (im_height/2 + target_pixels/2), :, :);
        else
            % make target pixels odd
            if mod(ceil(target_pixels),2) == 1
                target_pixels = ceil(target_pixels);
            else
                target_pixels = floor(target_pixels);
            end
            myimg = myimg((ceil(im_height/2) - floor(target_pixels/2)) : ...
                          (ceil(im_height/2) + floor(target_pixels/2)), :, :);
        end
    else
        % keep height, crop width
        target_pixels = 16/9*im_height;
        if mod(im_width,2) == 0
            % make target pixels even
            if mod(ceil(target_pixels),2) == 0
                target_pixels = ceil(target_pixels);
            else
                target_pixels = floor(target_pixels);
            end
            myimg = myimg(:, (im_width/2 - target_pixels/2 + 1) : ...
                             (im_width/2 + target_pixels/2), :);
        else
            % make target pixels odd
            if mod(ceil(target_pixels),2) == 1
                target_pixels = ceil(target_pixels);
            else
                target_pixels = floor(target_pixels);
            end
            myimg = myimg(:, (ceil(im_width/2) - floor(target_pixels/2)) : ...
                             (ceil(im_width/2) + floor(target_pixels/2)), :);
        end
    end    
    
end

%% process


%% plot

% figure;
% image(myimg)
% 
% % get the figure and axes handles
% hFig = gcf;
% hAx  = gca;
% 
% % set the figure to full screen
% set(hFig,'units','normalized','outerposition',[0 0 1 1]);
% 
% % set the axes to full screen
% set(hAx,'Unit','normalized','Position',[0 0 1 1]);
% 
% % hide the toolbar
% set(hFig,'menubar','none')
% 
% % to hide the title
% set(hFig,'NumberTitle','off');

