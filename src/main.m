proc_type = 'edge_detect';
fps = 60;
print_fps = 1;

tic

myimg = imread('data/test2.jpg');
myimg = single(myimg);
myimg = myimg/max(myimg(:));
main2;
myimg2 = myimg;
myimg2(:,:,1) = conv2(myimg(:,:,1), 1/8*[ 0, -1, 0;
                                          -1, 4, -1; 
                                          0, -1, 0], 'same');

figure;
image(myimg2)

% get the figure and axes handles
hFig = gcf;
hAx  = gca;

% set the figure to full screen
set(hFig,'units','normalized','outerposition',[0 0 1 1]);

% set the axes to full screen
set(hAx,'Unit','normalized','Position',[0 0 1 1]);

% hide the toolbar
set(hFig,'menubar','none')

% to hide the title
set(hFig,'NumberTitle','off');

while 1

    time_old = toc;
    
    %% pick image for processing
    
    % load new source if necessary
    
    % pick image in video stack
    
    
    %% choose processing technique
    switch proc_type
        case 'rgb_walk'
        case 'edge_detect'
            
    end
    
    %% plot current image
    image(myimg2)
    
    %% pause to hit desired fps
    time_new = toc;
    if (time_new - time_old) < 1/fps
        pause(1/fps - (time_new - time_old))
        time_new = toc;
    end
    if print_fps
        fprintf('\b\b\b\b\b\b\b\b%4.1f fps', 1 / (time_new - time_old))
    end
    
end