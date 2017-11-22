disp('get path...');

root_src = '~/data/MARS/bbox_train';
root_dst = '~/data/MARS/noise';
label_file = '~/data/MARS/noise/label.txt';

folder_src = dir(root_src);
fid = fopen(label_file, 'w');

if ~exist(root_dst, 'dir')
	mkdir(root_dst);
	mkdir([root_dst, '/whole']);
	mkdir([root_dst, '/occlude'])
	mkdir([root_dst, '/pure'])
end

% Traversal along folder.
for i=1:length(folder_src)-2
	% Show pregress.
	if mod(i, 10) == 0
		fprintf('Deal with %d/%d folders.\n', i, length(folder_src) - 2);
	end
    img_filepath = [root_src, '/', folder_src(i+2).name];
    img = dir(img_filepath);

	% Select part of images at random.
    id = randperm(length(img)-2,30)+2;
    for j = 1:30
		img_name = img(id(j)).name;
        copyfile([img_filepath, '/', img_name], [root_dst,'/whole/',img_name]);
        im_data = imread([img_filepath,'/',img_name]);
        im_data_gray = im_data; 
        [H, W, ~] = size(im_data);

		% Generate patch to occlude.
        im_occlude = im_data(1:30, 1:30, :); 
        im_occlude_gray = im_occlude;
        im_occlude_gray(:) = 256;
%         imshow(im_occlude_gray);

		occlude_type = randi(3);
        if occlude_type == 1   %col shelther
			% Generate horizontal occulding block.
            point_w = randperm(round(W*0.6),1)+round(W*0.2);
			occlude_w = ceil((rand() * 0.3 + 0.2) * W);  % 20~50% of width
			occlude_h = H;
			occlude_y = 1;

            if point_w <= W / 2
                im_occlude = imresize(im_occlude, [H, point_w]);
                im_occlude_gray = imresize(im_occlude_gray, [H, point_w]);
                im_data(:,1:point_w,:)= im_occlude;
                im_data_gray(:,1:point_w,:)= im_occlude_gray;
				occlude_x = 1;
            else
                im_occlude = imresize(im_occlude, [H, W - point_w]);
                im_occlude_gray = imresize(im_occlude_gray, [H, W - point_w]);
                im_data(:,point_w+1:end,:) = im_occlude;
                im_data_gray(:,point_w+1:end,:) = im_occlude_gray;
				occlude_x = W - occlude_w + 1;
            end
        elseif occlude_type == 2
			% Generate vertical occulding block.
			occlude_h = ceil((rand() * 0.3 + 0.2) * H);  % 20~50% of height
            point_h = randperm(round(H*0.6),1)+round(H*0.2);
			occlude_w = W;
			occlude_x = 1;

            if point_h <= H / 2
                im_occlude = imresize(im_occlude, [point_h, W]);
                im_occlude_gray = imresize(im_occlude_gray, [point_h, W]);
                im_data(1:point_h,:,:) = im_occlude;
                im_data_gray(1:point_h,:,:) = im_occlude_gray;
				occlude_y = 1;
            else
                im_occlude = imresize(im_occlude, [H - point_h, W]);
                im_occlude_gray = imresize(im_occlude_gray, [H - point_h,W]);
                im_data(point_h+1:end,:,:) = im_occlude;
                im_data_gray(point_h+1:end,:,:) = im_occlude_gray;
				occlude_y = H - occlude_h + 1;
            end
        else
			% Generate random-float occluding block.
			height = ceil(rand() * 60) + 60;
			width = ceil(rand() * 60) + 60;
            im_occlude = imresize(im_occlude, [height, width]);
            im_occlude_gray = imresize(im_occlude_gray, [height, width]);
            occlude_y = randperm(H - height,1);
            occlude_x = randperm(W - width,1);
            im_data(occlude_y:occlude_y+height-1,occlude_x:occlude_x+width-1,:)= im_occlude;
            im_data_gray(occlude_y:occlude_y+height-1,occlude_x:occlude_x+width-1,:)= im_occlude_gray;
			occlude_h = height;
			occlude_w = width;
        end
        
        imwrite(im_data,[root_dst,'/occlude/', img_name]);
        imwrite(im_data_gray,[root_dst,'/white/', img_name]);
		fprintf(fid, '%s %d %d %d %d %d\n', img_name, occlude_x, occlude_y, occlude_w, occlude_h, occlude_type);
    end

end

fclose(fid);
fprintf('Done!\n');
