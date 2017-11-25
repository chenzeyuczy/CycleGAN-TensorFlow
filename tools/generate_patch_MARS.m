% Script tp generate occlusion on images in MARS dataset, whose size are 256 * 128.

disp('get path...');

root_src = '~/data/MARS/bbox_train';
root_dst = '~/data/MARS/noise_multipatch';
label_file = [root_dst, '/label.txt'];

folder_src = dir(root_src);
fid = fopen(label_file, 'w');

if ~exist(root_dst, 'dir')
	mkdir(root_dst);
	mkdir([root_dst, '/whole']);
	mkdir([root_dst, '/occlude'])
	mkdir([root_dst, '/white'])
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
	num_id_select = min(35, length(img) - 2);
    id = randperm(length(img)-2, num_id_select) + 2;
    for j = 1:35
		img_name = img(id(j)).name;
        copyfile([img_filepath, '/', img_name], [root_dst,'/whole/',img_name]);
        im_data = imread([img_filepath,'/',img_name]);
        im_data_gray = im_data; 
        [H, W, ~] = size(im_data);

		% Generate random patch along left/right side to occlude.
		patch_w = ceil(W * 0.2);
		patch_h = ceil(H * 0.2);
		switch randi(2)
			case 1 % Crop patch along the left side.
			patch_x = 1;
			case 2 % Crop patch along the right side.
			patch_x = W - patch_w + 1;
		end
		patch_y = randi(H - patch_h + 1);
        im_occlude = im_data(patch_y: patch_y + patch_h - 1, patch_x:patch_x + patch_w - 1, :); 
        im_occlude_gray = im_occlude;
        im_occlude_gray(:) = 256;
%         imshow(im_occlude_gray);

		occlude_type = randi(3);
        if occlude_type == 1   %col shelther
			% Generate horizontal occulding block.
            point_w = randi(round(W * 0.6)) + round(W * 0.2);
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
            point_h = randi(round(H*0.6))+round(H*0.2);
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
			patch_num = randi(3);
			occlude_y = [];
			occlude_x = [];
			occlude_w = [];
			occlude_h = [];
			for idx = 1:patch_num
				height = randi(60) + 40;
				width = randi(60) + 40;
				im_occlude = imresize(im_occlude, [height, width]);
				im_occlude_gray = imresize(im_occlude_gray, [height, width]);
				patch_y = randperm(H - height,1);
				patch_x = randperm(W - width,1);
				im_data(patch_y:patch_y+height-1,patch_x:patch_x+width-1,:) = im_occlude;
				im_data_gray(patch_y:patch_y+height-1,patch_x:patch_x+width-1,:) = im_occlude_gray;
				occlude_y(end + 1) = patch_y;
				occlude_x(end + 1) = patch_x;
				occlude_h(end + 1) = height;
				occlude_w(end + 1) = width;
			end
        end
        
        imwrite(im_data,[root_dst,'/occlude/', img_name]);
        imwrite(im_data_gray,[root_dst,'/white/', img_name]);
		if occlude_type < 3
			fprintf(fid, '%s %d %d %d %d %d\n', img_name, occlude_type, occlude_x, occlude_y, occlude_w, occlude_h);
		else
			fprintf(fid, '%s %d', img_name, occlude_type);
			for idx = 1:length(occlude_x)
				fprintf(fid, ' %d %d %d %d', occlude_x, occlude_y, occlude_w, occlude_h);
			end
			fprintf(fid, '\n');
		end
    end

end

fclose(fid);
fprintf('Done!\n');

