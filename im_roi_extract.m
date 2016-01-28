%%%提取飞机数据集的proposal以及ori信息,包括label和bbox
%%给国利师兄做实验用
clc;clear;
load('dataset_plane.mat')
image_roidb_train = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                        struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                        'overlap', y.rois(z).overlap,'ori_overlap', y.rois(z).ori_overlap, 'boxes', y.rois(z).boxes,'ori', y.rois(z).ori, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
        dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
im_roi_train=cat(1,image_roidb_train{:});
image_roidb_test = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                        struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                        'overlap', y.rois(z).overlap,'ori_overlap', y.rois(z).ori_overlap, 'boxes', y.rois(z).boxes,'ori', y.rois(z).ori, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
        {dataset.imdb_test}, {dataset.roidb_test}, 'UniformOutput', false);
im_roi_test=cat(1,image_roidb_test{:});
for i=1:length(im_roi_train)
    [overlaps, labels] = max(im_roi_train(i).overlap, [], 2);
%     labels = im_roi_train(1).max_classes;
%     overlaps = im_roi_train(1).max_overlaps;
    rois = im_roi_train(i).boxes;
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(overlaps >= 0.5);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image = min(16, length(fg_inds));
    % Sample foreground regions without replacement
    if ~isempty(fg_inds)
       fg_inds = fg_inds(randperm(length(fg_inds), fg_rois_per_this_image));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(overlaps < 0.5 & overlaps >= 0.1);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image = 64 - fg_rois_per_this_image;
    bg_rois_per_this_image = min(bg_rois_per_this_image, length(bg_inds));
    % Sample foreground regions without replacement
    if ~isempty(bg_inds)
       bg_inds = bg_inds(randperm(length(bg_inds), bg_rois_per_this_image));
    end
    % The indices that we're selecting (both fg and bg)
    keep_inds = [fg_inds; bg_inds];
    % Select sampled values from various arrays
    labels = labels(keep_inds);
    % Clamp labels for the background ROIs to 0
    labels((fg_rois_per_this_image+1):end) = 0;
    overlaps = overlaps(keep_inds);
    rois = rois(keep_inds, :);
    train(i).labels=labels;
    train(i).boxes=rois;
    train(i).image_id=im_roi_train(i).image_id;
    train(i).im_size=im_roi_train(i).im_size;
    
    
    %%%%%%%%%%%%%%ori label and bboxes%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     [overlaps_ori, labels_ori] = max(im_roi_train(i).ori_overlap, [], 2);
    % Select sampled values from various arrays
    ori_labels = labels_ori(keep_inds);
    % Clamp labels for the background ROIs to 0
    ori_labels((fg_rois_per_this_image+1):end) = 0;
    train(i).ori_labels=ori_labels;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
for i=1:length(im_roi_test)
    [overlaps, labels] = max(im_roi_test(i).overlap, [], 2);
%     labels = im_roi_test(1).max_classes;
%     overlaps = im_roi_test(1).max_overlaps;
    rois = im_roi_test(i).boxes;
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(overlaps >= 0.5);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image = min(16, length(fg_inds));
    % Sample foreground regions without replacement
    if ~isempty(fg_inds)
       fg_inds = fg_inds(randperm(length(fg_inds), fg_rois_per_this_image));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(overlaps < 0.5 & overlaps >= 0.1);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image = 64 - fg_rois_per_this_image;
    bg_rois_per_this_image = min(bg_rois_per_this_image, length(bg_inds));
    % Sample foreground regions without replacement
    if ~isempty(bg_inds)
       bg_inds = bg_inds(randperm(length(bg_inds), bg_rois_per_this_image));
    end
    % The indices that we're selecting (both fg and bg)
    keep_inds = [fg_inds; bg_inds];
    % Select sampled values from various arrays
    labels = labels(keep_inds);
    % Clamp labels for the background ROIs to 0
    labels((fg_rois_per_this_image+1):end) = 0;
    overlaps = overlaps(keep_inds);
    rois = rois(keep_inds, :);
    test(i).labels=labels;
    test(i).boxes=rois;
    test(i).image_id=im_roi_test(i).image_id;
    test(i).im_size=im_roi_test(i).im_size;
 
    
    
    %%%%%%%%%%%%%%ori label and bboxes%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     [overlaps_ori, labels_ori] = max(im_roi_test(i).ori_overlap, [], 2);
    % Select sampled values from various arrays
    ori_labels = labels_ori(keep_inds);
    % Clamp labels for the background ROIs to 0
    ori_labels((fg_rois_per_this_image+1):end) = 0;
    test(i).ori_labels=ori_labels;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% save('train_test_plane_whz.mat','train','test','-v7.3')