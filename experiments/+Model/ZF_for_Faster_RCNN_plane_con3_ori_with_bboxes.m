function model = ZF_for_Faster_RCNN_plane_con3_ori_with_bboxes(model)

model.mean_image                                = fullfile(pwd, 'models', 'pre_trained_models', 'ZF', 'mean_image');
% model.pre_trained_net_file                      = fullfile(pwd, 'output', 'rpn_cachedir', 'faster_rcnn_VOC2007_ZF_stage1_rpn', 'voc_2007_trainval','final');
model.pre_trained_net_file                      = fullfile(pwd,  'models', 'pre_trained_models', 'ZF','ZF.caffemodel' );
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'ZF', 'solver_plane_10k20k_con3.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'ZF', 'test_plane_con3.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN              	= -1;
model.stage1_rpn.nms.nms_overlap_thres        	= 0.7;
model.stage1_rpn.nms.after_nms_topN           	= 2000;

%% stage 1 fast rcnn, inited from pre-trained network
model.stage1_fast_rcnn.solver_def_file          = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'ZF', 'solver_plane_20k30k_con3.prototxt');
model.stage1_fast_rcnn.test_net_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'ZF', 'test_plane_con3.prototxt');
model.stage1_fast_rcnn.init_net_file            = model.pre_trained_net_file;

%% stage 2 rpn, only finetune fc layers
model.stage2_rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'ZF_fc6', 'solver_plane_10k20k_con3.prototxt');
model.stage2_rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'ZF_fc6', 'test_plane_con3.prototxt');

% rpn test setting
model.stage2_rpn.nms.per_nms_topN             	= -1;
model.stage2_rpn.nms.nms_overlap_thres       	= 0.7;
model.stage2_rpn.nms.after_nms_topN           	= 2000;

%% stage 2 fast rcnn, only finetune fc layers
model.stage2_fast_rcnn.solver_def_file          = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'ZF_fc6', 'solver_plane_20k30k_con3.prototxt');
model.stage2_fast_rcnn.test_net_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'ZF_fc6', 'test_plane_con3.prototxt');

%% stage  ori, only finetune fc layers
model.stage_ori.solver_def_file          = fullfile(pwd, 'models', 'ori_prototxts', 'ZF_con3 _with_bbox', 'solver_ori_20k30k_con3.prototxt');
model.stage_ori.test_net_def_file        = fullfile(pwd, 'models', 'ori_prototxts', 'ZF_con3 _with_bbox', 'test_ori_con3.prototxt');

%% final test
model.final_test.nms.per_nms_topN            	= 6000; % to speed up nms
model.final_test.nms.nms_overlap_thres       	= 0.7;
model.final_test.nms.after_nms_topN          	= 300;
end