function [ tp,fp,fn ] = compute_Ap_recall(boxes_cell,annotation_path,im_name )
%计算某一次结果输出的Ap和recall
%   Detailed explanation goes here
% load /home/whz/DNN/faster_rcnn-master/boxes_cell.mat;
% annotation_path='/home/whz/image_set/plane20151105/AnnotationMat';
% im_name='108_per.bmp';
annotation_path=fullfile(annotation_path,'%s.mat');
[~,name,~]=fileparts(im_name);
load(sprintf(annotation_path,name));
gt_boxes=cat(1,rec.objects(:).bbox);
det=zeros(size(gt_boxes,1),1);
thre=0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%class_to_id
classes ={...
        'airfreighter'
        'fighter'
        'bomber'
        'AEM'
        'aerobus'
        'tanker'};
num_classes = length(classes);
tp=zeros(num_classes,1);fp=zeros(num_classes,1);fn=zeros(num_classes,1);
class_to_id =containers.Map(classes, 1:num_classes);
tem2_whz=cat(1,rec.objects(:).class);
gt_classes = class_to_id.values(tem2_whz);
gt_classes = cat(1, gt_classes{:});
num_gt_boxes = size(gt_boxes, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(boxes_cell)
    out_boxes=boxes_cell{i}(:,1:4);
    overlap= boxoverlap(out_boxes, gt_boxes);
    [~, gt_assignment] = max(overlap, [], 2);
    [num_out_boxes,~]=size(out_boxes);
    label_out_boxes=zeros(num_out_boxes,1);
    for j=1:num_out_boxes
        if overlap(j,gt_assignment(j))>thre
            if i==gt_classes(gt_assignment(j))
                if det(gt_assignment(j))==0
                    label_out_boxes(j)=1;
                    det(gt_assignment(j))=1;
                else
                    label_out_boxes(j)=0;
                end
            else
               label_out_boxes(j)=0;
            end
        else
            label_out_boxes(j)=0;
        end
    end
%     assert(sum(det(:))==sum(label_out_boxes(:)));
    tp(i)=sum(label_out_boxes(:));
    fn(i)=length(find(gt_classes==i))-tp(i);
    fp(i)=num_out_boxes-tp(i);
end

