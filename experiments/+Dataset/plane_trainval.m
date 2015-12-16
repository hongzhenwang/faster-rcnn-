function dataset = plane_trainval(dataset, usage, use_flip)
% Pascal voc 2007 trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit                      = '/home/whz/image_set/plane20151105/image_train';

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_plane(devkit, 'trainval', '2015', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_plane(devkit, 'trainval', '2015', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''train'' or ''test''');
end

end