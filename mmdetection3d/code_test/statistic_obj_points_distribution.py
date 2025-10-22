import pickle

anno_path = "data/kitti/kitti_infos_train.pkl"
with open(anno_path, 'rb') as f:
    kitti_infos = pickle.load(f)
    import ipdb ; ipdb.set_trace()