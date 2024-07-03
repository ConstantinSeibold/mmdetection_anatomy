import requests
import shutil
import os
import json
from mmengine import Config
import numpy as np
def download_file(url, save_path):
    if os.path.exists(save_path):
        print("File already exists.")
        return
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        print("File downloaded successfully to:", save_path)
    else:
        print("Failed to download file")

def add_head_cfg(cfg):
    cfg.data_root = '/home/jovyan/xray-data-datavol-1/ChestXDet/'


    img_scales = [
            (384, 384),
            (512, 512),
            (640, 640),
            (768, 768),
    ]
    
    cfg.train_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/head_train.json'
    
    json_info = json.load(open(cfg.train_dataloader.dataset.ann_file))
    
    cfg.metainfo = {
        'classes': [c["name"] for c in json_info["categories"]],
        'palette': [np.random.randint(0,255,3).tolist() for c in json_info["categories"]]
    }
    
    
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'train/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    cfg.train_dataloader.dataset.pipeline = [
                            {'type': 'LoadImageFromFile', 'backend_args': None},
                            {'type': 'LoadAnnotations',
                            'with_bbox': True,
                            'with_mask': True,
                            'poly2mask': True},
                            {"type": "CutOut",
                             "n_holes": (0,4),
                             "cutout_shape": [(16,16),(24,24),(32,32),(48,48)]
                            },
                            {"type": "RandAugment",
                             "aug_num": 3,
                            },
                            {'type': 'RandomChoiceResize',
                            'scales': img_scales,
                            'keep_ratio': False},
                            {"type":"RandomCrop",
                             "crop_size": (512,512),                            
                            },
                            {'type': 'PackDetInputs'}
                        ]
    cfg.train_dataloader.num_workers = 4
    cfg.train_dataloader.batch_size = 16
    cfg.train_pipeline = cfg.train_dataloader.dataset.pipeline

    
    cfg.val_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/head_val.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = 'train/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=img_scales[1], type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
            ]
    cfg.val_dataloader.num_workers = 4

    cfg.test_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/head_val.json'
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.pipeline = cfg.val_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.data_prefix.img = 'train/'
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.test_dataloader.num_workers = 4
    

    cfg.test_pipeline = [
                                    dict(type='LoadImageFromFile', backend_args=None),
                                    dict(
                                        type='TestTimeAug',
                                        transforms=[
                                            [
                                                dict(type='Resize', scale=s, keep_ratio=False)
                                                for s in img_scales
                                            ],
                                            [
                                                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                                                # bounding box coordinates after flipping cannot be
                                                # recovered correctly.
                                                dict(type='RandomFlip', prob=1.),
                                                dict(type='RandomFlip', prob=0.)
                                            ],
                                            [
                                                dict(
                                                    type='Pad',
                                                    size= img_scales[-1],
                                                    pad_val=dict(img=(114, 114, 114))),
                                            ],
                                            [dict(type='LoadAnnotations', with_bbox=True)],
                                            [
                                                dict(
                                                    type='PackDetInputs',
                                                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                                               'scale_factor', 'flip', 'flip_direction'))
                                            ]
                                        ])
                                    ]
    cfg.tta_pipeline = cfg.test_pipeline
    cfg.tta_model = dict(
        type="DetTTAModel",
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=200))

    # Modify metric config
    cfg.val_evaluator.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/head_val.json'
    cfg.val_evaluator.metric = ["segm","bbox"]
    cfg.val_evaluator.classwise=True
    cfg.test_evaluator.ann_file =  '/home/jovyan/xray-data-datavol-1/jsons/head_val.json'
    cfg.test_evaluator.metric = ["segm","bbox"]
    cfg.test_evaluator.classwise=True



    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 10
    cfg.default_hooks.logger.interval = 10

    cfg.work_dir = '/home/jovyan/xray-models/segmentation_models/anatomy/head'
    cfg.seed = 0
    return cfg

def add_breast_cfg(cfg):
    cfg.data_root = '/home/jovyan/xray-data-datavol-1/ChestXDet/'


    img_scales = [
            (384, 384),
            (512, 512),
            (640, 640),
            (768, 768),
    ]
    
    cfg.train_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/breasts_train.json'
    
    json_info = json.load(open(cfg.train_dataloader.dataset.ann_file))
    
    cfg.metainfo = {
        'classes': [c["name"] for c in json_info["categories"]],
        'palette': [np.random.randint(0,255,3).tolist() for c in json_info["categories"]]
    }
    
    
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'train/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    cfg.train_dataloader.dataset.pipeline = [
                            {'type': 'LoadImageFromFile', 'backend_args': None},
                            {'type': 'LoadAnnotations',
                            'with_bbox': True,
                            'with_mask': True,
                            'poly2mask': True},
                            {"type": "CutOut",
                             "n_holes": (0,4),
                             "cutout_shape": [(16,16),(24,24),(32,32),(48,48)]
                            },
                            {"type": "RandAugment",
                             "aug_num": 3,
                            },
                            {'type': 'RandomChoiceResize',
                            'scales': img_scales,
                            'keep_ratio': False},
                            {"type":"RandomCrop",
                             "crop_size": (512,512),                            
                            },
                            {'type': 'PackDetInputs'}
                        ]
    cfg.train_dataloader.num_workers = 4
    cfg.train_dataloader.batch_size = 16
    cfg.train_pipeline = cfg.train_dataloader.dataset.pipeline

    
    cfg.val_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/breasts_val.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = 'train/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=img_scales[1], type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
            ]
    cfg.val_dataloader.num_workers = 4

    cfg.test_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/breasts_val.json'
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.pipeline = cfg.val_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.data_prefix.img = 'images/'
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.test_dataloader.num_workers = 4
    

    cfg.test_pipeline = [
                                    dict(type='LoadImageFromFile', backend_args=None),
                                    dict(
                                        type='TestTimeAug',
                                        transforms=[
                                            [
                                                dict(type='Resize', scale=s, keep_ratio=False)
                                                for s in img_scales
                                            ],
                                            [
                                                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                                                # bounding box coordinates after flipping cannot be
                                                # recovered correctly.
                                                dict(type='RandomFlip', prob=1.),
                                                dict(type='RandomFlip', prob=0.)
                                            ],
                                            [
                                                dict(
                                                    type='Pad',
                                                    size= img_scales[-1],
                                                    pad_val=dict(img=(114, 114, 114))),
                                            ],
                                            [dict(type='LoadAnnotations', with_bbox=True)],
                                            [
                                                dict(
                                                    type='PackDetInputs',
                                                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                                               'scale_factor', 'flip', 'flip_direction'))
                                            ]
                                        ])
                                    ]
    cfg.tta_pipeline = cfg.test_pipeline
    cfg.tta_model = dict(
        type="DetTTAModel",
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=200))

    # Modify metric config
    cfg.val_evaluator.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/breasts_val.json'
    cfg.val_evaluator.metric = ["segm","bbox"]
    cfg.val_evaluator.classwise=True
    cfg.test_evaluator.ann_file =  '/home/jovyan/xray-data-datavol-1/jsons/breasts_val.json'
    cfg.test_evaluator.metric = ["segm","bbox"]
    cfg.test_evaluator.classwise=True



    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 10
    cfg.default_hooks.logger.interval = 10

    cfg.work_dir = '/home/jovyan/xray-models/segmentation_models/anatomy/breast'
    cfg.seed = 0
    return cfg

def add_humerus_cfg(cfg):
    cfg.data_root = '/home/jovyan/xray-data-datavol-1/OpenI/'


    img_scales = [
            (384, 384),
            (512, 512),
            (640, 640),
            (768, 768),
    ]
    
    cfg.train_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/humerus_train.json'
    
    json_info = json.load(open(cfg.train_dataloader.dataset.ann_file))
    
    cfg.metainfo = {
        'classes': [c["name"] for c in json_info["categories"]],
        'palette': [np.random.randint(0,255,3).tolist() for c in json_info["categories"]]
    }
    
    
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'images/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    cfg.train_dataloader.dataset.pipeline = [
                            {'type': 'LoadImageFromFile', 'backend_args': None},
                            {'type': 'LoadAnnotations',
                            'with_bbox': True,
                            'with_mask': True,
                            'poly2mask': False},
                            {"type": "CutOut",
                             "n_holes": (0,4),
                             "cutout_shape": [(16,16),(24,24),(32,32),(48,48)]
                            },
                            {"type": "RandAugment",
                             "aug_num": 3,
                            },
                            {'type': 'RandomChoiceResize',
                            'scales': img_scales,
                            'keep_ratio': False},
                            {"type":"RandomCrop",
                             "crop_size": (512,512),                            
                            },
                            {'type': 'PackDetInputs'}
                        ]
    cfg.train_dataloader.num_workers = 4
    cfg.train_dataloader.batch_size = 16
    cfg.train_pipeline = cfg.train_dataloader.dataset.pipeline

    
    cfg.val_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/humerus_val.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = 'images/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=img_scales[1], type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
            ]
    cfg.val_dataloader.num_workers = 4

    cfg.test_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/humerus_val.json'
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.pipeline = cfg.val_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.data_prefix.img = 'images/'
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.test_dataloader.num_workers = 4
    

    cfg.test_pipeline = [
                                    dict(type='LoadImageFromFile', backend_args=None),
                                    dict(
                                        type='TestTimeAug',
                                        transforms=[
                                            [
                                                dict(type='Resize', scale=s, keep_ratio=False)
                                                for s in img_scales
                                            ],
                                            [
                                                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                                                # bounding box coordinates after flipping cannot be
                                                # recovered correctly.
                                                dict(type='RandomFlip', prob=1.),
                                                dict(type='RandomFlip', prob=0.)
                                            ],
                                            [
                                                dict(
                                                    type='Pad',
                                                    size= img_scales[-1],
                                                    pad_val=dict(img=(114, 114, 114))),
                                            ],
                                            [dict(type='LoadAnnotations', with_bbox=True)],
                                            [
                                                dict(
                                                    type='PackDetInputs',
                                                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                                               'scale_factor', 'flip', 'flip_direction'))
                                            ]
                                        ])
                                    ]
    cfg.tta_pipeline = cfg.test_pipeline
    cfg.tta_model = dict(
        type="DetTTAModel",
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=200))

    # Modify metric config
    cfg.val_evaluator.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/humerus_val.json'
    cfg.val_evaluator.metric = ["segm","bbox"]
    cfg.val_evaluator.classwise=True
    cfg.test_evaluator.ann_file =  '/home/jovyan/xray-data-datavol-1/jsons/humerus_val.json'
    cfg.test_evaluator.metric = ["segm","bbox"]
    cfg.test_evaluator.classwise=True



    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 10
    cfg.default_hooks.logger.interval = 10

    cfg.work_dir = '/home/jovyan/xray-models/segmentation_models/anatomy/humerus'
    cfg.seed = 0
    return cfg

def add_paxray_cfg(cfg):
    cfg.data_root = '/home/jovyan/xray-data-datavol-1/paxray/'


    img_scales = [
            (384, 384),
            (512, 512),
            (640, 640),
            (768, 768),
    ]
    
    cfg.train_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/paxray_train_val.json'
    
    json_info = json.load(open(cfg.train_dataloader.dataset.ann_file))
    
    cfg.metainfo = {
        'classes': [c["name"] for c in json_info["categories"]],
        'palette': [np.random.randint(0,255,3).tolist() for c in json_info["categories"]]
    }
    
    
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'images/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    cfg.train_dataloader.dataset.pipeline = [
                            {'type': 'LoadImageFromFile', 'backend_args': None},
                            {'type': 'LoadAnnotations',
                            'with_bbox': True,
                            'with_mask': True,
                            'poly2mask': True},
                            {"type": "CutOut",
                             "n_holes": (0,4),
                             "cutout_shape": [(16,16),(24,24),(32,32),(48,48)]
                            },
                            {"type": "RandAugment",
                             "aug_num": 3,
                            },
                            {'type': 'RandomChoiceResize',
                            'scales': img_scales,
                            'keep_ratio': False},
                            {"type":"RandomCrop",
                             "crop_size": (512,512),                            
                            },
                            {'type': 'PackDetInputs'}
                        ]
    cfg.train_dataloader.num_workers = 4
    cfg.train_dataloader.batch_size = 32
    cfg.train_pipeline = cfg.train_dataloader.dataset.pipeline

    
    cfg.val_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/paxray_test_frontal.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = 'images/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=img_scales[1], type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
            ]
    cfg.val_dataloader.num_workers = 4

    cfg.test_dataloader.dataset.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/paxray_test_frontal.json'
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.pipeline = cfg.val_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.data_prefix.img = 'images/'
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.test_dataloader.num_workers = 4
    

    cfg.test_pipeline = [
                                    dict(type='LoadImageFromFile', backend_args=None),
                                    dict(
                                        type='TestTimeAug',
                                        transforms=[
                                            [
                                                dict(type='Resize', scale=s, keep_ratio=False)
                                                for s in img_scales
                                            ],
                                            [
                                                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                                                # bounding box coordinates after flipping cannot be
                                                # recovered correctly.
                                                dict(type='RandomFlip', prob=1.),
                                                dict(type='RandomFlip', prob=0.)
                                            ],
                                            [
                                                dict(
                                                    type='Pad',
                                                    size= img_scales[-1],
                                                    pad_val=dict(img=(114, 114, 114))),
                                            ],
                                            [dict(type='LoadAnnotations', with_bbox=True)],
                                            [
                                                dict(
                                                    type='PackDetInputs',
                                                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                                               'scale_factor', 'flip', 'flip_direction'))
                                            ]
                                        ])
                                    ]
    cfg.tta_pipeline = cfg.test_pipeline
    cfg.tta_model = dict(
        type="DetTTAModel",
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=200))

    # Modify metric config
    cfg.val_evaluator.ann_file = '/home/jovyan/xray-data-datavol-1/jsons/paxray_test_frontal.json'
    cfg.val_evaluator.metric = ["segm","bbox"]
    cfg.val_evaluator.classwise=True
    cfg.test_evaluator.ann_file =  '/home/jovyan/xray-data-datavol-1/jsons/paxray_test_frontal.json'
    cfg.test_evaluator.metric = ["segm","bbox"]
    cfg.test_evaluator.classwise=True



    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 10
    cfg.default_hooks.logger.interval = 10

    cfg.work_dir = '/home/jovyan/xray-models/segmentation_models/anatomy/paxray_3randaug'
    cfg.seed = 0
    return cfg



def add_mimic_anatomy_cfg(cfg):
    cfg.data_root = '/home/jovyan/xray-data-datavol-1/MIMIC/'


    img_scales = [
            (384, 384),
            (512, 512),
            (640, 640),
            (768, 768),
    ]
    
    cfg.train_dataloader.dataset.ann_file = '/home/jovyan/xray-models/predictions/anatomy_predictions.json'
    
    json_info = json.load(open(cfg.train_dataloader.dataset.ann_file))
    
    cfg.metainfo = {
        'classes': [c["name"] for c in json_info["categories"]],
        'palette': [np.random.randint(0,255,3).tolist() for c in json_info["categories"]]
    }
    
    
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'images_512/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    
    cfg.train_dataloader.dataset = dict(
            type = 'CocoDataset',
            data_prefix=dict(img='images_512'),
            data_root=cfg.data_root,
            ann_file='balloon_train.json',
            metainfo= cfg.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline= [
                {'type': 'LoadImageFromFile',
                       'to_float32': True,
                       'backend_args': None},
                {'type': 'LoadAnnotations',
                    'with_bbox': True,
                    'with_mask': True,
                    'poly2mask': True
                },
                {'type': 'RandomChoiceResize',
                'scales': img_scales,
                'keep_ratio': False
                },
                {"type": "RandomRandAugment",
                 "aug_space": [
                     [
                         {"type": "RandomColorFormCutOut",
                         "n_holes": (0,4),
                         "cutout_shape": [(16,16),(24,24),(32,32),(48,48),(64,64),(72,72)]
                        },
                     ],
                     [
                        {"type": "RandomColorFormCutOutAlpha",
                         "n_holes": (0,4),
                         "cutout_shape": [(16,16),(24,24),(32,32),(48,48),(64,64),(72,72)]
                        },
                     ],
                     [
                         {'type': 'RandomElasticDeformation',
                            'min_sigma': 0,
                            'max_sigma': 26,
                            'min_points': 1,
                            'max_points': 4},
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "gaussian_noise",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "shot_noise",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "defocus_blur",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "motion_blur",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "zoom_blur",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "snow",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "frost",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "fog",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "brightness",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "contrast",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "elastic_transform",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "pixelate",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'Corrupt',
                            "corruption": "jpeg_compression",
                            "severity": 1,
                        }
                     ],
                     [
                         {'type': 'PhotoMetricDistortion',
                            "brightness_delta": 32,
                            "contrast_range": (0.5, 1.5),
                            "saturation_range": (0.5, 1.5),
                            "hue_delta": 18
                        }
                     ],
                     [dict(type='Invert')], 
                     [dict(type='Solarize')],
                     [dict(type='SolarizeAdd')], 
                     [dict(type='Color')],
                     [dict(type='Contrast')], 
                     [dict(type='Brightness')],
                     [dict(type='Sharpness')], 
                     [dict(type='Rotate')],
                     [dict(type='ShearX')],
                     [dict(type='ShearY')], 
                     [dict(type='TranslateX')],
                     [dict(type='TranslateY')]
                 ],
                 "aug_num": 5,
                 
                },
                
            ],
            serialize_data=False,
            lazy_init=False)
                                                  
    train_pipeline = [
        dict(type="CutMix",
            ratio_range =  (0.5,1.5),
            flip_ratio =  0.5,
            pad_val =  114.0,
            max_iters =  15,
            bbox_clip_border =  True ,
            crop_min= 64,
            crop_max= 256,
            p = 0.3
            ),
        dict(type='PackDetInputs')
    ]

    cfg.train_dataloader.dataset = dict(
        type='MultiImageMixDataset',
        dataset= cfg.train_dataloader.dataset,
        pipeline=train_pipeline)

    cfg.train_dataloader.num_workers = 4
    cfg.train_dataloader.batch_size = 32
    cfg.train_pipeline = cfg.train_dataloader.dataset.pipeline

    
    cfg.val_dataloader.dataset.ann_file = '/home/jovyan/xray-models/predictions/anatomy_predictions_val.json'
    cfg.val_dataloader.dataset.data_root = '/home/jovyan/xray-data-datavol-1/MIMIC/'
    cfg.val_dataloader.dataset.data_prefix.img = 'images_512/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=img_scales[1], type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
            ]
    cfg.val_dataloader.num_workers = 4

    cfg.test_dataloader.dataset.ann_file = '/home/jovyan/xray-models/predictions/anatomy_predictions_val.json'
    cfg.test_dataloader.dataset.data_root = '/home/jovyan/xray-data-datavol-1/MIMIC/'
    cfg.test_dataloader.dataset.pipeline = cfg.val_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.data_prefix.img = 'images_512/'
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.test_dataloader.num_workers = 4
    

    cfg.test_pipeline = [
                                    dict(type='LoadImageFromFile', backend_args=None),
                                    dict(
                                        type='TestTimeAug',
                                        transforms=[
                                            [
                                                dict(type='Resize', scale=s, keep_ratio=False)
                                                for s in img_scales
                                            ],
                                            [
                                                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                                                # bounding box coordinates after flipping cannot be
                                                # recovered correctly.
                                                dict(type='RandomFlip', prob=1.),
                                                dict(type='RandomFlip', prob=0.)
                                            ],
                                            [
                                                dict(
                                                    type='Pad',
                                                    size= img_scales[-1],
                                                    pad_val=dict(img=(114, 114, 114))),
                                            ],
                                            [dict(type='LoadAnnotations', with_bbox=True)],
                                            [
                                                dict(
                                                    type='PackDetInputs',
                                                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                                               'scale_factor', 'flip', 'flip_direction'))
                                            ]
                                        ])
                                    ]
    cfg.tta_pipeline = cfg.test_pipeline
    cfg.tta_model = dict(
        type="DetTTAModel",
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=200))

    # Modify metric config
    cfg.val_evaluator.ann_file = '/home/jovyan/xray-models/predictions/anatomy_predictions_val.json'
    cfg.val_evaluator.metric = ["segm","bbox"]
    cfg.val_evaluator.classwise=True
    cfg.test_evaluator.ann_file =  '/home/jovyan/xray-models/predictions/anatomy_predictions_val.json'
    cfg.test_evaluator.metric = ["segm","bbox"]
    cfg.test_evaluator.classwise=True



    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 10
    cfg.default_hooks.logger.interval = 10

    cfg.work_dir = '/home/jovyan/xray-models/segmentation_models/anatomy/MIMIC_pseudolabels'
    cfg.seed = 0
    return cfg

def download_file(url, save_path):
    if os.path.exists(save_path):
        print("File already exists.")
        return
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        print("File downloaded successfully to:", save_path)
    else:
        print("Failed to download file")



    

def add_mask_rcnn(cfg, num_classes = 10, epoch=100, val_interval=10):
    cfg.train_cfg.max_epochs = epoch
    cfg.max_epochs = epoch
    cfg.train_cfg.val_interval = val_interval
    cfg.train_dataloader.batch_size = 32
    
    # Modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = num_classes
    cfg.model.roi_head.mask_head.num_classes = num_classes

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = '/home/jovyan/workspace-seg/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optim_wrapper.optimizer.lr = 0.02 / 8
    cfg.work_dir += 'maskrcnn'
    
    return cfg

def add_pointrend(cfg,num_classes=10, epoch=100, val_interval=10):
    
    cfg.train_dataloader.batch_size = 32
    cfg.train_cfg.val_interval = val_interval
    cfg.train_cfg.max_epochs = epoch
    cfg.max_epochs = epoch
    
    # Modify num classes of the model in box head and mask head
    cfg.model.roi_head.point_head.num_classes = num_classes
    cfg.model.roi_head.mask_head.num_classes = num_classes
    cfg.model.roi_head.bbox_head.num_classes = num_classes

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = '/home/jovyan/workspace-seg/mmdetection/checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
    download_file(
        "https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth"
        , '/home/jovyan/workspace-seg/mmdetection/checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth')
    
    optim_wrapper = dict(
                type='OptimWrapper',
                optimizer=dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05))
    
    cfg.work_dir += 'pointrend'
        
    return cfg

def add_sparseinst(cfg, num_classes=10, max_iter=15000,step1=12500,step2=14000):
    
    cfg.train_dataloader.batch_size = 16
    
    # Modify num classes of the model in box head and mask head
    cfg.model.decoder.num_classes = num_classes
    cfg.model.criterion.num_classes = num_classes

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = '/home/jovyan/workspace-seg/mmdetection/checkpoints/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth'
    download_file(
        "https://download.openmmlab.com/mmdetection/v3.0/sparseinst/sparseinst_r50_iam_8xb8-ms-270k_coco/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth"
        , '/home/jovyan/workspace-seg/mmdetection/checkpoints/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth')

    
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    
    cfg.default_hooks.checkpoint.interval = 1000
    cfg.default_hooks.logger.interval = 100
    cfg.train_cfg.val_interval = 1000
    
    cfg.train_cfg.max_iters = max_iter+1
    cfg.param_scheduler = dict(
                type='MultiStepLR',
                begin=0,
                end=cfg.train_cfg.max_iters,
                by_epoch=False,
                milestones=[step1, step2],
                gamma=0.1)
    
    
    cfg.work_dir += 'sparseinst'
    return cfg

def add_queryinst(cfg,num_classes=10, epoch=100, val_interval=10):
    cfg.train_cfg.max_epochs = epoch
    cfg.max_epochs = epoch
    cfg.train_cfg.val_interval = val_interval
    # Modify num classes of the model in box head and mask head
    # import pdb;pdb.set_trace()
    for i in range(6):
        cfg.model.roi_head.bbox_head[i].num_classes = num_classes
        cfg.model.roi_head.mask_head[i].num_classes = num_classes
    
    
    cfg.model.train_cfg.rcnn = [
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1,
                mask_size=28,
            ) for _ in range(6)
        ]
    
    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = '/home/jovyan/workspace-seg/mmdetection/checkpoints/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth'
    download_file(
        "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth"
        , '/home/jovyan/workspace-seg/mmdetection/checkpoints/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth')
    
    
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.work_dir += 'queryinst'
    return cfg

def add_mask2former(cfg, num_classes=10, max_iter=15000,step1=12500,step2=14000, warmup=5000, val_steps = 1000, size = (640, 640), batchsize = 16, queries=100, sampler='InfiniteSampler'):
        
    cfg.train_dataloader.batch_size = batchsize
#    cfg.train_dataloader.sampler = dict(shuffle=True, type=sampler)
    cfg.train_dataloader.sampler = dict(type=sampler)
    # Modify num classes of the model in box head and mask head
    cfg.model.panoptic_head.num_things_classes = num_classes
    
    cfg.model.panoptic_head.num_queries = queries
    cfg.model.test_cfg.max_per_image = queries
    
    cfg.model.panoptic_head.loss_cls.class_weight = [1.0] * cfg.model.panoptic_head.num_things_classes + [0.1]
    cfg.model.panoptic_fusion_head.num_things_classes = num_classes

    cfg.batch_augments[0].size = size
    cfg.data_preprocessor.batch_augments[0].size = size
    cfg.image_size = size
    cfg.model.data_preprocessor.batch_augments[0].size = size
    
    cfg.num_classes = num_classes
    cfg.num_things_classes = num_classes
    
    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = '/home/jovyan/workspace-seg/mmdetection/checkpoints/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'
    download_file(
        "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
        , '/home/jovyan/workspace-seg/mmdetection/checkpoints/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth')

    cfg.default_hooks.checkpoint.interval = val_steps
    cfg.default_hooks.logger.interval = 100
    cfg.train_cfg.val_interval = val_steps

    
    # import pdb;pdb.set_trace()
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.train_cfg.max_iters = max_iter+1
    cfg.param_scheduler = [
                # Linear learning rate warm-up scheduler
                dict(type='LinearLR',
                     start_factor=0.000001,
                     by_epoch=False,  # Updated by iterations
                     begin=0,
                     end=warmup),  # Warm up for the first 50 iterations
                # The main LRScheduler
                dict(
                type='MultiStepLR',
                begin=0,
                end=cfg.train_cfg.max_iters,
                by_epoch=False,
                milestones=[step1, step2],
                gamma=0.1)
            ]

                
    
    
    cfg.work_dir += 'mask2former'
    return cfg

def add_rtmdet(cfg, num_classes=10, epoch=200, val_interval=10):
    cfg.train_dataloader.batch_size = 12
    cfg.train_cfg.max_epochs = epoch
    cfg.max_epochs = epoch
    cfg.train_cfg.val_interval = val_interval
    # Modify num classes of the model in box head and mask head
    cfg.model.bbox_head.num_classes = num_classes
    
    cfg.train_dataloader.dataset.pipeline = [
                            {'type': 'LoadImageFromFile', 'backend_args': None},
                            {'type': 'LoadAnnotations',
                            'with_bbox': True,
                            'with_mask': True,
                            'poly2mask': False},
                            {"type": "RandAugment",
                             "aug_num": 2,
                            },
                            {'type': 'RandomChoiceResize',
                            'scales': [(640,640)],
                            'keep_ratio': False},
                            {"type":"RandomCrop",
                             "crop_size": (640,640),                            
                            },
                            {'type': 'RandomFlip', 'prob': 0.5},
                            {'type': 'PackDetInputs'}
                        ]
    
    cfg.val_dataloader.dataset.pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(640,640), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
            ]
    
    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = '/home/jovyan/workspace-seg/mmdetection/checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
    download_file(
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
        , '/home/jovyan/workspace-seg/mmdetection/checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth')

    
    cfg.work_dir += 'rtmdet'
    return cfg


file_name = '../configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'
cfg = Config.fromfile(file_name)
cfg = add_mimic_anatomy_cfg(cfg)
cfg = add_mask2former(cfg, 
                       num_classes=128, 
                       max_iter=200000,
                       step1=100000,
                       step2=140000,
                       warmup=30000,
                       size = (512, 512), 
                       batchsize = 16, 
                       queries=200)
cfg.load_from="~/xray-models/segmentation_models/anatomy/MIMIC_3randaugmask2former/iter_164000.pth"
config=file_name.replace("coco", "mimic")
with open(config, 'w') as f:
    f.write(cfg.pretty_text)
