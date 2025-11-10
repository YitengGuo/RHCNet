# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy: 按原比例在 35 轮下进行 step 降级
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 30]
)

# 训练总轮数改为 35
runner = dict(type='EpochBasedRunner', max_epochs=35)
