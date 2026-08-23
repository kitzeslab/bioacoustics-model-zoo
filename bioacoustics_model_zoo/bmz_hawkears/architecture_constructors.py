# these constructors are copied from hawkears repo tag 0.1.0

# Define custom hgnet_v2 configurations.
# Is hgnet from the following paper? https://arxiv.org/pdf/2205.00841.pdf
import timm

from timm.models import hgnet, byobnet, vovnet, efficientnet, dla, fastvit, mobilenetv3


# return a new model based on the given name
def create_model(model_name, num_classes, pretrained=False):
    # create a dict of optional keyword arguments to pass to model creation
    # I haven't copied all of the architectures from HawkEars repo, only those used in models v1.0.0-1.0.8

    # create the model
    if model_name.startswith("custom_dla"):
        tokens = model_name.split("_")
        model = get_dla(tokens[-1], num_classes=num_classes)
    elif model_name.startswith("custom_efficientnet"):
        tokens = model_name.split("_")
        model = build_efficientnet_architecture(tokens[-1], num_classes=num_classes)
    elif model_name.startswith("custom_fastvit"):
        tokens = model_name.split("_")
        model = get_fastvit(tokens[-1], num_classes=num_classes)
    elif model_name.startswith("custom_gernet"):
        tokens = model_name.split("_")
        model = get_gernet(tokens[-1], num_classes=num_classes)
    elif model_name.startswith("custom_hgnet"):
        tokens = model_name.split("_")
        model = get_hgnet(tokens[-1], num_classes=num_classes)
    elif model_name.startswith("custom_mobilenet"):
        tokens = model_name.split("_")
        model = get_mobilenet(tokens[-1], num_classes=num_classes)
    elif model_name.startswith("custom_vovnet"):
        tokens = model_name.split("_")
        model = get_vovnet(tokens[-1], num_classes=num_classes)
    else:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
        )
    return model


# all models are mobilenetv3_large_100 with different channel multipliers
def get_mobilenet(model_name, **kwargs):
    if model_name == "0":
        # ~200K parameters
        channel_multiplier = 0.1
    elif model_name == "1":
        # ~1.5M parameters
        channel_multiplier = 0.5
    elif model_name == "2":
        # ~2.4M parameters
        channel_multiplier = 0.7
    elif model_name == "2B":
        # ~3.2M parameters
        channel_multiplier = 0.8
    elif model_name == "3":
        # ~4.2M parameters
        channel_multiplier = 1.0  # i.e. this is mobilenetv3_large_100
    elif model_name == "4":
        # ~5.0M parameters
        channel_multiplier = 1.1
    elif model_name == "5":
        # ~5.8M parameters
        channel_multiplier = 1.15
    elif model_name == "6":
        # ~6.3M parameters
        channel_multiplier = 1.25
    elif model_name == "7":
        # ~7.2M parameters
        channel_multiplier = 1.35
    elif model_name == "8":
        # ~8.5M parameters
        channel_multiplier = 1.5
    else:
        raise Exception(f"Unknown custom Mobilenet model name: {model_name}")

    # TODO: switch to V4, which trains even faster but gets similar precision/recall
    model = mobilenetv3._gen_mobilenet_v3(
        "mobilenetv3_large_100",
        channel_multiplier,
        pretrained=False,
        in_chans=1,
        **kwargs,
    )

    return model


def get_fastvit(model_name, **kwargs):
    if model_name == "1":
        # about 1.6M parameters
        model_args = dict(
            layers=(1, 1, 1, 1),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "2A":
        # about 2.8M parameters
        model_args = dict(
            layers=(2, 2, 2, 2),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "2B":
        # about 3.0M parameters
        model_args = dict(
            layers=(2, 2, 3, 2),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "3":
        # about 3.3M parameters (this is fastvit_t8)
        model_args = dict(
            layers=(2, 2, 4, 2),
            embed_dims=(48, 96, 192, 384),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "4":
        # about 4.8M parameters
        model_args = dict(
            layers=(1, 1, 2, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "5":
        # about 5.6M parameters
        model_args = dict(
            layers=(2, 2, 3, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "6":
        # about 6.6M parameters (this is fastvit_t12)
        model_args = dict(
            layers=(2, 2, 6, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(3, 3, 3, 3),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "7":
        # about 7.1M parameters
        model_args = dict(
            layers=(2, 2, 8, 4),
            embed_dims=(64, 128, 256, 256),
            mlp_ratios=(4, 4, 4, 4),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    elif model_name == "8":
        # about 8.5M parameters (this is fastvit_s12)
        model_args = dict(
            layers=(2, 2, 6, 2),
            embed_dims=(64, 128, 256, 512),
            mlp_ratios=(4, 4, 4, 4),
            token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
        )
    else:
        raise Exception(f"Unknown custom FastViT model name: {model_name}")

    return fastvit._create_fastvit(
        "fastvit_t8", pretrained=False, in_chans=1, **dict(model_args, **kwargs)
    )


"""
# timm's eca_vovnet39b has 21.6M parameters with 23 classes and uses this config:
config = dict(
    stem_chs=[64, 64, 128],
    stage_conv_chs=[128, 160, 192, 224],
    stage_out_chs=[256, 512, 768, 1024],
    layer_per_block=5,
    block_per_stage=[1, 1, 2, 2],
    residual=True,
    depthwise=False,
    attn='eca',
)

# timm's ese_vovnet19b_dw has 5.5M parameters with 23 classes and uses this config:
config = dict(
    stem_chs=[64, 64, 64],
    stage_conv_chs=[128, 160, 192, 224],
    stage_out_chs=[256, 512, 768, 1024],
    layer_per_block=3,
    block_per_stage=[1, 1, 1, 1],
    residual=True,
    depthwise=True,
    attn='ese',
)
"""


# Default parameters to VovNet:
#    global_pool='avg',
#    output_stride=32,
#    norm_layer=BatchNormAct2d,
#    act_layer=nn.ReLU,
#    drop_rate=0.,
#    drop_path_rate=0., # stochastic depth drop-path rate
def get_vovnet(model_name, **kwargs):
    if model_name == "1":
        #  about 2.1M parameters
        config = dict(
            stem_chs=[32, 32, 64],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=1,
            block_per_stage=[1, 1, 1, 1],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "2":
        #  about 3.1M parameters
        config = dict(
            stem_chs=[32, 32, 32],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 384],
            layer_per_block=1,
            block_per_stage=[1, 1, 1, 2],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "3":
        #  about 3.7M parameters
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=2,
            block_per_stage=[1, 1, 1, 1],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "4":
        #  about 4.4M parameters
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=2,
            block_per_stage=[1, 2, 1, 1],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "5":
        # about 5.6M parameters
        config = dict(
            stem_chs=[32, 32, 64],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 384],
            layer_per_block=2,
            block_per_stage=[1, 2, 2, 1],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "6":
        # about 6.2M parameters
        config = dict(
            stem_chs=[32, 32, 64],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 384],
            layer_per_block=2,
            block_per_stage=[1, 1, 2, 2],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "7":
        # about 7.6M parameters (they get much slower with layers_per_block=3 though)
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=3,
            block_per_stage=[1, 1, 1, 2],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    elif model_name == "8":
        # about 9.3M parameters
        config = dict(
            stem_chs=[64, 64, 128],
            stage_conv_chs=[128, 160, 192, 224],
            stage_out_chs=[128, 256, 384, 512],
            layer_per_block=3,
            block_per_stage=[1, 1, 2, 2],
            residual=True,
            depthwise=False,
            attn="eca",
        )
    else:
        raise Exception(f"Unknown custom VovNet model name: {model_name}")

    model = vovnet.VovNet(cfg=config, in_chans=1, **kwargs)

    return model


# timm's dla34 has 15.2M parameters with 23 classes and uses this config:
"""
config = dict(
    levels=[1, 1, 1, 2, 2, 1],
    channels=[16, 32, 64, 128, 256, 512],
    block=DlaBasic)
"""

# Config parameters are:
#   base_width=n                # default is 64
#   block=DlaBasic/DlaBottleneck/DlaBottle2neck
#   cardinality=n               # default is 1
#   shortcut_root=True/False    # default is False


# Default arguments:
#    global_pool='avg',
#    output_stride=32,
#    drop_rate=0.,
#
# The following are all smaller variations of dla34.
#
def get_dla(model_name, **kwargs):
    if model_name == "0":
        # ~630K parameters
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 64, 64, 64],
            block=dla.DlaBasic,
        )
    elif model_name == "1":
        # ~1.5M parameters
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 64, 128, 128],
            block=dla.DlaBasic,
        )
    elif model_name == "2":
        # ~3.0M parameters
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 96, 96, 256],
            block=dla.DlaBasic,
        )
    elif model_name == "3":
        # DlaBottleneck (~4.0M parameters)
        config = dict(
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBottleneck,
        )
    elif model_name == "4":
        # DlaBottle2neck (~4.9M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 2, 1],
            channels=[16, 32, 64, 64, 256, 256],
            block=dla.DlaBottle2neck,
        )
    elif model_name == "5":
        # DlaBottle2Neck (~6.0M parameters)
        config = dict(
            levels=[1, 1, 1, 2, 1, 1],
            channels=[16, 32, 64, 128, 256, 384],
            block=dla.DlaBottle2neck,
        )
    elif model_name == "6":
        # DlaBottle2Neck (~7.2M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 2, 1],
            channels=[16, 32, 64, 128, 256, 384],
            block=dla.DlaBottle2neck,
        )
    elif model_name == "7":
        # DlaBottle2Neck (~8.2M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBottle2neck,
        )
    elif model_name == "8":
        # dla34 but DlaBottle2neck (~10.2M parameters)
        config = dict(
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBottle2neck,
        )
    elif model_name == "9":
        # levels are 1 (~12.0M parameters)
        config = dict(
            levels=[1, 1, 1, 1, 1, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block=dla.DlaBasic,
        )
    else:
        raise Exception(f"Unknown custom DLA model name: {model_name}")

    model = dla.DLA(in_chans=1, **config, **kwargs)

    return model


def get_gernet(model_name, **kwargs):
    if model_name == "1":
        # ~550K parameters
        config = byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="bottle", d=3, c=64, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type="bottle", d=2, c=128, s=2, gs=1, br=3.0),
                byobnet.ByoBlockCfg(type="bottle", d=1, c=64, s=1, gs=1, br=3.0),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    elif model_name == "2":
        # ~1.5M parameters
        config = byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="bottle", d=3, c=128, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type="bottle", d=2, c=256, s=2, gs=1, br=3.0),
                byobnet.ByoBlockCfg(type="bottle", d=1, c=128, s=1, gs=1, br=3.0),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    elif model_name == "3":
        # ~3.3M parameters
        config = byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="bottle", d=3, c=256, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type="bottle", d=2, c=384, s=2, gs=1, br=3.0),
                byobnet.ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    elif model_name == "6":
        # ~6.4M parameters (gernet_s)
        config = byobnet.ByoModelCfg(
            blocks=(
                byobnet.ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
                byobnet.ByoBlockCfg(type="bottle", d=7, c=384, s=2, gs=0, br=1 / 4),
                byobnet.ByoBlockCfg(type="bottle", d=2, c=560, s=2, gs=1, br=3.0),
                byobnet.ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
            ),
            stem_chs=13,
            stem_pool=None,
            num_features=1920,
        )
    else:
        raise Exception(f"Unknown custom gernet model name: {model_name}")

    model = byobnet.ByobNet(config, in_chans=1, **kwargs)

    return model


# this function is hgnet_v2.get_model() in the repo
def get_hgnet(model_name, **kwargs):
    if model_name == "1":
        # custom config with ~830K parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 32, 1, False, False, 3, 1],
            "stage2": [32, 32, 64, 1, True, False, 3, 1],
            "stage3": [64, 64, 128, 2, True, False, 3, 1],
            "stage4": [128, 64, 256, 1, True, True, 5, 1],
        }
    elif model_name == "2":
        # custom config with ~1.6M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 1],
            "stage2": [64, 32, 128, 1, True, False, 3, 1],
            "stage3": [128, 64, 256, 2, True, True, 5, 1],
            "stage4": [256, 64, 512, 1, True, True, 5, 1],
        }
    elif model_name == "3A":
        # custom config with ~2.8M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 128, 1, True, False, 3, 3],
            "stage3": [128, 64, 384, 2, True, True, 5, 3],
            "stage4": [384, 96, 768, 1, True, True, 5, 3],
        }
    elif model_name == "3B":
        # custom config with ~3.1M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
            "stage4": [512, 96, 768, 1, True, True, 5, 3],
        }
    elif model_name == "4":
        # this is hgnetv2_b0, with ~4.0M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [16, 16],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
            "stage4": [512, 128, 1024, 1, True, True, 5, 3],
        }
    elif model_name == "5":
        # this is hgnetv2_b1, with ~4.3M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 3],
            "stage2": [64, 48, 256, 1, True, False, 3, 3],
            "stage3": [256, 96, 512, 2, True, True, 5, 3],
            "stage4": [512, 192, 1024, 1, True, True, 5, 3],
        }
    elif model_name == "6":
        # custom config with ~4.6M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 4],
            "stage2": [64, 48, 256, 1, True, False, 3, 4],
            "stage3": [256, 96, 512, 2, True, True, 5, 4],
            "stage4": [512, 192, 1024, 1, True, True, 5, 4],
        }
    elif model_name == "7":
        # custom config with ~5.7M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 4],
            "stage2": [64, 48, 256, 1, True, False, 3, 4],
            "stage3": [256, 96, 512, 3, True, True, 5, 4],
            "stage4": [512, 192, 1024, 1, True, True, 5, 4],
        }
    elif model_name == "7B":
        # same as #7 except for larger kernels
        config = {
            "stem_type": "v2",
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 5, 4],
            "stage2": [64, 48, 256, 1, True, False, 5, 4],
            "stage3": [256, 96, 512, 3, True, True, 7, 4],
            "stage4": [512, 192, 1024, 1, True, True, 7, 4],
        }
    elif model_name == "8":
        # custom config with ~6.1M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 96, 1, False, False, 3, 4],
            "stage2": [96, 64, 384, 1, True, False, 3, 4],
            "stage3": [384, 128, 512, 3, True, True, 5, 4],
            "stage4": [512, 192, 1024, 1, True, True, 5, 4],
        }
    elif model_name == "9":
        # custom config with ~6.7M parameters
        config = {
            "stem_type": "v2",
            "stem_chs": [24, 32],
            # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [32, 32, 64, 1, False, False, 3, 3],
            "stage2": [64, 48, 256, 1, True, False, 3, 3],
            "stage3": [256, 128, 768, 1, True, True, 5, 3],
            "stage4": [768, 256, 1536, 1, True, True, 5, 3],
        }
    else:
        raise Exception(f"Unknown custom hgnet_v2 model name: {model_name}")

    model = hgnet.HighPerfGpuNet(cfg=config, in_chans=1, **kwargs)

    return model


# Define custom EfficientNet_v2 configurations.


def build_efficientnet_architecture(model_name, **kwargs):
    if model_name == "1":
        # ~ 1.5M parameters
        channel_multiplier = 0.4
        depth_multiplier = 0.4
    elif model_name == "2":
        # ~ 2.0M parameters
        channel_multiplier = 0.4
        depth_multiplier = 0.5
    elif model_name == "3":
        # ~ 3.4M parameters
        channel_multiplier = 0.5
        depth_multiplier = 0.6
    elif model_name == "4":
        # ~ 4.8M parameters
        channel_multiplier = 0.6
        depth_multiplier = 0.6
    elif model_name == "5":
        # ~ 5.7M parameters
        channel_multiplier = 0.6
        depth_multiplier = 0.7
    elif model_name == "6":
        # ~ 7.5M parameters
        channel_multiplier = 0.7
        depth_multiplier = 0.7
    elif model_name == "7":
        # ~ 8.3M parameters
        channel_multiplier = 0.7
        depth_multiplier = 0.8
    else:
        raise Exception(f"Unknown custom EfficientNetV2 model name: {model_name}")

    arch = efficientnet._gen_efficientnetv2_s(
        "efficientnetv2_rw_t",
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        in_chans=1,
        **kwargs,
    )

    arch.classifier_layer = "classifier"
    arch.embedding_layer = "global_pool"
    arch.cam_layer = "conv_head"

    return arch
