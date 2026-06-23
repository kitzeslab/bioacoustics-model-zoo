from opensoundscape import ONNXModel
import huggingface_hub
import pandas as pd

PERCH2_ONNX_HF_HANDLES = {
    "with_classifier": {
        "repo": "justinchuby/Perch-onnx",
        "checkpoint": "perch_v2.onnx",
        "no_dft_checkpoint": "perch_v2_no_dft.onnx",
    },
    "without_classifier": {
        "repo": "sammlapp/Perch_v2_headless",
        "checkpoint": "perch_v2_embedding_only.onnx",
        "no_dft_checkpoint": "perch_v2_no_dft_embedding_only.onnx",
    },
}
class Perch2ONNX(ONNXModel):
    def __init__(self,headless=False,no_dft=True):
        """Initialize Perch V2 ONNX model, downloading checkpoint from HuggingFace
        
        Args:
            headless (bool): If True, load the model without the classifier head
                for efficient embedding workflows.
            no_dft (bool): If True, load the model with DFT operation flattened for
                enhanced compatibility with ONNX runtimes that do not support DFT
                operations. False is the default as it appears to be slightly more efficient.
        """
        key = "without_classifier" if headless else "with_classifier"
        checkpoint_key = "no_dft_checkpoint" if no_dft else "checkpoint"
        model_path = PERCH2_ONNX_HF_HANDLES[key][checkpoint_key]
        # download or get local cached path to model from HuggingFace Hub
        model_path = huggingface_hub.hf_hub_download(
            repo_id=PERCH2_ONNX_HF_HANDLES[key]["repo"],
            filename=model_path,
        )
        # load class list
        if headless:
            classes = []
        else:
            labels_path = huggingface_hub.hf_hub_download(
                repo_id="sammlapp/Perch_v2_headless",
                filename="labels.csv",
            )
            classes = pd.read_csv(labels_path).values[:, 0].tolist()
        # Note: this will require onnx runtime in the environment
        super().__init__(onnx_model_path=model_path, sample_rate=32000, sample_duration=5.0, classes=classes, class_outputs_key="label")
        self.add_channel_dim=False
        self.embedding_outputs_key = 'embedding'
        from opensoundscape.preprocess.actions import Action
        from opensoundscape import Audio
        normalize_action = Action(Audio.normalize, peak_level=0.25)
        self.preprocessor.insert_action('normalize', normalize_action)

    