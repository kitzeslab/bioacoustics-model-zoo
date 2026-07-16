"""
Example Usage:
geo = BirdNETGeomodel()
local_sp = geo(40.7128, -74.0060, 20) # Example for New York City, week 20
subset_classifier_labels("HawkEars", local_sp)
"""
import numpy as np
import pandas as pd
import huggingface_hub

def load_labels(labels_path):
    """Load species labels from labels.txt."""
    labels = []
    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            labels.append({"code": parts[0], "sci": parts[1], "common": parts[2]})
    return labels

class BirdNETGeomodel:
    def __init__(self,version="3.0.3"):
        """BirdNET 3 GeoModel for predicting species presence probabilities based on latitude, longitude, and week of the year.
        
        This model is developed by the BirdNET team andprovided under an MIT license from https://github.com/birdnet-team/geomodel
        
        Usage:
        
        ```python
        import bioacoustics_model_zoo as bmz
        geo = bmz.BirdNETGeomodel()

        # pass latitude, longitude, week of year; returns dataframe with species probabilities
        local_sp = geo(40.7128, -74.0060, 20) 

        # batch processing: pass array of [lat, lon, week] rows; returns list of dicts with species probabilities
        batched = geo.batched_predict(np.array([[40.7128, -74.0060, 20], [40.7128, -74.0060, 21]]))
        ```

        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required to use BirdNETGeomodel. Please install it with `pip install onnxruntime`.")
        labels_path = huggingface_hub.hf_hub_download(repo_id="sammlapp/BirdNET_GeoModel", filename=f"BirdNET+_Geomodel_V{version}_Global_12K_Labels.txt")
        onnx_path = huggingface_hub.hf_hub_download(repo_id="sammlapp/BirdNET_GeoModel", filename=f"BirdNET+_Geomodel_V{version}_Global_12K_FP32.onnx")

        self.session = ort.InferenceSession(onnx_path)
        self.labels = load_labels(labels_path)

    def __call__(self, lat, lon, week, min_probability=0.05):
        """Generate species presence probabilities for a single lat/lon/week 

        See also: self.batched_predict for batch processing.

        """
        inputs = np.array([[lat, lon, week]], dtype=np.float32)

        probs = self.session.run(None, {"input": inputs})[0] # batch size 1, get first element

        probs_df = pd.DataFrame({
            "ebird_code": [label["code"] for label in self.labels],
            "scientific_name": [label["sci"] for label in self.labels],
            "common_name": [label["common"] for label in self.labels],
            "probability": probs[0]
        })

        if min_probability is not None:
            probs_df = probs_df[probs_df["probability"] >= min_probability]
    
        return probs_df.sort_values(by="probability", ascending=False)
    
    def batched_predict(self, lat_lon_week_array, min_probability=0.05):
        """Generate species presence probabilities for a batch of lat/lon/week inputs.

        lat_lon_week_array: numpy array of shape (N, 3) where each row is [lat, lon, week]

        Returns a list of dictionaries:
            {
                "latitude":,
                "longitude":,
                "week":,
                "probabilities": [
                    {
                        "ebird_code":,
                        "scientific_name":,
                        "common_name":,
                        "probability":
                    },
                    ...
                ]
            }
        """
        # try casting input to numpy array (allows for list input)
        lat_lon_week_array = np.array(lat_lon_week_array, dtype=np.float32)
        probs = self.session.run(None, {"input": lat_lon_week_array})[0]

        results = []
        if min_probability is None:
            min_probability = -1
        for i, probs_i in enumerate(probs):
            filtered = [
                {
                    "ebird_code": label["code"],
                    "scientific_name": label["sci"],
                    "common_name": label["common"],
                    "probability": prob
                }
                for label, prob in zip(self.labels, probs_i)
                if prob >= min_probability
            ]

            results.append({
                "latitude": lat_lon_week_array[i, 0],
                "longitude": lat_lon_week_array[i, 1],
                "week": lat_lon_week_array[i, 2],
                "probabilities": filtered
            })
        return results
