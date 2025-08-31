from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

## Deepchecks
class DeepchecksResultHeaders(Enum):
    # Train-Test Validation
    LabelDrift = "Label Drift"
    ImageDatasetDrift = "Image Dataset Drift"
    ImagePropertyDrift = "Image Property Drift"
    PropertyLabelCorrelationChange = "Property Label Correlation Change"
    HeatmapComparison = "Heatmap Comparison"
    NewLabels = "New Labels"
    # Data Integrity
    ImagePropertyOutliers = "Image Property Outliers"
    PropertyLabelCorrelation = "Property Label Correlation"
    LabelPropertyOutliers = "Label Property Outliers"

class DeepchecksParsedResult(BaseModel):
    header: DeepchecksResultHeaders = Field(description="Header of the result")
    json: Dict[str,Any] = Field(description="JSON result of the result")
    display_images: Optional[List[str]] = Field(default=None,description="Display images of the result as base64 encoded strings")
    display_txt: Optional[str] = Field(default=None,description="Display text of the result")

    def to_dict(self)->Dict[str,Any]:
        dumped_dict = self.model_dump()
        dumped_dict["header"] = dumped_dict["header"].value
        return dumped_dict
    
    @classmethod
    def from_dict(self,d:Dict[str,Any])->"DeepchecksParsedResult":
        return DeepchecksParsedResult(header=DeepchecksResultHeaders(d["header"]),
                            json_result=d["json_result"],
                            display_images=d["display_images"],
                            display_txt=d["display_txt"])

class DeepchecksArtifact(BaseModel):
    dataset_name: str = Field(description="Name of the dataset")
    results: Dict[str,List[DeepchecksParsedResult]] = Field(description="Results of the artifact")

    def to_dict(self)->Dict[str,Any]:
        dumped_dict = self.model_dump()
        dumped_dict["results"] = {k:[r.to_dict() for r in v] for k,v in self.results.items()}
        return dumped_dict
    
    @classmethod
    def from_dict(self,d:Dict[str,Any])->"DeepchecksArtifact":
        return DeepchecksArtifact(dataset_name=d["dataset_name"],
                            results={k:DeepchecksParsedResult.from_dict(v) for k,v in d["results"].items()})

## Dataset
class ClassificationDataElement(BaseModel):
    index: int = Field(description="Index of the data element")
    embedding: List[float] = Field(description="Embedding of the data element")
    label: int = Field(description="Label of the data element")
    prediction: Optional[int] = Field(default=None,description="Prediction of the data element")
    probabilities: Optional[List[float]] = Field(default=None,description="Probabilities of the data element")

class ClassificationDataset(BaseModel):
    dataset_name: str = Field(description="Name of the dataset")
    data: List[ClassificationDataElement] = Field(description="Data of the dataset")
    embedding_model: Optional[str] = Field(default=None,description="Name of the embedding model used to generate the embeddings")
    embedding_model_params: Optional[Dict[str,Any]] = Field(default=None,description="Params of the embedding model")
