"""
Deepchecks integration for automated model validation.

This module provides comprehensive Deepchecks integration including:
- Pre-configured computer vision test suites
- Custom overfitting detection checks
- Batch processing workflows
- Result parsing and analysis
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from deepchecks.vision.suites import train_test_validation,data_integrity,model_evaluation
from deepchecks.vision import VisionData
from deepchecks.core.suite import SuiteResult
from deepchecks.core.checks import CheckResult
import json
from pathlib import Path
from enum import Enum
from io import BytesIO
from PIL import Image

from ..utils.config import DeepchecksConfig
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class DeepchecksRunner:
    """
    Deepchecks integration for automated model validation and testing.
    
    Provides high-level interface for running Deepchecks suites and custom
    checks specifically designed for overfitting detection in CV models.
    """
    
    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration.
        """
        self.config = config or DeepchecksConfig()

        self.suite_train_test_validation = train_test_validation()
        self.suite_data_integrity = data_integrity()
        self.suite_model_evaluation = model_evaluation()
        self.output_dir = Path(self.config.output_dir or 'results')
        
    
    def save_results(self, results: SuiteResult, output_path: str,output_format: str = "json")->None:
        if output_format == "json":
            with open(output_path, 'w') as f:
                json.dump(json.loads(results.to_json(with_display=self.config.save_display)), f, indent=3)
            LOGGER.info(f"Results saved to {output_path}")
        elif output_format == "html":
            results.save_as_html(output_path)
            LOGGER.info(f"Results saved to {output_path}")
        else:
            raise ValueError(f"Invalid output format: {output_format}")

    def run_suites(self, train_data: VisionData, 
                   test_data: Optional[VisionData] = None,
                   )->Dict[str, SuiteResult]:

        output = {}
        if self.config.train_test_validation:
            out_train_test_validation = self.run_suite_train_test_validation(train_data, test_data)
            output['train_test_validation'] = out_train_test_validation

        if self.config.data_integrity:
            out_data_integrity = self.run_suite_data_integrity(train_data, test_data)
            output['data_integrity'] = out_data_integrity

        if self.config.model_evaluation:
            out_model_evaluation = self.run_suite_model_evaluation(train_data, test_data)
            output['model_evaluation'] = out_model_evaluation
        
        if self.config.save_results:
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)

            for name, result in output.items():
                output_path = self.output_dir / f"{name}.{self.config.save_results_format}"
                self.save_results(result, str(output_path), output_format=self.config.save_results_format)
                LOGGER.info(f"Results saved to {output_path} in {self.config.save_results_format} format")
        return output
        
    def run_suite_train_test_validation(self, 
                                train_data: VisionData, 
                                test_data: Optional[VisionData] = None)->SuiteResult:
        LOGGER.info("Running train-test validation suite")
        return self.suite_train_test_validation.run(train_dataset=train_data, 
        test_dataset=test_data,
        max_samples=self.config.max_samples,
        random_state=self.config.random_state)

    def run_suite_data_integrity(self, train_data: VisionData, 
                                test_data: Optional[VisionData] = None)->SuiteResult:
        LOGGER.info("Running data integrity suite")
        return self.suite_data_integrity.run(train_dataset=train_data, 
        test_dataset=test_data,
        max_samples=self.config.max_samples,
        random_state=self.config.random_state)

    def run_suite_model_evaluation(self, train_data: VisionData, 
                                test_data: Optional[VisionData] = None)->SuiteResult:
        LOGGER.info("Running model evaluation suite")
        return self.suite_model_evaluation.run(train_dataset=train_data, 
        test_dataset=test_data,
        max_samples=self.config.max_samples,
        random_state=self.config.random_state)


class ResultHeaders(Enum):
    LabelDrift = "Label Drift"
    ImageDatasetDrift = "Image Dataset Drift"
    ImagePropertyDrift = "Image Property Drift"
    PropertyLabelCorrelationChange = "Property Label Correlation Change"
    HeatmapComparison = "Heatmap Comparison"
    NewLabels = "New Labels"
    


class CheckResultsParser:

    def __init__(self, results: SuiteResult):
        self.results = results

    def parse_txt(self):
        parsed_results = {}
        for result in self.results.results:
            header = result.get_metadata().get('header')
            parsed_results[header] = json.loads(result.to_json(with_display=False))
        return parsed_results
    
    def parse_display(self,result:CheckResult)->Dict[ResultHeaders,Dict[str,Union[List[Image.Image],str]]]:
        if not result.have_display():
            return None

        display_result = {}
        
        if result.get_metadata().get('header') == ResultHeaders.LabelDrift.value:
            image,txt = self._parse_display(result)
            display_result[ResultHeaders.LabelDrift] = {'image':image, 'txt':txt}

        elif result.get_metadata().get('header') == ResultHeaders.ImageDatasetDrift.value:
            image,txt = self._parse_display(result)
            display_result[ResultHeaders.ImageDatasetDrift] = {'image':image, 'txt':txt}

        elif result.get_metadata().get('header') == ResultHeaders.ImagePropertyDrift.value:
            image,txt = self._parse_display(result)
            display_result[ResultHeaders.ImagePropertyDrift] = {'image':image, 'txt':txt}

        elif result.get_metadata().get('header') == ResultHeaders.PropertyLabelCorrelationChange.value:
            image,txt = self._parse_display(result)
            display_result[ResultHeaders.PropertyLabelCorrelationChange] = {'image':image, 'txt':txt}

        elif result.get_metadata().get('header') == ResultHeaders.HeatmapComparison.value:
            image,txt = self._parse_display(result)
            display_result[ResultHeaders.HeatmapComparison] = {'image':image, 'txt':txt}

        elif result.get_metadata().get('header') == ResultHeaders.NewLabels.value:
            image,txt = self._parse_display(result)
            display_result[ResultHeaders.NewLabels] = {'image':image, 'txt':txt}
        
        else:
            raise ValueError(f"Unknown header: {result.get_metadata().get('header')}")

        return display_result

    def _load_display_as_image(self,result:CheckResult)-> List[Image.Image]:
        images = []
        for d in result.display:
            if hasattr(d,'to_image'):
                image = BytesIO(d.to_image())
                images.append(Image.open(image))
        return images
    
    def _parse_display_txt(self,result:CheckResult)->List[str]:
        txts = [d.replace("<span>","").replace("</span>","") for d in txts if isinstance(d,str)]
        txts = " ".join(txts)
        return txts
    
    def _parse_display(self,result:CheckResult)->Tuple[List[Image.Image],str]:
        images = self._load_display_as_image(result)
        txt = self._parse_display_txt(result)
        return images,txt

    
    