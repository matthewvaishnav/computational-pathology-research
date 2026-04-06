"""
Python client for Computational Pathology API.

This module provides a convenient Python interface for interacting with
the deployed API, including batch processing, error handling, and retries.

Example:
    >>> from examples.api_client import PathologyAPIClient
    >>> 
    >>> client = PathologyAPIClient("http://localhost:8000")
    >>> 
    >>> # Single prediction
    >>> result = client.predict(
    ...     wsi_features=wsi_data,
    ...     genomic=genomic_data,
    ...     clinical_text=clinical_data
    ... )
    >>> print(f"Predicted class: {result['predicted_class']}")
    >>> 
    >>> # Batch prediction
    >>> results = client.batch_predict(samples)
"""

import requests
import time
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from a single prediction."""
    predicted_class: int
    confidence: float
    probabilities: List[float]
    available_modalities: List[str]
    
    def __repr__(self):
        return (f"PredictionResult(class={self.predicted_class}, "
                f"confidence={self.confidence:.3f}, "
                f"modalities={self.available_modalities})")


class PathologyAPIClient:
    """
    Client for Computational Pathology API.
    
    Provides methods for making predictions, checking health, and batch processing.
    
    Args:
        base_url: Base URL of the API (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retries for failed requests (default: 3)
        retry_delay: Delay between retries in seconds (default: 1)
    
    Example:
        >>> client = PathologyAPIClient("http://localhost:8000")
        >>> health = client.health_check()
        >>> print(health)
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        logger.info(f"Initialized API client for {self.base_url}")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
        
        Returns:
            Response object
        
        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
            
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise
                
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise requests.RequestException("Max retries exceeded")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status dictionary
        
        Example:
            >>> client.health_check()
            {'status': 'healthy', 'model_loaded': True, 'device': 'cpu'}
        """
        response = self._request('GET', '/health')
        return response.json()
    
    def model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        
        Example:
            >>> client.model_info()
            {'architecture': 'MultimodalFusionModel', 'embed_dim': 256, ...}
        """
        response = self._request('GET', '/model-info')
        return response.json()
    
    def predict(
        self,
        wsi_features: Optional[Union[np.ndarray, List[List[float]]]] = None,
        genomic: Optional[Union[np.ndarray, List[float]]] = None,
        clinical_text: Optional[Union[np.ndarray, List[int]]] = None
    ) -> PredictionResult:
        """
        Make a single prediction.
        
        At least one modality must be provided.
        
        Args:
            wsi_features: WSI patch features [num_patches, 1024]
            genomic: Genomic features [2000]
            clinical_text: Tokenized clinical text [seq_len]
        
        Returns:
            PredictionResult object
        
        Raises:
            ValueError: If no modalities provided
            requests.RequestException: If request fails
        
        Example:
            >>> result = client.predict(
            ...     wsi_features=np.random.randn(50, 1024),
            ...     genomic=np.random.randn(2000)
            ... )
            >>> print(result.predicted_class)
        """
        # Validate input
        if all(x is None for x in [wsi_features, genomic, clinical_text]):
            raise ValueError("At least one modality must be provided")
        
        # Prepare request data
        data = {}
        
        if wsi_features is not None:
            if isinstance(wsi_features, np.ndarray):
                wsi_features = wsi_features.tolist()
            data['wsi_features'] = wsi_features
        
        if genomic is not None:
            if isinstance(genomic, np.ndarray):
                genomic = genomic.tolist()
            data['genomic'] = genomic
        
        if clinical_text is not None:
            if isinstance(clinical_text, np.ndarray):
                clinical_text = clinical_text.tolist()
            data['clinical_text'] = clinical_text
        
        # Make request
        response = self._request('POST', '/predict', json=data)
        result = response.json()
        
        return PredictionResult(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            available_modalities=result['available_modalities']
        )
    
    def batch_predict(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[Union[PredictionResult, Dict[str, str]]]:
        """
        Make predictions for multiple samples.
        
        Automatically splits into batches if needed.
        
        Args:
            samples: List of sample dictionaries with modality data
            batch_size: Maximum batch size (default: 32)
        
        Returns:
            List of PredictionResult objects or error dictionaries
        
        Example:
            >>> samples = [
            ...     {'genomic': np.random.randn(2000)},
            ...     {'genomic': np.random.randn(2000)}
            ... ]
            >>> results = client.batch_predict(samples)
            >>> for result in results:
            ...     print(result.predicted_class)
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            # Convert numpy arrays to lists
            batch_data = []
            for sample in batch:
                sample_data = {}
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        sample_data[key] = value.tolist()
                    else:
                        sample_data[key] = value
                batch_data.append(sample_data)
            
            # Make request
            try:
                response = self._request('POST', '/batch-predict', json=batch_data)
                batch_results = response.json()['predictions']
                
                # Convert to PredictionResult objects
                for result in batch_results:
                    if 'error' in result:
                        all_results.append(result)
                    else:
                        all_results.append(PredictionResult(
                            predicted_class=result['predicted_class'],
                            confidence=result['confidence'],
                            probabilities=result['probabilities'],
                            available_modalities=result['available_modalities']
                        ))
            
            except requests.RequestException as e:
                logger.error(f"Batch prediction failed: {e}")
                # Add error for each sample in batch
                for _ in batch:
                    all_results.append({'error': str(e)})
        
        return all_results
    
    def predict_from_file(
        self,
        file_path: str,
        modality: str = 'genomic'
    ) -> PredictionResult:
        """
        Make prediction from file.
        
        Args:
            file_path: Path to data file (numpy or text)
            modality: Modality type ('wsi', 'genomic', or 'clinical')
        
        Returns:
            PredictionResult object
        
        Example:
            >>> result = client.predict_from_file('data.npy', modality='genomic')
        """
        # Load data
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.npz'):
            data = np.load(file_path)[modality]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Make prediction
        kwargs = {modality: data}
        return self.predict(**kwargs)
    
    def close(self):
        """Close the session."""
        self.session.close()
        logger.info("API client session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncPathologyAPIClient:
    """
    Async client for Computational Pathology API.
    
    Requires aiohttp: pip install aiohttp
    
    Example:
        >>> import asyncio
        >>> 
        >>> async def main():
        ...     async with AsyncPathologyAPIClient("http://localhost:8000") as client:
        ...         result = await client.predict(genomic=data)
        ...         print(result)
        >>> 
        >>> asyncio.run(main())
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
        
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for async client. Install with: pip install aiohttp")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = self.aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.session.close()
    
    async def predict(
        self,
        wsi_features: Optional[Union[np.ndarray, List[List[float]]]] = None,
        genomic: Optional[Union[np.ndarray, List[float]]] = None,
        clinical_text: Optional[Union[np.ndarray, List[int]]] = None
    ) -> PredictionResult:
        """
        Make async prediction.
        
        Args:
            wsi_features: WSI patch features
            genomic: Genomic features
            clinical_text: Clinical text tokens
        
        Returns:
            PredictionResult object
        """
        # Prepare data
        data = {}
        if wsi_features is not None:
            data['wsi_features'] = wsi_features.tolist() if isinstance(wsi_features, np.ndarray) else wsi_features
        if genomic is not None:
            data['genomic'] = genomic.tolist() if isinstance(genomic, np.ndarray) else genomic
        if clinical_text is not None:
            data['clinical_text'] = clinical_text.tolist() if isinstance(clinical_text, np.ndarray) else clinical_text
        
        # Make request
        url = f"{self.base_url}/predict"
        async with self.session.post(url, json=data, timeout=self.timeout) as response:
            response.raise_for_status()
            result = await response.json()
        
        return PredictionResult(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            available_modalities=result['available_modalities']
        )


def main():
    """Example usage of the API client."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pathology API Client Example')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--test', action='store_true', help='Run test predictions')
    args = parser.parse_args()
    
    # Create client
    client = PathologyAPIClient(args.url)
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Health: {health}")
    print()
    
    # Get model info
    print("Getting model info...")
    info = client.model_info()
    print(f"Model: {info['architecture']}")
    print(f"Parameters: {info['total_parameters']:,}")
    print()
    
    if args.test:
        # Test prediction with synthetic data
        print("Making test prediction...")
        
        # Generate synthetic data
        wsi_features = np.random.randn(50, 1024)
        genomic = np.random.randn(2000)
        clinical_text = np.random.randint(0, 30000, size=100)
        
        result = client.predict(
            wsi_features=wsi_features,
            genomic=genomic,
            clinical_text=clinical_text
        )
        
        print(f"Result: {result}")
        print(f"Predicted class: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Probabilities: {[f'{p:.3f}' for p in result.probabilities]}")
        print()
        
        # Test batch prediction
        print("Making batch prediction...")
        samples = [
            {'genomic': np.random.randn(2000)},
            {'genomic': np.random.randn(2000)},
            {'genomic': np.random.randn(2000)}
        ]
        
        results = client.batch_predict(samples)
        print(f"Batch results: {len(results)} predictions")
        for i, result in enumerate(results):
            if isinstance(result, PredictionResult):
                print(f"  Sample {i+1}: class={result.predicted_class}, confidence={result.confidence:.3f}")
            else:
                print(f"  Sample {i+1}: error={result.get('error')}")
    
    client.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
