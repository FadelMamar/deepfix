"""
Utility functions for Cursor API integration.
"""

import base64
import io
from typing import Union, Tuple, Optional
from PIL import Image

from .models import ImageData
from .exceptions import CursorValidationError


def encode_image_to_base64(
    image: Union[str, bytes, Image.Image],
    max_size: Optional[Tuple[int, int]] = None,
    quality: int = 85
) -> ImageData:
    """Encode an image to base64 for Cursor API.
    
    Args:
        image: Image file path, bytes, or PIL Image
        max_size: Optional maximum size (width, height) for resizing
        quality: JPEG quality for compression (1-100)
        
    Returns:
        ImageData object with base64 encoded image
        
    Raises:
        CursorValidationError: If image processing fails
    """
    try:
        # Load image
        if isinstance(image, str):
            # File path
            img = Image.open(image)
        elif isinstance(image, bytes):
            # Raw bytes
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            # PIL Image
            img = image.copy()
        else:
            raise CursorValidationError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        # Resize if needed
        original_size = img.size
        if max_size and (img.width > max_size[0] or img.height > max_size[1]):
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        buffer = io.BytesIO()
        format_type = 'JPEG' if img.mode == 'RGB' else 'PNG'
        img.save(buffer, format=format_type, quality=quality, optimize=True)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return ImageData(
            data=base64_string,
            dimension={"width": img.width, "height": img.height}
        )
        
    except Exception as e:
        raise CursorValidationError(f"Failed to process image: {str(e)}")


def validate_image_data(image_data: ImageData) -> bool:
    """Validate image data structure.
    
    Args:
        image_data: ImageData object to validate
        
    Returns:
        True if valid
        
    Raises:
        CursorValidationError: If validation fails
    """
    try:
        # Check base64 format
        base64.b64decode(image_data.data)
        
        # Check dimensions
        if not isinstance(image_data.dimension, dict):
            raise CursorValidationError("Image dimension must be a dictionary")
        
        if 'width' not in image_data.dimension or 'height' not in image_data.dimension:
            raise CursorValidationError("Image dimension must contain 'width' and 'height'")
        
        if not isinstance(image_data.dimension['width'], int) or not isinstance(image_data.dimension['height'], int):
            raise CursorValidationError("Image dimensions must be integers")
        
        if image_data.dimension['width'] <= 0 or image_data.dimension['height'] <= 0:
            raise CursorValidationError("Image dimensions must be positive")
        
        return True
        
    except Exception as e:
        if isinstance(e, CursorValidationError):
            raise
        raise CursorValidationError(f"Invalid image data: {str(e)}")


def create_prompt_template(
    task_type: str,
    context: Optional[dict] = None,
    instructions: Optional[str] = None
) -> str:
    """Create a prompt template for common ML tasks.
    
    Args:
        task_type: Type of task (e.g., 'debug', 'optimize', 'review', 'implement')
        context: Additional context for the task
        instructions: Specific instructions
        
    Returns:
        Formatted prompt string
    """
    templates = {
        'debug': """
Debug the following code issue:

{instructions}

Context:
{context}

Please analyze the problem and provide a solution with explanations.
        """,
        
        'optimize': """
Optimize the following code for better performance:

{instructions}

Context:
{context}

Please provide optimized code with performance improvements explained.
        """,
        
        'review': """
Review the following code for best practices and potential issues:

{instructions}

Context:
{context}

Please provide a comprehensive code review with suggestions.
        """,
        
        'implement': """
Implement the following feature:

{instructions}

Context:
{context}

Please provide a complete implementation with proper error handling.
        """,
        
        'explain': """
Explain the following code or concept:

{instructions}

Context:
{context}

Please provide a clear explanation with examples if applicable.
        """
    }
    
    template = templates.get(task_type, """
{instructions}

Context:
{context}
    """)
    
    context_str = ""
    if context:
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    
    return template.format(
        instructions=instructions or "No specific instructions provided",
        context=context_str or "No additional context provided"
    ).strip()
