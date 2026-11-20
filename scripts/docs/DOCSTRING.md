# Docstring standards for PixelFlow

How to write the docstrings of PixelFlow's functions, classes and modules. Docstrings are read in
`help()` and in editors, and the API reference is generated from them (`generate_docs.py` in this
folder). They describe one function precisely; the guides that explain tasks live in `docs/` and
follow [DOCS-GUIDE.md](../../DOCS-GUIDE.md).

## Module-Level Documentation

**When to Include Module-Level Docstrings:**
- **Multi-function modules**: Modules containing multiple functions or classes
- **Complex modules**: Modules with intricate logic or multiple responsibilities
- **API entry points**: Main modules that serve as package interfaces

**Skip Module-Level Docstrings for:**
- **Single-function modules**: Simple modules containing only one function (like annotators)
- **Utility modules**: Simple helper modules where the function name clearly indicates purpose

For modules that need module-level documentation, start with:

```python
"""
[Module Name] for [primary purpose/use case].

[2-3 sentence description of the module's functionality, target use cases, 
and key differentiators. Focus on what problems it solves.]
"""

from typing import List, Optional, Union, Any
import [required imports]
from [internal imports]

__all__ = ["function1", "function2"]  # Export list for API docs
```

## Standard Template for Functions

Use this template for public functions, and leave out the sections that have nothing to say.

```python
def function_name(
    param1: Type1,
    param2: Optional[Type2] = None,
    param3: float = 0.05
) -> ReturnType:
    """
    [One-line summary of what the function does].
    
    [2-3 sentences providing more detailed explanation of the function's 
    purpose, approach, and key benefits. Include algorithmic approach if relevant.]
    
    Args:
        param1 (Type1): [Description with expected format/range/constraints]
        param2 (Optional[Type2]): [Description including default behavior].
                                  [Additional details on separate line if needed].
        param3 (float): [Description with range/constraints]. 
                       Range: [min-max]. Default is [value] ([meaning]).
        
    Returns:
        ReturnType: [Description of return value, including any side effects
                   like in-place modifications].
    
    Raises:
        ExceptionType1: [When this exception occurs]
        ExceptionType2: [When this exception occurs]
        
    Example:
        >>> import pixelflow as pf
        >>> from mayaku import from_pretrained
        >>>
        >>> # [Comment describing the setup]
        >>> predictor = from_pretrained("mayaku-n-det")
        >>> image = pf.read_image("street.jpg")  # RGB
        >>> result = predictor(image[:, :, ::-1])  # Mayaku reads arrays as BGR
        >>> detections = pf.from_mayaku(result, labels=predictor.class_names)
        >>>
        >>> # [Comment describing basic usage]
        >>> result = pf.module.function_name(param1, detections)
        >>>
        >>> # [Comment describing another option]
        >>> result = pf.module.function_name(param1, detections, param3=0.2)
    
    Notes:
        - [Implementation detail or behavior note]
        - [Memory/performance consideration]
        - [Limitation or constraint]
        - [Automatic behavior or adaptation]
        
    Performance Notes: (Optional - only for computationally intensive functions)
        - [Optimization details]
        - [Scaling characteristics]
        - [Efficiency considerations]
        
    See Also: (Optional - only when related functions exist)
        function_name : [Brief description of related function]
        module.other_function : [Brief description of alternative/complementary function]
    """
```

## Documentation Standards

### 1. **Args Section**
- Always include type annotations in both signature AND docstring
- Specify ranges, constraints, and default behaviors
- Use "Optional[Type]" for nullable parameters
- Include units where relevant (e.g., "in pixels", "as percentage")
- Explain what "None" means for optional parameters

### 2. **Examples Section**
- **Show the PixelFlow workflow** (for functions taking detections): model output, then
  `pf.from_*()`, then the function
- One to three examples: basic usage with defaults, then the options that matter most
- Every example runs as written: real function names and parameters, no `model` that is never
  defined, nothing removed from the API
- Read images with `pf.read_image` (RGB), never `cv2.imread` (BGR)
- Use realistic variable names and paths
- Add comments explaining each example's purpose

### 3. **Error Handling**
- Document all exceptions that can be raised
- Include both explicit raises and implicit errors (IndexError, AttributeError, etc.)
- Specify the conditions that trigger each exception

### 4. **Type Hints**
- Use complete type annotations for all parameters and returns
- Import typing modules: `from typing import List, Optional, Union, Any`
- Be specific: `List[Results]` instead of `List`
- Use `Optional[Type]` instead of `Union[Type, None]`
- Follow Python 3.10+ compatibility

### 5. **Cross-References** (Conditional)
- **Include "See Also" only when related functions actually exist**
- **Skip for isolated utilities** with no clear related functions
- Use format: `function_name : Description`
- Link to complementary, alternative, or prerequisite functions

### 6. **Implementation Notes**
- Say whether the input is modified in place or a new object is returned
- Explain automatic adaptations or validations
- Note any clamping, clipping, or constraint enforcement
- Explain default parameter selection logic

## PixelFlow-Specific Patterns

### Framework Adapter Pattern
Always show the two-step process in examples: the model's output, then a `pf.from_*()` converter.
Use Mayaku unless the function is about another framework (the docstring of `from_ultralytics`
uses Ultralytics):
```python
>>> import pixelflow as pf
>>> from mayaku import from_pretrained
>>> predictor = from_pretrained("mayaku-n-det")
>>> image = pf.read_image("street.jpg")  # RGB
>>> result = predictor(image[:, :, ::-1])  # Mayaku reads arrays as BGR
>>> detections = pf.from_mayaku(result, labels=predictor.class_names)
>>> image = pf.annotate.box(image, detections)
```

### Common Result Types
- Name the type: `detections: Detections`, not `results: List`
- Document expected attributes: "Each detection must have a `bbox`"
- Mention the coordinate format: `[x1, y1, x2, y2]` in pixels
- Images are RGB `uint8` arrays of shape `(height, width, 3)`

### Parameter Validation
Document automatic validation behaviors:
- Range clamping: "Automatically clamped to range [0.0, 1.0]"
- Format correction: "Automatically converted to odd number if even"
- Boundary checking: "Clipped to image boundaries"

## Quality Checklist

- [ ] Module docstring explains purpose and use cases
- [ ] Function signature has complete type hints
- [ ] Args section includes types, ranges, and constraints
- [ ] Examples use realistic code and variable names
- [ ] Raises section covers relevant exceptions
- [ ] No unused imports in typing section
- [ ] Follows Python 3.10+ compatibility requirements
- [ ] Examples show the PixelFlow workflow (model output, then `pf.from_*()`) when applicable
- [ ] Every example runs as written, and reads images with `pf.read_image`
- [ ] Notes section explains implementation details
- [ ] Performance notes included if computationally intensive
- [ ] See Also references related functions if they exist

Not every section belongs in every docstring: a short helper needs a summary line, `Args` and
`Returns`.