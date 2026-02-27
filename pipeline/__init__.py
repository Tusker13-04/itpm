"""
Pipeline package for the open-vocabulary vision system.

Contains:
- `state`  : Typed `VisionState` definition
- `tools`  : LangChain tools wrapping the underlying models
- `nodes`  : LangGraph node callables operating over `VisionState`
- `graph`  : StateGraph builder and compiled `pipeline`
"""

