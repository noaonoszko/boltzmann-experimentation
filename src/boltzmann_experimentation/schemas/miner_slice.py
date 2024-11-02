from datetime import datetime
from pydantic import BaseModel, Field
import torch


class MinerSlice(BaseModel):
    miner_id: int = Field(..., description="Unique identifier for the miner")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of when the slice was uploaded",
    )
    data: torch.Tensor = Field(..., description="Slice of the model's parameters")

    class Config:
        arbitrary_types_allowed = True
