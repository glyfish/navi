"""Typed models describing common FRED API payloads."""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class FredBaseModel(BaseModel):
    """Base config for immutable FRED models."""

    model_config = {"frozen": True}


class Pagination(FredBaseModel):
    realtime_start: date = Field(alias="realtime_start")
    realtime_end: date = Field(alias="realtime_end")
    order_by: Optional[str] = None
    sort_order: Optional[str] = None
    count: int
    offset: int
    limit: int


class Category(FredBaseModel):
    id: int = Field(alias="id")
    name: str
    parent_id: int | None = Field(default=None, alias="parent_id")


class CategoryResponse(Pagination):
    categories: List[Category]


class Series(FredBaseModel):
    id: str
    title: str
    observation_start: date
    observation_end: date
    frequency: str
    frequency_short: Optional[str] = None
    units: str
    units_short: Optional[str] = None
    seasonal_adjustment: Optional[str] = None
    seasonal_adjustment_short: Optional[str] = None
    last_updated: datetime
    popularity: Optional[int] = None
    notes: Optional[str] = None


class SeriesResponse(Pagination):
    seriess: List[Series]


class Observation(FredBaseModel):
    realtime_start: date
    realtime_end: date
    date: date
    value: str


class ObservationsResponse(Pagination):
    observations: List[Observation]


class Release(FredBaseModel):
    id: int
    name: str
    press_release: bool = Field(alias="press_release")
    link: Optional[str] = None
    realtime_start: date
    realtime_end: date


class ReleasesResponse(Pagination):
    releases: List[Release]
