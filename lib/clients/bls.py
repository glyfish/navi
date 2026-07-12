"""Async client for the Bureau of Labor Statistics (BLS) Public Data API.

BLS differs from the FRED/Tiingo clients in three ways: multi-series reads use
POST with a JSON body, the API key travels in the body/query as
``registrationkey`` (and is optional — without it you get the lower v1 limits),
and the API returns HTTP 200 even on logical failures, signalling the outcome in
the ``status`` field. ``status != "REQUEST_SUCCEEDED"`` is raised as an error; a
successful response may still carry advisory ``message`` entries.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import httpx

from lib.env import get_bls_api_key, get_bls_base_url
from lib.clients.models.bls import BlsSeriesResponse, BlsSurveysResponse


class BlsAPIError(RuntimeError):
    """Raised when the BLS API returns an error response."""


class BlsClient:
    """Thin wrapper around the BLS Public Data API (v2)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        # Key is optional: fall back to the environment, tolerating its absence.
        self.api_key = api_key if api_key is not None else get_bls_api_key(required=False)
        self.base_url = (base_url or get_bls_base_url()).rstrip("/")
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> "BlsClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _check_status(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Raise on a non-success status; BLS reports failures with HTTP 200."""
        if payload.get("status") != "REQUEST_SUCCEEDED":
            messages = "; ".join(payload.get("message") or []) or str(payload.get("status"))
            raise BlsAPIError(f"BLS request failed: {messages}")
        return payload

    async def _get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        query: dict[str, Any] = dict(params or {})
        if self.api_key:
            query["registrationkey"] = self.api_key
        try:
            response = await self._client.get(path, params=query)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BlsAPIError(f"BLS request failed: {exc.response.text}") from exc
        return self._check_status(response.json())

    async def _post_timeseries(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        payload: dict[str, Any] = dict(body)
        if self.api_key:
            payload["registrationkey"] = self.api_key
        try:
            response = await self._client.post("/timeseries/data/", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BlsAPIError(f"BLS request failed: {exc.response.text}") from exc
        return self._check_status(response.json())

    async def get_series_data(
        self,
        series_ids: Sequence[str],
        *,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        catalog: bool = False,
        calculations: bool = False,
        annualaverage: bool = False,
        aspects: bool = False,
    ) -> BlsSeriesResponse:
        """Fetch observations for one or more series (POST /timeseries/data/)."""
        body: dict[str, Any] = {"seriesid": list(series_ids)}
        if start_year is not None:
            body["startyear"] = str(start_year)
        if end_year is not None:
            body["endyear"] = str(end_year)
        for name, flag in (
            ("catalog", catalog),
            ("calculations", calculations),
            ("annualaverage", annualaverage),
            ("aspects", aspects),
        ):
            if flag:
                body[name] = True
        data = await self._post_timeseries(body)
        return BlsSeriesResponse.model_validate(data)

    async def get_series_latest(self, series_id: str) -> BlsSeriesResponse:
        """Return the single most-recent datapoint for a series."""
        data = await self._get(f"/timeseries/data/{series_id}", {"latest": "true"})
        return BlsSeriesResponse.model_validate(data)

    async def get_popular_series(self, survey: Optional[str] = None) -> BlsSeriesResponse:
        """Return the 25 most popular series IDs, optionally within a survey."""
        params = {"survey": survey} if survey else None
        data = await self._get("/timeseries/popular", params)
        return BlsSeriesResponse.model_validate(data)

    async def get_all_surveys(self) -> BlsSurveysResponse:
        """Return the list of all BLS surveys."""
        data = await self._get("/surveys")
        return BlsSurveysResponse.model_validate(data)

    async def get_survey(self, survey_abbreviation: str) -> BlsSurveysResponse:
        """Return the metadata for a single BLS survey."""
        data = await self._get(f"/surveys/{survey_abbreviation}")
        return BlsSurveysResponse.model_validate(data)
