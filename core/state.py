from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    run_id: str = Field(default="", description="Unique identifier for this pipeline run")
    status: str = Field(default="initialized", description="Current pipeline status")

    uploaded_files: list[str] = Field(default_factory=list, description="Paths to uploaded CSV files")
    business_intent: str = Field(default="", description="User's business question in plain English")

    profile_path: str = Field(default="", description="Path to the data profile JSON")

    sttm_bronze_path: str = Field(default="", description="Path to Bronze layer STTM rules")
    sttm_silver_path: str = Field(default="", description="Path to Silver layer STTM rules")
    sttm_gold_path: str = Field(default="", description="Path to Gold layer STTM rules")
    hitl_approved: bool = Field(default=False, description="Whether human approved the STTM rules")

    bronze_output_paths: list[str] = Field(default_factory=list, description="Bronze Parquet output paths")
    silver_output_paths: list[str] = Field(default_factory=list, description="Silver Parquet output paths")
    gold_output_paths: list[str] = Field(default_factory=list, description="Gold Parquet output paths")

    report_path: str = Field(default="", description="Path to the final HTML report")

    error: str | None = Field(default=None, description="Error message if the pipeline failed")
