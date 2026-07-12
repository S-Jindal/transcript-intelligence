from pathlib import Path

import typer
from openai import AsyncOpenAI

from transcript_intelligence.config import Settings
from transcript_intelligence.logging_setup import configure_logging, get_logger
from transcript_intelligence.pii import (
    PiiProcessor,
    PresidioDetector,
    RegexDetector,
)
from transcript_intelligence.pipeline import PipelineDependencies, run_pipeline
from transcript_intelligence.semantic import load_embedding_model


def main(
    input_directory: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_directory: Path = typer.Option(
        Path("executions"),
        "--output",
        file_okay=False,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run transcript intelligence on local dataset folders."""
    configure_logging(verbose)
    log = get_logger(__name__)
    settings = Settings()
    log.info(
        "pipeline_start",
        input=str(input_directory),
        output=str(output_directory),
        model=settings.llm_model,
    )
    dependencies = PipelineDependencies(
        pii_processor=PiiProcessor(
            regex_detector=RegexDetector(settings.account_pattern_list),
            ner_detector=PresidioDetector(settings.spacy_model),
            allowlist=settings.allowlist_terms,
        ),
        embedding_model=load_embedding_model(
            settings.embedding_model,
            settings.resolved_embedding_device(),
        ),
        llm_client=AsyncOpenAI(api_key=settings.openai_api_key),
    )
    result = run_pipeline(
        settings,
        input_directory,
        output_directory,
        dependencies,
    )
    log.info("done", execution=str(result.directory))


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
