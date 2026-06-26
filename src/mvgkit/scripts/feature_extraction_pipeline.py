"""Luigi pipeline that extracts features from images and matches them.

DAG::

    WorkspacePreparation -> FeatureExtraction -> FeatureMatching

Each task writes into its own sub-directory of the workspace so that Luigi's
directory-existence completion check is meaningful:

    <workspace>/            (WorkspacePreparation)
    <workspace>/features/   (FeatureExtraction, *.feat files)
    <workspace>/matches/    (FeatureMatching, *.match files)
"""

from __future__ import annotations

import os
import subprocess

import click
import luigi


class WorkspacePreparation(luigi.Task):
    workspace_dirpath = luigi.PathParameter()

    def run(self):
        os.makedirs(self.workspace_dirpath, exist_ok=True)

    def output(self):
        return luigi.LocalTarget(str(self.workspace_dirpath))


class FeatureExtraction(luigi.Task):
    input_dirpath = luigi.PathParameter()
    workspace_dirpath = luigi.PathParameter()
    feature_type = luigi.Parameter(default="SIFT")

    def requires(self):
        return WorkspacePreparation(workspace_dirpath=self.workspace_dirpath)

    def output(self):
        return luigi.LocalTarget(str(self.workspace_dirpath / "features"))

    def run(self):
        subprocess.check_call(
            [
                "mvgkit_extract_features",
                "extract",
                "--input-dirpath",
                str(self.input_dirpath),
                "--output-dirpath",
                self.output().path,
                "--feature-type",
                str(self.feature_type).upper(),
            ]
        )


class FeatureMatching(luigi.Task):
    input_dirpath = luigi.PathParameter()
    workspace_dirpath = luigi.PathParameter()
    feature_type = luigi.Parameter(default="SIFT")
    match_mode = luigi.Parameter(default="sequential")

    def requires(self):
        return FeatureExtraction(
            input_dirpath=self.input_dirpath,
            workspace_dirpath=self.workspace_dirpath,
            feature_type=self.feature_type,
        )

    def output(self):
        return luigi.LocalTarget(str(self.workspace_dirpath / "matches"))

    def run(self):
        subprocess.check_call(
            [
                "mvgkit_match_features",
                "match",
                "--feature-dirpath",
                self.input().path,
                "--output-dirpath",
                self.output().path,
                "--match-mode",
                str(self.match_mode),
            ]
        )


@click.command()
@click.option(
    "--input-dirpath",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the input images.",
)
@click.option(
    "--workspace-dirpath",
    required=True,
    type=click.Path(file_okay=False),
    help="Output workspace; features/ and matches/ are written under it.",
)
@click.option("--feature-type", default="SIFT", show_default=True, help="Feature type (SIFT/ORB).")
@click.option(
    "--match-mode",
    default="sequential",
    show_default=True,
    type=click.Choice(["sequential", "exhaustive"]),
    help="Feature matching mode.",
)
@click.option("--workers", default=1, show_default=True, help="Number of Luigi workers.")
def cli(input_dirpath, workspace_dirpath, feature_type, match_mode, workers):
    """Run the feature extraction and matching pipeline."""
    success = luigi.build(
        [
            FeatureMatching(
                input_dirpath=input_dirpath,
                workspace_dirpath=workspace_dirpath,
                feature_type=feature_type,
                match_mode=match_mode,
            )
        ],
        local_scheduler=True,
        workers=workers,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    cli()
