from __future__ import annotations

import os
import subprocess
from pathlib import Path

import luigi


class WorkspacePreparation(luigi.Task):
    workspace_dirpath = luigi.PathParameter()

    def run(self):
        os.makedirs(self.workspace_dirpath, exist_ok=True)

    def output(self):
        return luigi.LocalTarget(self.workspace_dirpath)


class FeatureExtraction(luigi.Task):
    input_dirpath = luigi.PathParameter()
    feature_type = luigi.Parameter()

    def requires(self):
        return WorkspacePreparation()

    def output(self):
        return luigi.LocalTarget(self.input().path)

    def run(self):
        subprocess.check_call(
            [
                "mvgkit_extract_features",
                "extract",
                "--input-dirpath",
                str(self.input_dirpath / "features"),
                "--output-dirpath",
                str(self.output().path),
                "--feature-type",
                self.feature_type.upper(),
            ]
        )


class FeatureMatching(luigi.Task):
    match_mode = luigi.Parameter()

    def requires(self):
        return FeatureExtraction()

    def output(self):
        return luigi.LocalTarget(self.input().path)

    def run(self):
        feature_input_dirpath = Path(self.input().path) / "features"
        subprocess.check_call(
            [
                "mvgkit_match_features",
                "match",
                "--feature-dirpath",
                feature_input_dirpath,
                "--output-dirpath",
                self.output().path,
                "--match-mode",
                self.match_mode,
            ]
        )
