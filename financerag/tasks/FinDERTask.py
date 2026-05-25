from typing import Optional

from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class FinDER(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="FinDER",
            description="Prepared for competition from Linq",
            reference=None,
            dataset={
                "path": "/mnt/parscratch/users/acp24akp/FinanceRAG/FinanceRAG/dataset/FinDER",
                "subset": ".",
                "corpus_file": "corpus_prep.jsonl",
                "queries_file": "queries_prep.jsonl",
                "qrels_file": "FinDER_qrels.tsv"
            },
            type="RAG",
            category="s2p",
            modalities=["text"],
            date=None,
            domains=["Report"],
            task_subtypes=[
                "Financial retrieval",
                "Question answering",
            ],
            license=None,
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation=None,
        )
        super().__init__(self.metadata)


