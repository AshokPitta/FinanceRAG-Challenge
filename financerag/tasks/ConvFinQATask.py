from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class ConvFinQA(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="ConvFinQA",
            description="ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering",
            reference="https://github.com/czyssrs/ConvFinQA",
            dataset={
                "path": "/mnt/parscratch/users/acp24akp/FinanceRAG/FinanceRAG/dataset/ConvFinQA",
                "subset": ".",
                "corpus_file": "ConvFinQA_corpus_convert.csv",
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
            license="mit",
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation="""
                @article{chen2022convfinqa,
                    title={ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering},
                    author={Chen, Zhiyu and Li, Shiyang and Smiley, Charese and Ma, Zhiqiang and Shah, Sameena and Wang, William Yang},
                    journal={Proceedings of EMNLP 2022},
                    year={2022}
                }
            """,
        )
        super().__init__(self.metadata)
