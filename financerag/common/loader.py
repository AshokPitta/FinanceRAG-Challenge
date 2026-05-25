import logging
from pathlib import Path
from typing import Optional, Tuple, cast

from datasets import Dataset, Value, load_dataset

logger = logging.getLogger(__name__)

class HFDataLoader:
    """
    A Hugging Face Dataset loader for corpus and query data. Supports loading datasets from local files
    (in JSONL format) or directly from a Hugging Face repository.
    """

    def __init__(
        self,
        hf_repo: Optional[str] = None,
        data_folder: Optional[str] = None,
        subset: Optional[str] = None,
        prefix: Optional[str] = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        keep_in_memory: bool = False,
    ):
        self.corpus: Optional[Dataset] = None
        self.queries: Optional[Dataset] = None
        self.hf_repo = hf_repo
        self.subset = subset

        if hf_repo:
            logger.warning(
                "A Hugging Face repository is provided. This will override the data_folder, prefix, and *_file arguments."
            )
        else:
            if (data_folder is None) or (subset is None):
                raise ValueError("A Hugging Face repository or local directory is required.")

            if prefix:
                query_file = prefix + "_" + query_file

            self.corpus_file = (Path(data_folder) / subset / corpus_file).as_posix()
            self.query_file = (Path(data_folder) / subset / query_file).as_posix()

        self.streaming = False
        self.keep_in_memory = keep_in_memory

    @staticmethod
    def check(file_in: str, ext: str):
        if not Path(file_in).exists():
            raise ValueError(f"File {file_in} not present! Please provide an accurate file.")

        if not file_in.endswith(ext):
            raise ValueError(f"File {file_in} must have the extension {ext}")

    def load(self) -> Tuple[Dataset, Dataset]:
        if not self.hf_repo:
            self.check(file_in=self.corpus_file, ext="jsonl")
            self.check(file_in=self.query_file, ext="jsonl")

        if self.corpus is None:
            logger.info("Loading Corpus...")
            self._load_corpus()
            self.corpus = cast(Dataset, self.corpus)
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Corpus Example: %s", self.corpus[0])

        if self.queries is None:
            logger.info("Loading Queries...")
            self._load_queries()
            self.queries = cast(Dataset, self.queries)
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", self.queries[0])

        return self.corpus, self.queries

    def load_corpus(self) -> Dataset:
        if not self.hf_repo:
            self.check(file_in=self.corpus_file, ext="jsonl")

        if self.corpus is None or not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            self.corpus = cast(Dataset, self.corpus)
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Corpus Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(
                path=self.hf_repo,
                name=self.subset,
                split="corpus",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                split="train",
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )

        corpus_ds = cast(Dataset, corpus_ds)

        # Optional cast/rename if _id exists
        if "_id" in corpus_ds.column_names:
            corpus_ds = corpus_ds.cast_column("_id", Value("string"))
            corpus_ds = corpus_ds.rename_column("_id", "id")

        corpus_ds = corpus_ds.remove_columns(
            [col for col in corpus_ds.column_names if col not in ["id", "text", "title"]]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                path=self.hf_repo,
                name=self.subset,
                split="queries",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                split="train",
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )

        queries_ds = cast(Dataset, queries_ds)

        # Optional cast/rename if _id exists
        if "_id" in queries_ds.column_names:
            queries_ds = queries_ds.cast_column("_id", Value("string"))
            queries_ds = queries_ds.rename_column("_id", "id")

        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds
