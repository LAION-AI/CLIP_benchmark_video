import torch
import pytest

from clip_benchmark.metrics import zeroshot_retrieval


class IdentityModel:
    def encode_image(self, images):
        return images

    def encode_text(self, texts):
        return texts


class EmbeddingTokenizer:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, texts):
        return torch.stack([self.embeddings[text] for text in texts])


def test_query_subset_changes_only_the_query_population():
    images = torch.eye(3)
    captions = [
        ["caption-0"],
        ["caption-1a", "caption-1b"],
        ["caption-2"],
    ]
    tokenizer = EmbeddingTokenizer(
        {
            "caption-0": torch.tensor([0.0, 0.8, 0.6]),
            "caption-1a": torch.tensor([0.0, 1.0, 0.0]),
            "caption-1b": torch.tensor([0.0, 1.0, 0.0]),
            "caption-2": torch.tensor([0.8, 0.6, 0.0]),
        }
    )
    dataloader = [(images, captions)]

    full_metrics = zeroshot_retrieval.evaluate(
        IdentityModel(),
        dataloader,
        tokenizer,
        device="cpu",
        amp=False,
        recall_k_list=[1],
    )
    subset_metrics = zeroshot_retrieval.evaluate(
        IdentityModel(),
        dataloader,
        tokenizer,
        device="cpu",
        amp=False,
        recall_k_list=[1],
        query_ids=[1],
    )

    assert full_metrics["image_retrieval_recall@1"] < 1.0
    assert full_metrics["text_retrieval_recall@1"] < 1.0
    assert subset_metrics == {
        "image_retrieval_recall@1": 1.0,
        "text_retrieval_recall@1": 1.0,
    }


def test_webdataset_keys_select_queries_but_keep_complete_galleries():
    images = torch.eye(2)
    captions = [["selected-caption"], ["distractor-caption"]]
    tokenizer = EmbeddingTokenizer(
        {
            # The selected text retrieves the unselected image.
            "selected-caption": torch.tensor([0.6, 0.8]),
            # The selected image retrieves the unselected text.
            "distractor-caption": torch.tensor([1.0, 0.0]),
        }
    )
    dataloader = [
        (images, captions, ["selected-key", "distractor-key"])
    ]

    metrics = zeroshot_retrieval.evaluate(
        IdentityModel(),
        dataloader,
        tokenizer,
        device="cpu",
        amp=False,
        recall_k_list=[1],
        query_ids=["selected-key"],
    )

    assert metrics == {
        "image_retrieval_recall@1": 0.0,
        "text_retrieval_recall@1": 0.0,
    }


def test_missing_query_ids_warn_and_present_ids_are_used():
    dataloader = [
        (
            torch.eye(2),
            [["caption-0"], ["caption-1"]],
            ["key-0", "key-1"],
        )
    ]
    tokenizer = EmbeddingTokenizer(
        {
            "caption-0": torch.tensor([1.0, 0.0]),
            "caption-1": torch.tensor([0.0, 1.0]),
        }
    )

    with pytest.warns(UserWarning, match="missing-key"):
        metrics = zeroshot_retrieval.evaluate(
            IdentityModel(),
            dataloader,
            tokenizer,
            device="cpu",
            amp=False,
            recall_k_list=[1],
            query_ids=["key-0", "missing-key"],
        )

    assert metrics == {
        "image_retrieval_recall@1": 1.0,
        "text_retrieval_recall@1": 1.0,
    }


def test_all_query_ids_missing_still_fails():
    dataloader = [
        (
            torch.eye(2),
            [["caption-0"], ["caption-1"]],
            ["key-0", "key-1"],
        )
    ]
    tokenizer = EmbeddingTokenizer(
        {
            "caption-0": torch.tensor([1.0, 0.0]),
            "caption-1": torch.tensor([0.0, 1.0]),
        }
    )

    with pytest.warns(UserWarning, match="missing-key"):
        with pytest.raises(ValueError, match="None of the requested"):
            zeroshot_retrieval.evaluate(
                IdentityModel(),
                dataloader,
                tokenizer,
                device="cpu",
                amp=False,
                recall_k_list=[1],
                query_ids=["missing-key"],
            )
