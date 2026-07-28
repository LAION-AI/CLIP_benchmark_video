import time
import warnings

import torch
import torch.nn.functional as F

def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    amp=True,
    recall_k_list=(5,),
    query_ids=None,
):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use

    query_ids: iterable, optional
        Sample IDs to use as queries. All images/videos and texts still serve as
        retrieval candidates. IDs are WebDataset ``__key__`` values for
        WebDatasets and zero-based iteration indices for regular datasets.
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    sample_ids = []
    dataloader = dataloader_with_sample_ids(dataloader)
    
    data_loading_checkpoint = time.time()
    it = 0
    for batch_images, batch_texts, batch_ids in dataloader:
        batch_images = batch_images.to(device)

        data_loading_time = time.time() - data_loading_checkpoint
        
        forward_pass_checkpoint = time.time()
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        image_start = len(sample_ids)
        batch_image_indices = range(image_start, image_start + len(batch_images))
        batch_texts_image_index = [
            image_index
            for image_index, texts in zip(batch_image_indices, batch_texts)
            for _ in texts
        ]

        # compute the embedding of images and texts
        with torch.no_grad(), torch.autocast(device, enabled=amp):
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        sample_ids.extend(_batch_ids_to_list(batch_ids))
        forward_pass_time = time.time() - forward_pass_checkpoint
        
        data_loading_checkpoint = time.time()
        throughput = len(batch_images) / (data_loading_time + forward_pass_time)
        avg_throughput = avg_throughput + (throughput - avg_throughput ) / (it + 1) if it > 0 else throughput
        if it % 10 == 0:
            print(f"Data loading time: {data_loading_time:.3f}s, Forward pass time: {forward_pass_time:.3f}s, Throughput: {throughput:.1f} samples/s, Avg Throughput: {avg_throughput:.1f} samples/s")
        it += 1
        
    if not batch_images_emb_list:
        raise ValueError("Cannot evaluate retrieval on an empty dataloader")

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    texts_image_index = torch.as_tensor(texts_image_index, dtype=torch.long)

    if query_ids is None:
        # Text queries retrieve images/videos.
        image_retrieval_scores = texts_emb @ images_emb.t()
        image_retrieval_positive_pairs = torch.zeros_like(
            image_retrieval_scores, dtype=bool
        )
        image_retrieval_positive_pairs[
            torch.arange(len(image_retrieval_scores)), texts_image_index
        ] = True

        # Image/video queries retrieve texts.
        text_retrieval_scores = image_retrieval_scores.T
        text_retrieval_positive_pairs = image_retrieval_positive_pairs.T
    else:
        selected_image_indices = _resolve_query_indices(query_ids, sample_ids)
        selected_text_mask = torch.isin(texts_image_index, selected_image_indices)
        selected_text_indices = selected_text_mask.nonzero(as_tuple=False).flatten()
        if len(selected_text_indices) == 0:
            raise ValueError("None of the selected query samples has any text")

        # Texts belonging to selected IDs query the complete image/video gallery.
        image_retrieval_scores = (
            texts_emb[selected_text_indices] @ images_emb.T
        )
        image_retrieval_positive_pairs = torch.zeros_like(
            image_retrieval_scores, dtype=bool
        )
        image_retrieval_positive_pairs[
            torch.arange(len(selected_text_indices)),
            texts_image_index[selected_text_indices],
        ] = True

        # Selected images/videos query the complete text gallery.
        text_retrieval_scores = (
            images_emb[selected_image_indices] @ texts_emb.T
        )
        text_retrieval_positive_pairs = (
            selected_image_indices[:, None] == texts_image_index[None, :]
        )

    metrics = {}
    for recall_k in recall_k_list:
        if recall_k > len(images_emb):
            raise ValueError(
                f"recall@{recall_k} exceeds the image/video gallery size "
                f"({len(images_emb)})"
            )
        if recall_k > len(texts_emb):
            raise ValueError(
                f"recall@{recall_k} exceeds the text gallery size "
                f"({len(texts_emb)})"
            )
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
            batchify(
                recall_at_k,
                image_retrieval_scores,
                image_retrieval_positive_pairs,
                batch_size,
                device,
                k=recall_k,
            ) > 0
        ).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
            batchify(
                recall_at_k,
                text_retrieval_scores,
                text_retrieval_positive_pairs,
                batch_size,
                device,
                k=recall_k,
            ) > 0
        ).float().mean().item()

    return metrics


def dataloader_with_sample_ids(dataloader):
    start = 0
    for batch in dataloader:
        if len(batch) == 3:
            yield batch
            continue

        if len(batch) != 2:
            raise ValueError(
                "Retrieval batches must contain (media, texts) or "
                "(media, texts, sample_ids)"
            )
        x, y = batch
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def _batch_ids_to_list(batch_ids):
    if isinstance(batch_ids, torch.Tensor):
        return batch_ids.tolist()
    return list(batch_ids)


def _normalize_sample_id(sample_id):
    if isinstance(sample_id, bytes):
        sample_id = sample_id.decode("utf-8")
    if isinstance(sample_id, torch.Tensor):
        sample_id = sample_id.item()
    return str(sample_id)


def _resolve_query_indices(query_ids, sample_ids):
    requested_ids = {_normalize_sample_id(query_id) for query_id in query_ids}
    if not requested_ids:
        raise ValueError("query_ids must contain at least one sample ID")

    id_to_index = {}
    for index, sample_id in enumerate(sample_ids):
        normalized_id = _normalize_sample_id(sample_id)
        if normalized_id in id_to_index:
            raise ValueError(f"Duplicate sample ID in dataloader: {normalized_id!r}")
        id_to_index[normalized_id] = index

    missing_ids = requested_ids - id_to_index.keys()
    if missing_ids:
        preview = ", ".join(repr(value) for value in sorted(missing_ids)[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        warnings.warn(
            f"{len(missing_ids)} retrieval query ID(s) were not found: "
            f"{preview}{suffix}. Continuing with the IDs that are present.",
            UserWarning,
        )

    # Dataset order makes metric results deterministic regardless of query ID order.
    selected_indices = torch.as_tensor(
        [
            index
            for index, sample_id in enumerate(sample_ids)
            if _normalize_sample_id(sample_id) in requested_ids
        ],
        dtype=torch.long,
    )
    if len(selected_indices) == 0:
        raise ValueError("None of the requested retrieval query IDs were found")
    return selected_indices

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
