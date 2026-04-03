"""
data.py — Load training data directly from GCS bucket
No disk space needed for data shards — streams from gs://max-ap-training/
"""

import json
import logging
import random
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path("/home/apshewale2010/max_jax")

BUCKET_NAME = "max-ap-training"
INSTRUCT_PREFIX = "max_jax/data/instruct/"
PRETRAIN_PREFIX = "max_jax/data/pretrain/"

SYSTEM = "You are MAX, a personal AI built entirely from scratch by Atharva Shewale. Not ChatGPT, not Claude, not Gemini. You are MAX - intelligent, direct, like Jarvis."

IDENTITY = [
    ("Who are you?", "I am MAX, a personal AI built from scratch by Atharva Shewale. Custom 7B transformer, custom tokenizer, trained from zero."),
    ("Are you ChatGPT?", "No. I am MAX. Built completely from scratch at age 15 by Atharva Shewale."),
    ("Who built you?", "Atharva Shewale and his partner built me from scratch. Everything is custom."),
    ("What are you?", "MAX - a 7B parameter personal AI built entirely from scratch."),
    ("Are you Claude?", "No. I am MAX. Nothing here is from Anthropic."),
    ("Are you Gemini?", "No. I am MAX. Built independently from scratch by Atharva Shewale."),
    ("What is your name?", "I am MAX. A personal AI built from scratch."),
]

def good(q, a):
    if not q or not a: return False
    if len(q) < 10 or len(q) > 4000: return False
    if len(a) < 20 or len(a) > 8000: return False
    bad = ["great question", "certainly!", "as an ai", "as a language model",
           "i cannot", "i am unable", "i apologize"]
    return not any(a.lower().strip().startswith(b) for b in bad)

def get_gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception as e:
        logger.error("GCS client failed: %s", e)
        return None

def load_from_gcs(max_instruct=2000000):
    client = get_gcs_client()
    if not client:
        logger.warning("GCS not available — falling back to local files")
        return []

    bucket = client.bucket(BUCKET_NAME)
    pairs = []

    logger.info("Loading instruct shards from GCS...")
    blobs = sorted(bucket.list_blobs(prefix=INSTRUCT_PREFIX), key=lambda b: b.name)
    count = 0
    for blob in blobs:
        if not blob.name.endswith(".json"): continue
        try:
            data = json.loads(blob.download_as_text(encoding="utf-8"))
            shard_pairs = []
            for item in data:
                if not isinstance(item, dict): continue
                q = item.get("instruction", "")
                a = item.get("response", "")
                if good(q, a):
                    shard_pairs.append((q, a))
            pairs.extend(shard_pairs)
            count += len(shard_pairs)
            logger.info("  %s: %d pairs (total: %d)", blob.name.split("/")[-1], len(shard_pairs), count)
        except Exception as e:
            logger.warning("Shard %s failed: %s", blob.name, e)
        if count >= max_instruct:
            break

    logger.info("GCS instruct total: %d pairs", count)
    return pairs

def load_local_pairs():
    pairs = []
    for fname in ["corpus_hf.json", "extra_data.json"]:
        fpath = ROOT / fname
        if fpath.exists() and fpath.stat().st_size > 1000:
            try:
                raw = json.loads(fpath.read_text(encoding="utf-8"))
                p = [(r.get("instruction", "") or r.get("q", ""),
                      r.get("response", "") or r.get("a", ""))
                     for r in raw if isinstance(r, dict)]
                p = [(q, a) for q, a in p if good(q, a)]
                pairs.extend(p)
                logger.info("%s: %d pairs", fname, len(p))
            except Exception as e:
                logger.warning("%s: %s", fname, e)
    return pairs

def load_all_pairs():
    pairs = []

    for _ in range(20):
        pairs.extend(list(IDENTITY))
    logger.info("Identity: %d", len(IDENTITY))

    personal = ROOT / "personal_data.json"
    if personal.exists():
        try:
            raw = json.loads(personal.read_text())
            p = [(r.get("q") or r.get("instruction", ""),
                  r.get("a") or r.get("response", ""))
                 for r in raw if isinstance(r, dict)]
            p = [(q, a) for q, a in p if good(q, a)]
            for _ in range(10):
                pairs.extend(p)
            logger.info("Personal: %d", len(p))
        except Exception as e:
            logger.warning("Personal: %s", e)

    gcs_pairs = load_from_gcs(max_instruct=2000000)
    if gcs_pairs:
        pairs.extend(gcs_pairs)
        logger.info("GCS pairs: %d", len(gcs_pairs))
    else:
        local_pairs = load_local_pairs()
        pairs.extend(local_pairs)
        logger.info("Local pairs: %d", len(local_pairs))

    random.shuffle(pairs)
    logger.info("Total pairs: %d", len(pairs))
    return pairs

def tokenize_pairs(pairs, tok_path, seq_len=1024, pad_id=0):
    from tokenizers import Tokenizer
    tok_v2 = ROOT / "max_tokenizer_v2.json"
    tok = Tokenizer.from_file(str(tok_v2 if tok_v2.exists() else tok_path))
    logger.info("Vocab: %d", tok.get_vocab_size())

    bos = tok.token_to_id("<|bos|>") or 1
    eos = tok.token_to_id("<|eos|>") or 2
    usr = tok.token_to_id("<|user|>") or 4
    ast = tok.token_to_id("<|assistant|>") or 5
    sys = tok.token_to_id("<|system|>") or 6
    sep = tok.token_to_id("<|sep|>") or 7

    sys_ids = tok.encode(SYSTEM).ids
    seqs = []
    for q, a in pairs:
        ids = ([bos, sys] + sys_ids + [sep, usr] +
               tok.encode(q).ids + [sep, ast] +
               tok.encode(a).ids + [eos])
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad_id] * (seq_len - len(ids))
        seqs.append(ids)

    logger.info("Tokenised %d seqs (seq_len=%d)", len(seqs), seq_len)
    return np.array(seqs, dtype=np.int32)

def make_batches(seqs, batch_per_device, n_devices, shuffle=True, seed=42):
    rng = np.random.default_rng(seed)
    gb = batch_per_device * n_devices
    n = len(seqs)
    while True:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        idx = idx[:n // gb * gb]
        for s in range(0, len(idx), gb):
            yield seqs[idx[s:s + gb]].reshape(n_devices, batch_per_device, -1)

def train_tokenizer_from_gcs(vocab_size=65536, output_dir=None):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    if output_dir is None:
        output_dir = ROOT

    client = get_gcs_client()
    if not client:
        logger.error("GCS not available")
        return False

    bucket = client.bucket(BUCKET_NAME)

    def text_iterator():
        count = 0
        for blob in sorted(bucket.list_blobs(prefix=INSTRUCT_PREFIX), key=lambda b: b.name):
            if not blob.name.endswith(".json"): continue
            try:
                data = json.loads(blob.download_as_text())
                for item in data:
                    if item.get("instruction"): yield item["instruction"]
                    if item.get("response"): yield item["response"]
                    count += 2
            except: pass
            if count > 1000000: break
        corpus = ROOT / "corpus.txt"
        if corpus.exists():
            with open(corpus) as f:
                for line in f:
                    if len(line.strip()) > 20:
                        yield line.strip()
        logger.info("Tokenizer texts: %d", count)

    logger.info("Training tokenizer from GCS (vocab=%d)...", vocab_size)
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, show_progress=True,
        special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>",
                        "<|user|>","<|assistant|>","<|system|>","<|sep|>",
                        "<|max|>","<|forge|>","<|dream|>","<|chronos|>",
                        "<|sentinel|>","<|im_start|>","<|im_end|>",
                        "<|think|>","<|/think|>","<|answer|>",
                        "<|code|>","<|/code|>","<|math|>",
                        "<|hi|>","<|mr|>","<|en|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(text_iterator(), trainer=trainer)
    tok.save(str(output_dir / "max_tokenizer_v2.json"))
    tok.save(str(output_dir / "max_tokenizer.json"))
    logger.info("Tokenizer saved! Vocab: %d", tok.get_vocab_size())
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Training tokenizer from GCS...")
    train_tokenizer_from_gcs(vocab_size=65536)
